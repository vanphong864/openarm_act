#!/usr/bin/env python3
"""Hiệu chuẩn intrinsics + extrinsics của các camera RGB-D cố định CHỈ từ dữ liệu ghi và mô hình robot.

1. Khởi tạo (DLT + RANSAC): khi kẹp đang giữ hộp kẹo, tâm hộp ≈ TCP (FK từ qpos). Cặp (TCP 3D, tâm vùng xanh 2D)
   trên nhiều episode cho ma trận chiếu P = K[R|t], tách ra K, R, t.
2. Tinh chỉnh (ICP có mô hình): chiếu ngược depth (đã căn theo RGB) bằng K, đưa sang hệ thế giới, khớp điểm-tới-mặt
   với bề mặt mesh hiển thị của robot đặt đúng qpos của từng khung. Ẩn: R, t, fx, fy, cx, cy và hệ số thang depth.
   Pixel xanh (hộp kẹo, không có trong mô hình robot) bị loại; mất mát Huber để bỏ qua người/cáp/bàn sát robot.
3. Báo cáo sai số trên khung của các episode KHÔNG dùng khi fit.

Không dùng số đo của scene.yaml. Sai số còn lại gồm cả sai lệch giữa URDF và robot thật (zero encoder, độ võng).

Ví dụ:
  calibrate_cameras.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --blobs outputs/sim_20261013/tcp_blobs.json --out simulation/realdata/cameras_20261013.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openarm_sim.robot_surface import RobotSurface  # noqa: E402

ROBOT = Path(__file__).resolve().parent.parent / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"
W, H = 640, 480


# ------------------------------------------------------------------ DLT

def _normalize(x):
    c = x.mean(0)
    s = np.sqrt(x.shape[1]) / np.linalg.norm(x - c, axis=1).mean()
    T = np.eye(x.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return T


def dlt(X, x):
    TX, Tx = _normalize(X), _normalize(x)
    Xn = (TX @ np.c_[X, np.ones(len(X))].T).T
    xn = (Tx @ np.c_[x, np.ones(len(x))].T).T
    A = []
    for P, p in zip(Xn, xn):
        A.append(np.r_[np.zeros(4), -p[2] * P, p[1] * P])
        A.append(np.r_[p[2] * P, np.zeros(4), -p[0] * P])
    Pn = np.linalg.svd(np.array(A))[2][-1].reshape(3, 4)
    return np.linalg.inv(Tx) @ Pn @ TX


def project(P, X):
    h = (P @ np.c_[X, np.ones(len(X))].T).T
    return h[:, :2] / h[:, 2:3]


def dlt_ransac(X, x, iters=3000, thresh=12.0, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(X), 6, replace=False)
        try:
            P = dlt(X[idx], x[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(project(P, X) - x, axis=1)
        inl = err < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    P = dlt(X[best], x[best])
    for _ in range(3):
        best = np.linalg.norm(project(P, X) - x, axis=1) < thresh
        P = dlt(X[best], x[best])
    return P, best


def kabsch(A, B):
    """R, C với B ≈ (A − C) Rᵀ … tức A (thế giới) → B (camera): b = R (a − C)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, ca - R.T @ cb


def kabsch_ransac(A, B, thresh=0.03, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i = rng.choice(len(A), 3, replace=False)
        R, C = kabsch(A[i], B[i])
        inl = np.linalg.norm((A - C) @ R.T - B, axis=1) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(3):
        R, C = kabsch(A[best], B[best])
        best = np.linalg.norm((A - C) @ R.T - B, axis=1) < thresh
    return R, C, best


def init_from_tin_depth(rows, cam, focal_grid=range(360, 761, 10)):
    """Khởi tạo bằng khớp 3D–3D: tâm vùng xanh chiếu ngược theo depth (hệ camera) ↔ TCP (thế giới).

    Thử lưới tiêu cự (fx = fy, tâm ảnh ở giữa); tiêu cự sai làm lệch thang nên sai số Kabsch lớn hơn.
    """
    A = np.array([r["tcp"] for r in rows])
    uvz = np.array([r[cam][3:6] for r in rows])
    best = None
    for f in focal_grid:
        B = np.c_[(uvz[:, 0] - W / 2) / f * uvz[:, 2], (uvz[:, 1] - H / 2) / f * uvz[:, 2], uvz[:, 2]]
        R, C, inl = kabsch_ransac(A, B)
        e = np.linalg.norm((A - C) @ R.T - B, axis=1)
        score = np.median(e[inl]) / np.sqrt(inl.mean())
        if best is None or score < best[0]:
            best = (score, f, R, C, inl, e)
    _, f, R, C, inl, e = best
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K, R, C, inl, e


def decompose(P):
    K, R, C = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]
    C = (C[:3] / C[3]).ravel()
    if np.linalg.det(R) < 0:
        R = -R
    # Bảo đảm điểm nằm trước camera (z > 0)
    return K, R, C


# ------------------------------------------------------------------ ICP

def rodrigues(w):
    th = torch.linalg.norm(w) + 1e-12
    k = w / th
    Kx = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=w.dtype, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


class CamParams:
    """Camera-from-world: p_c = R (p_w − C), R = exp(w) R0; fy = aspect · fx (aspect cố định: mô hình cảm biến).

    θ = [w(3), C(3), log fx, cx/100, cy/100, log thang depth]. Thang depth mặc định cố định = 1: để tự do thì ICP một
    chiều (dữ liệu → mô hình) co đám điểm lại cho nhiều điểm lọt ngưỡng hơn (nghiệm suy biến), không phải thông tin.
    """

    def __init__(self, K, R, C, dev, aspect=1.0, fit_scale=False, fit_principal=False, fit_aspect=False):
        f64 = dict(dtype=torch.float64, device=dev)
        self.R0 = torch.tensor(R, **f64)
        self.theta = torch.tensor([0, 0, 0, *C, np.log(K[0, 0]), K[0, 2] / 100, K[1, 2] / 100, 0.0,
                                   np.log(aspect)], **f64)
        self.free = torch.ones(11, dtype=torch.bool, device=dev)
        self.free[10] = fit_aspect
        self.free[9] = fit_scale
        # Tâm quang học: robot chỉ chiếm một phần ảnh nên (cx, cy) bù trừ với góc quay camera (không định danh
        # được, đã thấy cy chạy ra ngoài ảnh). Mặc định cố định ở tâm ảnh; D435 lệch tâm vài px.
        self.free[7] = self.free[8] = fit_principal

    def R(self, th=None):
        th = self.theta if th is None else th
        return rodrigues(th[:3]) @ self.R0

    @property
    def C(self):
        return self.theta[3:6]

    def backproject(self, u, v, z, th=None):
        th = self.theta if th is None else th
        fx = th[6].exp()
        fy = fx * th[10].exp()
        z = z * th[9].exp()
        pc = torch.stack([(u - 100 * th[7]) / fx * z, (v - 100 * th[8]) / fy * z, z], -1)
        return pc @ self.R(th) + th[3:6]        # = Rᵀ p_c + C theo hàng

    def export(self):
        th = self.theta.cpu().numpy()
        fx = float(np.exp(th[6]))
        return {"fx": fx, "fy": fx * float(np.exp(th[10])), "cx": float(100 * th[7]), "cy": float(100 * th[8]),
                "aspect_fy_over_fx": float(np.exp(th[10])), "aspect_fitted": bool(self.free[10]), "depth_scale_factor": float(np.exp(th[9])),
                "R_cam_from_world": self.R().cpu().numpy().tolist(), "C_world": th[3:6].tolist()}


def green_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 50) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80)).astype(np.uint8)
    return cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0


def trajectory_frames(raw, eps, arm_of, surf, K, R, C, per_episode, rng):
    """Khung rải đều theo thời gian mà TCP tay đang thao tác chiếu vào trong ảnh (theo camera khởi tạo)."""
    picks = {}
    for ep in eps:
        with h5py.File(Path(raw) / f"{ep}.hdf5", "r") as f:
            q = f["observations/qpos"][:]
        ts = np.arange(0, len(q), 5)
        P = np.array([surf.tcp_pos(q[t], arm_of[ep]) for t in ts])
        pc = (P - C) @ R.T
        u, v = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        ok = (pc[:, 2] > 0.25) & (u > 30) & (u < W - 30) & (v > 30) & (v < H - 30)
        cand = ts[ok]
        if len(cand):
            picks[ep] = sorted(rng.choice(cand, min(per_episode, len(cand)), replace=False).tolist())
    return picks


def load_frames(raw, picks, cam, stride):
    frames = []
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    for ep, ts in picks.items():
        with h5py.File(Path(raw) / f"{ep}.hdf5", "r") as f:
            for t in ts:
                z = f[f"observations/images/{cam}_depth"][t][::stride, ::stride].astype(np.float64) * 1e-3
                g = green_mask(f[f"observations/images/{cam}_rgb"][t])[::stride, ::stride]
                q = f["observations/qpos"][t]
                ok = (z > 0.2) & (z < 1.2) & ~g
                frames.append({"ep": ep, "t": int(t), "q": q, "u": uu[ok].astype(np.float64),
                               "v": vv[ok].astype(np.float64), "z": z[ok]})
    return frames


def nearest(src, dst, chunk=1024):
    """Chỉ số và khoảng cách điểm dst gần nhất cho từng src (GPU, chia khối)."""
    if len(dst) == 0:
        return (torch.zeros(len(src), dtype=torch.long, device=src.device),
                torch.full((len(src),), float("inf"), device=src.device))
    idx, dist = [], []
    for i in range(0, len(src), chunk):
        dd = torch.cdist(src[i:i + chunk], dst)
        m = dd.min(1)
        idx.append(m.indices)
        dist.append(m.values)
    return torch.cat(idx), torch.cat(dist)


def icp(cam, frames, surf, dev, schedule=((0.06, 3), (0.03, 3), (0.015, 4), (0.008, 4)), lm_iters=6, log=print):
    """ICP điểm-tới-mặt; với tương ứng cố định giải bằng Levenberg–Marquardt có trọng số Huber (IRLS)."""
    robot = []
    for fr in frames:
        p, n = surf.world(fr["q"])
        robot.append((torch.tensor(p, device=dev, dtype=torch.float32), torch.tensor(n, device=dev, dtype=torch.float32)))
    data = [tuple(torch.tensor(fr[k], device=dev, dtype=torch.float64) for k in ("u", "v", "z")) for fr in frames]
    free = cam.free
    lam = 1e-3
    for tau, rounds in schedule:
        for r in range(rounds):
            U, V, Z, S, N = [], [], [], [], []
            with torch.no_grad():
                Cw = cam.C.float()
                for (u, v, z), (p, n) in zip(data, robot):
                    pw = cam.backproject(u, v, z).float()
                    lo, hi = pw.min(0).values - tau, pw.max(0).values + tau
                    facing = (((Cw - p) * n).sum(1) > 0) & ((p > lo) & (p < hi)).all(1)
                    ps, ns = p[facing], n[facing]
                    i, dist = nearest(pw, ps)
                    keep = dist < tau
                    U.append(u[keep]), V.append(v[keep]), Z.append(z[keep])
                    S.append(ps[i[keep]].double()), N.append(ns[i[keep]].double())
            U, V, Z, S, N = (torch.cat(x) for x in (U, V, Z, S, N))

            def res(th_free):
                th = cam.theta.clone()
                th[free] = th_free
                return ((cam.backproject(U, V, Z, th) - S) * N).sum(1)

            for _ in range(lm_iters):
                x0 = cam.theta[free].clone()
                e = res(x0)
                # Jacobian N×9 bằng sai phân trung tâm (9 tham số: rẻ hơn autograd ngược với N lớn)
                cols = []
                for k in range(len(x0)):
                    h = torch.zeros_like(x0)
                    h[k] = 1e-6
                    cols.append((res(x0 + h) - res(x0 - h)) / 2e-6)
                J = torch.stack(cols, 1)
                a = e.abs() / (0.3 * tau)
                w = torch.where(a <= 1, torch.ones_like(a), 1 / a)         # Huber IRLS, ngưỡng 0.3·tau
                JtW = J.T * w
                Hm, g = JtW @ J, JtW @ e
                cost0 = float((w * e * e).sum())
                while True:
                    dx = -torch.linalg.solve(Hm + lam * torch.diag(torch.diag(Hm)), g)
                    e1 = res(x0 + dx)
                    if float((w * e1 * e1).sum()) < cost0 or lam > 1e6:
                        lam = max(lam / 3, 1e-7)
                        break
                    lam *= 4
                cam.theta[free] = x0 + dx
            ex = cam.export()
            log(f"  tau={tau * 1000:.0f}mm vòng {r}: {len(e)} điểm; fx={ex['fx']:.2f} fy={ex['fy']:.2f} "
                f"cx={ex['cx']:.2f} cy={ex['cy']:.2f} s={ex['depth_scale_factor']:.4f} "
                f"C={np.round(ex['C_world'], 4)} |e| trung vị {float(e.abs().median()) * 1000:.2f}mm")
    return cam


@torch.no_grad()
def residuals(cam, frames, surf, dev, tau=0.02):
    """Khoảng cách điểm-tới-mặt (mm) của pixel depth gần robot (< tau)."""
    out = []
    for fr in frames:
        p, n = surf.world(fr["q"])
        p = torch.tensor(p, device=dev, dtype=torch.float32)
        n = torch.tensor(n, device=dev, dtype=torch.float32)
        f = lambda k: torch.tensor(fr[k], device=dev, dtype=torch.float64)  # noqa: E731
        pw = cam.backproject(f("u"), f("v"), f("z")).float()
        facing = ((cam.C.float() - p) * n).sum(1) > 0
        i, dist = nearest(pw, p[facing])
        keep = dist < tau
        e = ((pw[keep] - p[facing][i[keep]]) * n[facing][i[keep]]).sum(1).abs()
        out.append(e.cpu().numpy() * 1000)
    e = np.concatenate(out)
    if not len(e):
        return {"points": 0}
    return {"points": int(len(e)), "median_mm": float(np.median(e)), "p90_mm": float(np.percentile(e, 90)),
            "mean_mm": float(e.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", nargs="+", default=["chest", "front"])
    ap.add_argument("--frames-per-camera", type=int, default=120)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aspect", type=float, nargs="+", default=[1.0, 4 / 3],
                    help="các giả thiết fy/fx thử và so trên episode giữ lại (1 = pixel vuông, 4/3 = khung 16:9 nén "
                         "ngang về 640x480)")
    ap.add_argument("--fit-depth-scale", action="store_true")
    ap.add_argument("--fit-principal", action="store_true")
    ap.add_argument("--fit-aspect", action="store_true", help="fit tự do fy/fx (giá trị --aspect là điểm xuất phát)")
    ap.add_argument("--exclude-static", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--frame-source", choices=["trajectory", "hold"], default="trajectory",
                    help="trajectory: rải đều theo cả episode, khung có TCP trong ảnh; hold: chỉ pha giữ vật")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    blobs = json.load(open(args.blobs))
    episodes = sorted({b["episode"] for b in blobs})
    held = set(rng.choice(episodes, max(2, len(episodes) // 5), replace=False).tolist())
    # Chỉ các link chuyển động: phần đứng yên (thân, đế vai) có mặt ở mọi khung, nếu khác URDF (giá gắn camera,
    # cáp) sẽ kéo lệch nghiệm với trọng số rất lớn.
    static = ("openarm_body_link0", "openarm_left_link0", "openarm_right_link0")
    surf = RobotSurface(str(ROBOT), spacing_m=0.005, exclude_bodies=static if args.exclude_static else ())
    result = {"method": __doc__.split("\n")[0], "robot_model": str(ROBOT.relative_to(ROBOT.parents[2])),
              "image_size": [W, H], "heldout_episodes": sorted(held), "cameras": {}}
    for cam in args.cameras:
        rows = [b for b in blobs if b[cam] is not None and len(b[cam]) >= 7 and np.isfinite(b[cam]).all()]
        fit_rows = [b for b in rows if b["episode"] not in held]
        K, R, C, inl, e = init_from_tin_depth(fit_rows, cam)
        print(f"[{cam}] khởi tạo 3D–3D: f={K[0, 0]:.0f}px, {inl.sum()}/{len(fit_rows)} inlier (<3cm), "
              f"sai số trung vị {np.median(e[inl]) * 1000:.1f}mm; C={np.round(C, 3)}; "
              f"hướng nhìn={np.round(R[2], 3)}")
        if args.frame_source == "hold":
            pool = [b for b in rows if b["episode"] not in held]
            pick = rng.choice(len(pool), min(args.frames_per_camera, len(pool)), replace=False)
            picks = {}
            for i in pick:
                picks.setdefault(pool[i]["episode"], []).append(pool[i]["t"])
            hpool = [b for b in rows if b["episode"] in held]
            hpick = rng.choice(len(hpool), min(16, len(hpool)), replace=False)
            hpicks = {}
            for i in hpick:
                hpicks.setdefault(hpool[i]["episode"], []).append(hpool[i]["t"])
        else:
            arm_of = {b["episode"]: b["arm"] for b in blobs}
            fit_eps = [e for e in episodes if e not in held]
            per = max(1, int(np.ceil(args.frames_per_camera / len(fit_eps))))
            picks = trajectory_frames(args.raw, fit_eps, arm_of, surf, K, R, C, per, rng)
            hpicks = trajectory_frames(args.raw, sorted(held), arm_of, surf, K, R, C, 2, rng)
        frames = load_frames(args.raw, picks, cam, args.stride)
        hframes = load_frames(args.raw, hpicks, cam, args.stride)
        trials = []
        for aspect in args.aspect:
            K0 = K.copy()
            K0[0, 0] = K[0, 0] / np.sqrt(aspect)    # giữ nguyên diện tích pixel lúc khởi tạo
            p = CamParams(K0, R, C, dev, aspect=aspect, fit_scale=args.fit_depth_scale,
                          fit_principal=args.fit_principal, fit_aspect=args.fit_aspect)
            before = residuals(p, hframes, surf, dev)
            icp(p, frames, surf, dev)
            after = residuals(p, hframes, surf, dev)
            print(f"[{cam}] aspect={aspect}: held-out point-to-surface: init {before} → ICP {after}")
            trials.append((after["median_mm"], aspect, p, before, after))
        _, aspect, p, before, after = min(trials, key=lambda x: x[0])
        out = p.export()
        out["aspect_trials"] = {str(a): t[4] for t in trials for a in [t[1]]}
        out.update(init={"K": K.tolist(), "R": R.tolist(), "C": C.tolist(), "inliers": int(inl.sum()),
                         "tin_tcp_median_mm": float(np.median(e[inl]) * 1000)},
                   heldout_residual_init=before, heldout_residual=after, icp_frames=len(frames))
        result["cameras"][cam] = out
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
    print("ghi", args.out)


if __name__ == "__main__":
    main()
