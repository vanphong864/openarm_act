#!/usr/bin/env python3
"""Hiệu chuẩn CHUNG hai camera: robot (ICP) + hộp kẹo nằm yên ↔ FK lúc kẹp/nhả + nhất quán giữa hai camera.

Chỉ khớp với robot (calibrate_cameras.py) thì vùng mặt bàn — nơi cần chính xác nhất — là ngoại suy: đo được hộp kẹo
lệch ~4 cm so với TCP lúc kẹp và mặt hộp nghiêng ~6° trong khi FK cho thấy mặt hộp phẳng. Ở đây thêm:
  - Hộp kẹo đứng yên từ khung đầu tới lúc kẹp, và từ lúc nhả tới khung cuối. Tâm phần mặt trên nhìn thấy (depth,
    chiếu ngược bằng tham số camera đang tìm) phải trùng TCP (FK) lúc kẹp / nhả, cộng độ lệch z chung chưa biết
    (TCP so với mặt trên hộp kẹo). Lệch xy do kẹp không chính tâm coi là nhiễu (mất mát Huber, σ = 1.5 cm).
  - Cùng một hộp kẹo nhìn từ hai camera phải ra cùng điểm 3D (σ = 5 mm).
  - ICP điểm-tới-mặt với robot như calibrate_cameras.py (σ = 3 mm).
Ẩn mỗi camera: vị trí, hướng, fx, fy/fx (tâm quang học cố định ở giữa); thêm tâm hộp kẹo trong hệ bàn tay lúc
kẹp (3 mỗi tay) và độ cao mặt trên hộp kẹo h. Tâm hộp kẹo đo bằng tâm vùng xanh RGB giao với mặt z = h (vỏ kim
loại bóng làm depth hỏng ở camera ngực). Khung cuối (vật bị thả rơi, không gắn với TCP) chỉ ràng buộc hai camera.
Giải LM–IRLS.
Báo cáo trên episode giữ lại: tin↔TCP, tin camera↔camera, độ nghiêng mặt hộp so với FK.

Ví dụ:
  calibrate_joint.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --init outputs/sim_20261013/cameras_traj_free.json --blobs outputs/sim_20261013/tcp_blobs_depth.json \
      --out simulation/realdata/cameras_20261013.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_cameras import ROBOT, load_frames, nearest, rodrigues, trajectory_frames  # noqa: E402
from openarm_sim.robot_surface import RobotSurface  # noqa: E402

W, H = 640, 480
CAMS = ("chest", "front")
NP = 8          # tham số mỗi camera: w(3), C(3), log fx, log(fy/fx)


def green_hull(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 50) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    b = 1 + int(np.argmax(st[1:, 4]))
    if st[b, 4] < 150:
        return None
    keep = [i for i in range(1, n) if st[i, 4] >= 15 and np.linalg.norm(cen[i] - cen[b]) < 70]
    pts = np.column_stack(np.nonzero(np.isin(lab, keep)))[:, ::-1].astype(np.int32)
    out = np.zeros_like(m)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return cv2.erode(out, np.ones((5, 5), np.uint8)) > 0


class Model:
    def __init__(self, init: dict, dev, init_h=0.185):
        self.dev = dev
        f64 = dict(dtype=torch.float64, device=dev)
        self.R0 = {c: torch.tensor(init[c]["R_cam_from_world"], **f64) for c in CAMS}
        self.cxy = {c: (init[c]["cx"], init[c]["cy"]) for c in CAMS}
        th = []
        for c in CAMS:
            th += [0, 0, 0, *init[c]["C_world"], np.log(init[c]["fx"]), np.log(init[c]["fy"] / init[c]["fx"])]
        # δ trái (3), δ phải (3): tâm hộp kẹo = TCP + R_tcp δ lúc kẹp (hệ bàn tay); h: độ cao mặt trên hộp kẹo
        th += [0.0] * 6 + [init_h, init_h - 0.01]     # + h hộp kẹo, h mặt hộp kê
        self.theta = torch.tensor(th, **f64)

    def ray_plane(self, c, u, v, th=None, dz=0.0):
        """Giao tia qua pixel (u, v) với mặt phẳng ngang z = h_bàn + dz."""
        th = self.theta if th is None else th
        p = th[self.sl(c)]
        fx = p[6].exp()
        fy = fx * p[7].exp()
        cx, cy = self.cxy[c]
        d = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1) @ (rodrigues(p[:3]) @ self.R0[c])
        t = (th[2 * NP + 7] + dz - p[5]) / d[..., 2]
        return p[3:6] + t[..., None] * d

    def sl(self, c):
        i = CAMS.index(c) * NP
        return slice(i, i + NP)

    def backproject(self, c, u, v, z, th=None):
        th = self.theta if th is None else th
        p = th[self.sl(c)]
        fx = p[6].exp()
        fy = fx * p[7].exp()
        cx, cy = self.cxy[c]
        pc = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
        return pc @ (rodrigues(p[:3]) @ self.R0[c]) + p[3:6]

    def export(self, c):
        p = self.theta[self.sl(c)].cpu().numpy()
        R = (rodrigues(self.theta[self.sl(c)][:3]) @ self.R0[c]).cpu().numpy()
        fx = float(np.exp(p[6]))
        return {"fx": fx, "fy": fx * float(np.exp(p[7])), "cx": self.cxy[c][0], "cy": self.cxy[c][1],
                "aspect_fy_over_fx": float(np.exp(p[7])), "depth_scale_factor": 1.0,
                "R_cam_from_world": R.tolist(), "C_world": p[3:6].tolist()}


def pad_corners(rgb):
    """4 góc vùng trắng lớn nhất (ô giấy) trong ảnh, hoặc None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] < 50) & (hsv[..., 2] > 170)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    b = 1 + int(np.argmax(st[1:, 4]))
    if st[b, 4] < 1500:
        return None
    ys, xs = np.nonzero(lab == b)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    ap = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    return ap.astype(float) if len(ap) == 4 else None


def hold_events(raw, ep, arm):
    """(t_kẹp, t_nhả) và qpos tại đó: pha giữ đầu tiên/cuối cùng (kẹp đóng trên vật, lực > 0.8)."""
    g = 7 if arm == "left" else 15
    with h5py.File(Path(raw) / f"{ep}.hdf5", "r") as f:
        q = f["observations/qpos"][:]
        c = f["joint_commands"][:, g]
        e = f["observations/effort"][:, g]
    hold = (q[:, g] > 0.018) & (q[:, g] < 0.032) & (e > 0.8) & (c < 0.01)
    idx = np.flatnonzero(hold)
    if len(idx) < 10:
        return None
    return q[idx[0]], q[idx[-1]], len(q)


def tin_centroid(raw, ep, t, cam):
    """Tâm (u, v) bao lồi vùng xanh của hộp kẹo trong ảnh RGB (không dùng depth: vỏ kim loại bóng làm depth hỏng)."""
    with h5py.File(Path(raw) / f"{ep}.hdf5", "r") as f:
        rgb = f[f"observations/images/{cam}_rgb"][t]
    m = green_hull(rgb)
    if m is None or m.sum() < 150:
        return None
    ys, xs = np.nonzero(m)
    if xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3:
        return None          # bị cắt ở mép ảnh
    return float(xs.mean()), float(ys.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--blobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-per-camera", type=int, default=100)
    ap.add_argument("--sigma-icp", type=float, default=0.003)
    ap.add_argument("--sigma-tin", type=float, default=0.015)
    ap.add_argument("--sigma-cross", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fix-aspect", action="store_true", help="giữ fy/fx như khởi tạo")
    ap.add_argument("--no-grasp", action="store_true", help="bỏ ràng buộc hộp kẹo ↔ TCP lúc kẹp")
    ap.add_argument("--delta-axial-only", action="store_true",
                    help="tâm hộp kẹo nằm GIỮA hai ngón: chỉ cho lệch dọc trục bàn tay (z_tcp); ép camera khớp FK trên bàn")
    ap.add_argument("--fy-prior", nargs="*", default=[], metavar="CAM=FY",
                    help="prior mạnh cho fy (σ 3 px) đo độc lập; fx/fy để dữ liệu quyết định")
    ap.add_argument("--fix-camera", nargs="*", default=[], choices=CAMS,
                    help="giữ nguyên toàn bộ tham số camera này (vd front đã tin cậy) và hiệu chuẩn camera còn lại theo nó")
    ap.add_argument("--intrinsics", nargs="*", default=[], metavar="CAM=FX,FY",
                    help="cố định tiêu cự đo độc lập (vd front=453,604): ICP với robot làm phồng f một cách hệ thống")
    ap.add_argument("--sigma-plane", type=float, default=0.003)
    ap.add_argument("--rounds", type=int, nargs=3, default=[2, 3, 4], metavar=("R30", "R15", "R8"),
                    help="số vòng tương ứng ở ngưỡng ICP 30/15/8 mm")
    ap.add_argument("--sigma-pad", type=float, default=0.003)
    ap.add_argument("--tin-cross", action="store_true", help="thêm ràng buộc chéo tâm hộp kẹo (đặc trưng lệch theo "
                    "hướng nhìn, mặc định tắt)")
    ap.add_argument("--tin-thickness", type=float, default=0.0096,
                    help="độ dày hộp kẹo (m), đo tương đối bằng depth trong một camera; tia tâm vùng xanh giao mặt "
                         "z = h_bàn + độ dày/2")
    args = ap.parse_args()
    args.schedule = list(zip((0.03, 0.015, 0.008), args.rounds))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    init = json.loads(args.init.read_text())
    held = set(init["heldout_episodes"])
    blobs = json.loads(args.blobs.read_text())
    arm_of = {b["episode"]: b["arm"] for b in blobs}
    episodes = sorted(arm_of)
    rng = np.random.default_rng(args.seed)
    static = ("openarm_body_link0", "openarm_left_link0", "openarm_right_link0")
    surf = RobotSurface(str(ROBOT), spacing_m=0.005, exclude_bodies=static)
    fixed_f = {}
    for spec_ in args.intrinsics:
        cam_, vals = spec_.split("=")
        fx_, fy_ = (float(x) for x in vals.split(","))
        init["cameras"][cam_] = {**init["cameras"][cam_], "fx": fx_, "fy": fy_}
        fixed_f[cam_] = (fx_, fy_)
    model = Model(init["cameras"], dev, init_h=init.get("box_top_height_m", 0.175) + 0.01)

    # ---- dữ liệu ICP robot
    icp_data = {}
    for c in CAMS:
        ic = init["cameras"][c]
        K = np.array([[ic["fx"], 0, ic["cx"]], [0, ic["fy"], ic["cy"]], [0, 0, 1]])
        R, C = np.array(ic["R_cam_from_world"]), np.array(ic["C_world"])
        fit_eps = [e for e in episodes if e not in held]
        per = max(1, int(np.ceil(args.frames_per_camera / len(fit_eps))))
        frames = load_frames(args.raw, trajectory_frames(args.raw, fit_eps, arm_of, surf, K, R, C, per, rng), c, 3)
        robot = [tuple(torch.tensor(x, device=dev, dtype=torch.float32) for x in surf.world(fr["q"])) for fr in frames]
        data = [tuple(torch.tensor(fr[k], device=dev, dtype=torch.float64) for k in ("u", "v", "z")) for fr in frames]
        icp_data[c] = (data, robot)
        print(f"[{c}] {len(frames)} khung ICP", flush=True)

    # ---- dữ liệu hộp kẹo nằm yên: khung đầu (↔ TCP lúc kẹp) và khung cuối (chỉ ràng buộc hai camera)
    surf_tcp = RobotSurface(str(ROBOT), spacing_m=0.05)
    tins = []      # (episode, arm, phase, tcp[3], R_tcp[3,3] | None, {cam: (u, v)})
    for ep in episodes:
        ev = hold_events(args.raw, ep, arm_of[ep])
        if ev is None:
            continue
        q_g, _, T = ev
        surf_tcp.set_qpos(q_g)
        tid = surf_tcp.tcp[arm_of[ep]]
        tcp, Rt = surf_tcp.d.xpos[tid].copy(), surf_tcp.d.xmat[tid].reshape(3, 3).copy()
        for phase, t in (("start", 0), ("end", T - 1)):
            obs = {c: tin_centroid(args.raw, ep, t, c) for c in CAMS}
            obs = {c: o for c, o in obs.items() if o is not None}
            if obs:
                tins.append((ep, arm_of[ep], phase, tcp, Rt, obs))
    print(f"{len(tins)} quan sát hộp kẹo nằm yên ({sum(len(t[5]) for t in tins)} camera-quan sát)", flush=True)
    f64 = dict(device=dev, dtype=torch.float64)
    tin_t = [(ep, arm, ph, torch.tensor(tcp, **f64), torch.tensor(Rt, **f64),
              {c: (torch.tensor(o[0], **f64), torch.tensor(o[1], **f64)) for c, o in obs.items()})
             for ep, arm, ph, tcp, Rt, obs in tins]
    fit_tins = [t for t in tin_t if t[0] not in held]
    ho_tins = [t for t in tin_t if t[0] in held]
    D0 = 2 * NP
    HT, HB = D0 + 6, D0 + 7

    # ---- góc ô giấy ở khung đầu của các episode fit (ghép cặp theo vị trí chiếu lên mặt hộp với θ ban đầu)
    pad_obs = []
    for ep in [e for e in episodes if e not in held][::3]:
        with h5py.File(Path(args.raw) / f"{ep}.hdf5", "r") as f:
            cc = {c: pad_corners(f[f"observations/images/{c}_rgb"][0]) for c in CAMS}
        if any(v is None for v in cc.values()):
            continue
        # Ghép góc theo hướng nhìn đã biết (không phụ thuộc hiệu chuẩn ban đầu): camera ngực nhìn +x (lên ảnh = +x,
        # trái ảnh = +y), camera front nhìn −x (lên ảnh = −x, trái ảnh = −y). Sắp góc theo góc phương vị quanh tâm.
        def world_order(crn, cam):
            d = crn - crn.mean(0)
            X, Y = (-d[:, 1], -d[:, 0]) if cam == "chest" else (d[:, 1], d[:, 0])
            return np.argsort(np.arctan2(Y, X))
        oc, of = world_order(cc["chest"], "chest"), world_order(cc["front"], "front")
        pad_obs.append((torch.tensor(cc["chest"][oc], **f64), torch.tensor(cc["front"][of], **f64)))
    print(f"{len(pad_obs)} khung có đủ 4 góc ô giấy ở cả hai camera", flush=True)

    # ---- điểm mặt hộp kê (depth khung đầu, hai camera): phải nằm trên mặt ngang z = h_bàn
    table = {c: [] for c in CAMS}
    vv, uu = np.mgrid[0:H:4, 0:W:4]
    for ep in [e for e in episodes if e not in held][::6]:
        with h5py.File(Path(args.raw) / f"{ep}.hdf5", "r") as f:
            for c in CAMS:
                z = f[f"observations/images/{c}_depth"][0][::4, ::4].astype(np.float64) * 1e-3
                g = green_hull(f[f"observations/images/{c}_rgb"][0])
                g = np.zeros_like(z, bool) if g is None else (cv2.dilate(g.astype(np.uint8), np.ones((25, 25),
                                                                                                     np.uint8)) > 0)[::4, ::4]
                ok = (z > 0.2) & (z < 1.2) & ~g
                table[c].append(np.c_[uu[ok], vv[ok], z[ok]])
    table = {c: torch.tensor(np.concatenate(v), **f64) for c, v in table.items()}

    def table_points(th):
        """Điểm thuộc mặt hộp kê theo θ hiện tại (chọn trong vùng làm việc, gần mặt h_bàn)."""
        out = {}
        for c in CAMS:
            u, v, z = table[c].T
            P = model.backproject(c, u, v, z, th)
            with torch.no_grad():
                sel = ((P[:, 2] - th[HB]).abs() < 0.025) & (P[:, 0] > 0.08) & (P[:, 0] < 0.5) & (P[:, 1].abs() < 0.3)
            out[c] = sel
        return out

    def tin_terms(th, items):
        """(tin_camera − (TCP + R δ)) cho khung đầu; (chest − front) theo xy cho cả hai khung."""
        grasp, cross = [], []
        for ep, arm, ph, tcp, Rt, obs in items:
            X = {c: model.ray_plane(c, u, v, th, dz=args.tin_thickness / 2) for c, (u, v) in obs.items()}
            if ph == "start" and not args.no_grasp:
                dl = th[D0:D0 + 3] if arm == "left" else th[D0 + 3:D0 + 6]
                tgt = tcp + Rt @ dl
                for c, P in X.items():
                    grasp.append((c, P - tgt))
            if len(X) == 2:
                cross.append((X["chest"] - X["front"])[:2])
        for pc, pf in pad_obs:
            cross.append(None)                      # chỗ giữ: góc giấy tính riêng (pad_terms)
        return grasp, [x for x in cross if x is not None]

    def pad_terms(th):
        """Góc ô giấy (cùng điểm vật lý ở hai ảnh) chiếu lên mặt hộp kê phải trùng nhau theo xy."""
        out = []
        for pc, pf in pad_obs:
            Xc = model.ray_plane("chest", pc[:, 0], pc[:, 1], th)
            Xf = model.ray_plane("front", pf[:, 0], pf[:, 1], th)
            out.append((Xc - Xf)[:, :2].reshape(-1))
        return out

    def tin_residuals(th, items):
        grasp, cross = tin_terms(th, items)
        r = [g / args.sigma_tin for _, g in grasp]
        if args.tin_cross:
            r += [x / args.sigma_cross for x in cross]
        r += [x / args.sigma_pad for x in pad_terms(th)]
        for spec_ in args.fy_prior:
            cam_, val = spec_.split("=")
            p_ = th[CAMS.index(cam_) * NP:CAMS.index(cam_) * NP + NP]
            r.append(((p_[6].exp() * p_[7].exp() - float(val)) / 3.0).reshape(1))
        return torch.cat(r)

    def report(items, tag):
        with torch.no_grad():
            grasp, cross = tin_terms(model.theta, items)
        out = {}
        for c in CAMS:
            a = np.array([g.cpu().numpy() for cc, g in grasp if cc == c]) * 1000
            if len(a):
                out[f"{c}_tin_minus_grasp_mm"] = {"mean": a.mean(0).round(1).tolist(), "std": a.std(0).round(1).tolist(),
                                                  "rms_xy": float(np.sqrt((a[:, :2] ** 2).sum(1).mean()).round(1)),
                                                  "n": len(a)}
        with torch.no_grad():
            pads = np.concatenate([x.cpu().numpy().reshape(-1, 2) for x in pad_terms(model.theta)]) * 1000 \
                if pad_obs else np.zeros((0, 2))
        if len(pads):
            out["pad_corner_chest_minus_front_mm"] = {"rms": float(np.sqrt((pads ** 2).sum(1).mean()).round(1)),
                                                      "mean": pads.mean(0).round(1).tolist(), "n": len(pads)}
        a = np.array([x.cpu().numpy() for x in cross]) * 1000
        if len(a):
            out["tin_chest_minus_front_xy_mm"] = {"mean": a.mean(0).round(1).tolist(), "std": a.std(0).round(1).tolist(),
                                              "rms": float(np.sqrt((a ** 2).sum(1).mean()).round(1)), "n": len(a)}
        print(f"  [{tag}] {json.dumps(out)}", flush=True)
        return out

    rep0 = {"fit": report(fit_tins, "fit, trước"), "heldout": report(ho_tins, "giữ lại, trước")}

    lam = 1e-3
    for tau, rounds in args.schedule:
        for rnd in range(rounds):
            corr = {}
            with torch.no_grad():
                for c in CAMS:
                    U, V, Zs, S, N = [], [], [], [], []
                    Cw = model.theta[model.sl(c)][3:6].float()
                    for (u, v, z), (p, n) in zip(*icp_data[c]):
                        pw = model.backproject(c, u, v, z).float()
                        lo, hi = pw.min(0).values - tau, pw.max(0).values + tau
                        facing = (((Cw - p) * n).sum(1) > 0) & ((p > lo) & (p < hi)).all(1)
                        ps, ns = p[facing], n[facing]
                        i, dist = nearest(pw, ps)
                        k = dist < tau
                        U.append(u[k]), V.append(v[k]), Zs.append(z[k]), S.append(ps[i[k]].double())
                        N.append(ns[i[k]].double())
                    corr[c] = tuple(torch.cat(x) for x in (U, V, Zs, S, N))
            with torch.no_grad():
                tsel = table_points(model.theta)
            n_icp = sum(len(v[0]) for v in corr.values())
            n_tin = len(tin_residuals(model.theta, fit_tins))
            n_tab = sum(int(v.sum()) for v in tsel.values())
            w_tab = np.sqrt(max(n_tin, 1) / max(n_tab, 1))
            # cân hai nhóm: tổng trọng số ICP ≈ tổng trọng số hộp kẹo
            w_icp = np.sqrt(max(n_tin, 1) / max(n_icp, 1))

            def res(th):
                parts = []
                for c in CAMS:
                    U, V, Zs, S, N = corr[c]
                    parts.append(w_icp * ((model.backproject(c, U, V, Zs, th) - S) * N).sum(1) / args.sigma_icp)
                parts.append(tin_residuals(th, fit_tins))
                for c in CAMS:
                    u, v, z = table[c][tsel[c]].T
                    parts.append(w_tab * (model.backproject(c, u, v, z, th)[:, 2] - th[HB]) / args.sigma_plane)
                return torch.cat(parts)

            fixed_k = {CAMS.index(c) * NP + j for c in fixed_f for j in (6, 7)}
            fixed_k |= {CAMS.index(c) * NP + j for c in args.fix_camera for j in range(NP)}
            if args.delta_axial_only:
                fixed_k |= {D0, D0 + 1, D0 + 3, D0 + 4}
            free = [k for k in range(len(model.theta)) if not (args.fix_aspect and k < 2 * NP and k % NP == 7)
                    and k != HT and k not in fixed_k]
            for _ in range(5):
                x0 = model.theta.clone()
                e = res(x0)
                cols = []
                for k in free:
                    h = torch.zeros_like(x0)
                    h[k] = 1e-6
                    cols.append((res(x0 + h) - res(x0 - h)) / 2e-6)
                J = torch.stack(cols, 1)
                a = e.abs()
                w = torch.where(a <= 1.5, torch.ones_like(a), 1.5 / a)
                JtW = J.T * w
                Hm, g = JtW @ J, JtW @ e
                c0 = float((w * e * e).sum())
                while True:
                    step = -torch.linalg.solve(Hm + lam * torch.diag(torch.diag(Hm)) + 1e-9 * torch.eye(
                        len(free), dtype=Hm.dtype, device=Hm.device), g)
                    dx = torch.zeros_like(x0)
                    dx[free] = step
                    e1 = res(x0 + dx)
                    if float((w * e1 * e1).sum()) < c0 or lam > 1e6:
                        lam = max(lam / 3, 1e-7)
                        break
                    lam *= 4
                model.theta = x0 + dx
            msg = "; ".join(f"{c}: fx={model.export(c)['fx']:.1f} fy={model.export(c)['fy']:.1f} "
                            f"C={np.round(model.export(c)['C_world'], 4)}" for c in CAMS)
            print(f"  tau={tau * 1000:.0f}mm vòng {rnd}: ICP {n_icp} điểm, tin {n_tin} phần dư; {msg}; "
                  f"δL={model.theta[D0:D0 + 3].cpu().numpy().round(4)} δR={model.theta[D0 + 3:D0 + 6].cpu().numpy().round(4)} "
                  f"h_bàn={float(model.theta[HB]):.4f} ({n_tab} điểm bàn)", flush=True)

    rep1 = {"fit": report(fit_tins, "fit, sau"), "heldout": report(ho_tins, "giữ lại, sau")}
    with torch.no_grad():
        tsel = table_points(model.theta)
        tilt = {}
        for c in CAMS:
            u, v, z = table[c][tsel[c]].T
            P = model.backproject(c, u, v, z).cpu().numpy()
            A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
            k = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
            tilt[c] = {"dz_dx": float(k[0]), "dz_dy": float(k[1]), "z0": float(k[2]),
                       "resid_mm": float((P[:, 2] - A @ k).std() * 1000), "points": int(len(P))}
    print("  mặt hộp kê theo từng camera:", json.dumps(tilt), flush=True)
    rep1["table_plane"] = tilt
    out = {"method": __doc__.split("\n")[0], "init": str(args.init), "heldout_episodes": sorted(held),
           "fixed_intrinsics": {c: {"fx": v[0], "fy": v[1]} for c, v in fixed_f.items()},
           "image_size": [W, H], "cameras": {c: model.export(c) for c in CAMS},
           "tin_center_in_hand_frame_m": {"left": model.theta[D0:D0 + 3].cpu().numpy().tolist(),
                                          "right": model.theta[D0 + 3:D0 + 6].cpu().numpy().tolist()},
           "tin_top_height_m": float(model.theta[HB]) + args.tin_thickness, "box_top_height_m": float(model.theta[HB]),
           "tin_thickness_m": args.tin_thickness,
           "report_before": rep0, "report_after": rep1}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("ghi", args.out)


if __name__ == "__main__":
    main()
