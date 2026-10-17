#!/usr/bin/env python3
"""Tinh chỉnh camera bằng render-and-compare hai chiều với robot (qpos thật) — bước sau calibrate_cameras.py.

ICP một chiều (pixel depth → bề mặt robot) không phạt được trường hợp mô hình đặt robot ở chỗ ảnh thật thấy xuyên
qua (vd camera sim lùi quá nên thấy vai). Ở đây, mỗi khung render depth + phân đoạn robot rồi so với depth thật:
  - pixel sim thấy robot: |Δz| < 2 cm → khớp (sai số bậc hai, cắt); depth thật XA hơn sim > 2 cm → vi phạm không gian
    trống (phạt); depth thật GẦN hơn → bị vật khác che (hộp kê, người, hộp kẹo) → bỏ qua;
  - pixel thật là tiền cảnh (gần hơn nền tĩnh > 3 cm, nền = phân vị 90% depth theo thời gian của episode) mà sim
    không có robot trong bán kính vài px → robot thật ở chỗ mô hình không có (phạt).
Tham số: vị trí (3), quay nhỏ quanh tư thế ban đầu (3), tiêu cự fx, tỉ lệ fy/fx (tuỳ chọn); tâm quang học
ở giữa ảnh.
Tối ưu Nelder–Mead (không cần đạo hàm), render ở nửa độ phân giải. Báo cáo trên episode giữ lại.

Ví dụ:
  refine_cameras_render.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --cameras outputs/sim_20261013/cameras_traj.json --out simulation/realdata/cameras_20261013.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openarm_sim.calibrated_camera import cv_to_mujoco_quat  # noqa: E402
from openarm_sim.robot_surface import ARM16  # noqa: E402

ROBOT = Path(__file__).resolve().parent.parent / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"
W, H, DS = 640, 480, 2          # ảnh gốc và hệ số thu nhỏ khi render


def rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Renderer:
    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(ROBOT))
        w, h = W // DS, H // DS
        spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
        cam = spec.worldbody.add_camera(name="probe")
        cam.resolution = [w, h]
        cam.sensor_size = [float(w), float(h)]
        cam.focal_pixel = [300.0, 300.0]
        cam.principal_pixel = [0.0, 0.0]
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.cid = self.m.camera("probe").id
        m = self.m
        self.qadr = [m.joint(f"openarm_{s}_joint{i}").qposadr[0] for s in ("left", "right") for i in range(1, 8)]
        self.fadr = {s: [m.joint(f"openarm_{s}_finger_joint{i}").qposadr[0] for i in (1, 2)]
                     for s in ("left", "right")}
        self.r = mujoco.Renderer(m, h, w)

    def set_camera(self, cam: dict):
        m, w, h = self.m, W // DS, H // DS
        m.cam_pos[self.cid] = cam["C_world"]
        m.cam_quat[self.cid] = cv_to_mujoco_quat(np.array(cam["R_cam_from_world"]))
        m.cam_intrinsic[self.cid] = [cam["fx"] / DS, cam["fy"] / DS,
                                     w / 2 - ((cam["cx"] + 0.5) / DS), h / 2 - ((cam["cy"] + 0.5) / DS)]
        m.cam_sensorsize[self.cid] = [w, h]

    def render(self, q16):
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.qadr] = q16[ARM16]
        for s, g in (("left", 7), ("right", 15)):
            d.qpos[self.fadr[s]] = q16[g]
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        self.r.enable_depth_rendering()
        self.r.update_scene(d, "probe")
        z = self.r.render().astype(np.float64)
        self.r.disable_depth_rendering()
        self.r.enable_segmentation_rendering()
        self.r.update_scene(d, "probe")
        seg = self.r.render()[..., 0]
        self.r.disable_segmentation_rendering()
        return z, seg >= 0


def load_episode_frames(raw, ep, cam, n_frames, rng):
    with h5py.File(Path(raw) / f"{ep}.hdf5", "r") as f:
        q = f["observations/qpos"][:]
        T = len(q)
        dset = f[f"observations/images/{cam}_depth"]
        stack = np.stack([dset[t][::DS, ::DS] for t in range(0, T, 15)]).astype(np.float64) * 1e-3
        stack[(stack < 0.2) | (stack > 1.2)] = np.nan
        bg = np.nanpercentile(stack, 90, axis=0)
        ts = np.sort(rng.choice(np.arange(T), n_frames, replace=False))
        frames = [(q[t], dset[t][::DS, ::DS].astype(np.float64) * 1e-3) for t in ts]
    return bg, frames


def cost_frame(z_sim, robot, z_real, bg, near=0.02, fg=0.03):
    valid = (z_real > 0.2) & (z_real < 1.2)
    dz = z_real - z_sim
    S = robot & valid
    good = S & (np.abs(dz) < near)
    viol = S & (dz > near)
    foreground = valid & np.isfinite(bg) & (z_real < bg - fg)
    near_robot = cv2.dilate(robot.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    miss = foreground & ~near_robot
    n = S.sum() + foreground.sum()
    err = np.minimum(np.abs(dz[good]) / near, 1.0) ** 2
    return viol.sum() + miss.sum(), err.sum(), good.sum(), n


def evaluate(rend, cam, data):
    rend.set_camera(cam)
    bad = sq = good = n = 0
    for bg, frames in data:
        for q, z_real in frames:
            z, robot = rend.render(q)
            b, s, g, k = cost_frame(z, robot, z_real, bg)
            bad, sq, good, n = bad + b, sq + s, good + g, n + k
    n = max(n, 1)
    return {"bad_frac": bad / n, "good_frac": good / n, "sq_good": sq / max(good, 1), "cost": (bad + sq) / n}


def params_to_cam(x, base):
    R0, C0 = np.array(base["R_cam_from_world"]), np.array(base["C_world"])
    R = rotvec(x[3:6] * 0.01) @ R0          # đơn vị: 0.01 rad
    f = base["fx"] * np.exp(x[6] * 0.01)    # đơn vị: 1%
    a = base["fy"] / base["fx"] * (np.exp(x[7] * 0.01) if len(x) > 7 else 1.0)
    return {**base, "C_world": (C0 + x[:3] * 0.01).tolist(), "R_cam_from_world": R.tolist(), "fx": f, "fy": f * a}


def nelder_mead(fun, x0, step=1.0, iters=400, tol=1e-5, log=print):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [fun(p) for p in pts]
    for it in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if it % 20 == 0:
            log(f"    NM {it}: cost {vals[0]:.5f} x={np.round(pts[0], 3)}")
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1])
        fr = fun(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [fun(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-episodes", type=int, default=14)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--fit-aspect", action="store_true")
    args = ap.parse_args()
    info = json.loads(args.cameras.read_text())
    rng = np.random.default_rng(args.seed)
    held = info["heldout_episodes"]
    eps = sorted({p.stem for p in Path(args.raw).glob("*.hdf5") if p.stat().st_size > 0} - set(held))
    good_eps = []
    for e in eps:
        try:
            h5py.File(Path(args.raw) / f"{e}.hdf5", "r").close()
            good_eps.append(e)
        except OSError:
            pass
    fit_eps = sorted(rng.choice(good_eps, min(args.fit_episodes, len(good_eps)), replace=False).tolist())
    rend = Renderer()
    out = dict(info)
    out["render_refinement"] = {"fit_episodes": fit_eps, "method": __doc__.split("\n")[0]}
    for name, base in info["cameras"].items():
        print(f"[{name}] nạp dữ liệu...", flush=True)
        data = [load_episode_frames(args.raw, e, name, args.frames, rng) for e in fit_eps]
        hdata = [load_episode_frames(args.raw, e, name, args.frames, rng) for e in held]
        before = evaluate(rend, base, hdata)
        print(f"[{name}] giữ lại trước: {before}", flush=True)
        fun = lambda x: evaluate(rend, params_to_cam(x, base), data)["cost"]  # noqa: E731
        nx = 8 if args.fit_aspect else 7
        best_x, best_v = np.zeros(nx), fun(np.zeros(nx))
        for r in range(args.restarts):
            x, v = nelder_mead(fun, best_x, step=2.0 / (r + 1), log=lambda s: print(s, flush=True))
            if v < best_v:
                best_x, best_v = x, v
            print(f"  restart {r}: cost {v:.5f}", flush=True)
        cam = params_to_cam(best_x, base)
        after = evaluate(rend, cam, hdata)
        print(f"[{name}] giữ lại sau: {after}; fx={cam['fx']:.1f} C={np.round(cam['C_world'], 4)}", flush=True)
        cam["render_heldout_before"], cam["render_heldout_after"] = before, after
        out["cameras"][name] = cam
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    print("ghi", args.out)


if __name__ == "__main__":
    main()
