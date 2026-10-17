#!/usr/bin/env python3
"""Kiểm chứng đầu-cuối scene đo đạc: phát lại joint_commands thật (mở vòng) với hộp kẹo đặt đúng tư thế đầu đo được.

Mỗi episode: robot đặt ở qpos thật khung đầu, hộp kẹo ở tư thế đầu (reconstruct_scene.py), lệnh thật đi qua bộ giới
hạn tốc độ + servo PD nhận dạng. So với thật:
  - sai số qpos tay đang thao tác (rad), độ mở kẹp lúc giữ vật (sim vs thật: kiểm tra kích thước vật + hình học ngón);
  - vật có được nhấc lên (> 3 cm) không; vị trí cuối so với vị trí cuối đo được; vật có nằm trong ô giấy;
  - (tuỳ chọn) ảnh camera ghép nền thật cạnh ảnh thật + sai số depth trên toàn ảnh.
Đây là phát lại mở vòng: sai lệch nhỏ của mô hình tích luỹ và không được người điều khiển sửa như ngoài thực tế.

Ví dụ:
  replay_real_20261013.py --config simulation/config/scene_20261013.yaml \
      --raw "/media/space/New Volume/dataset/2026-10-13" --out outputs/sim_20261013/replay --render
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
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openarm_sim.composite_camera import CompositeCamera  # noqa: E402
from openarm_sim.measured_scene import compile_measured  # noqa: E402
from openarm_sim.runner import Sim  # noqa: E402
from openarm_sim.scene import PAD_GEOM  # noqa: E402


def load_cfg(path: Path, raw: Path) -> dict:
    """Config đo đạc đã điền số đo (cùng hàm với build_measured_scene.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_measured_scene import resolve
    from openarm_sim.measured_scene import load_measured_config
    return resolve(load_measured_config(path), raw)


def set_object(m, d, name, xy, yaw_deg, z):
    j = m.body(name).jntadr[0]
    a = m.jnt_qposadr[j]
    d.qpos[a:a + 3] = [xy[0], xy[1], z]
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0, 0, np.radians(yaw_deg)]), "xyz")
    d.qpos[a + 3:a + 7] = q
    d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", nargs="*", help="mặc định: episode giữ lại khi hiệu chuẩn camera")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--tin-from", choices=["camera", "fk"], default="fk",
                    help="vị trí đầu hộp kẹo: camera (đo ở khung đầu) hoặc fk (dưới TCP lúc robot thật kẹp, trục dài ⟂ "
                         "phương khép ngón, yaw quanh trục đó)")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.raw)
    sc = cfg["_measured"]["scene"]
    eps = args.episodes or cfg["_measured"]["cameras"]["heldout_episodes"]
    start = {r["episode"]: r for r in sc["tin_poses"]["start"]}
    end = {r["episode"]: r for r in sc["tin_poses"]["end"]}
    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    obj = "mentos_tin"
    obj_body = m.body(obj).id
    half_h = cfg["objects"][obj]["size"][2] / 2
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    pad = m.geom(PAD_GEOM).id
    box_xy0 = m.body_pos[m.geom_bodyid[pad]][:2].copy()
    pad_xy0 = np.array(sc["pad"]["center"][:2])
    cams, bgdir = {}, None
    if args.render:
        bgdir = (args.config.parent / cfg["background_dir"]).resolve()  # nền tham chiếu
        cams = {c: CompositeCamera(m, c, bgdir) for c in cfg["_measured"]["cameras"]["cameras"]}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in eps:
        if ep not in start:
            print(f"{ep}: không có tư thế đầu của hộp kẹo, bỏ qua")
            continue
        with h5py.File(args.raw / f"{ep}.hdf5", "r") as f:
            q = f["observations/qpos"][:].astype(np.float64)
            cmd = f["joint_commands"][:].astype(np.float64)
            eff = f["observations/effort"][:]
            vis_t = np.linspace(0, len(q) - 1, 5).astype(int) if args.render else []
            real_imgs = {t: {c: (f[f"observations/images/{c}_rgb"][t], f[f"observations/images/{c}_depth"][t])
                             for c in cams} for t in vis_t}
        arm = "left" if q[:, 7].min() < 0.035 else "right"
        g = 7 if arm == "left" else 15
        cols = [*range(7)] if arm == "left" else [*range(8, 15)]
        pe = sc["pad"].get("per_episode", {}).get(ep)
        if pe:                                     # hộp kê/ô giấy bị xê dịch giữa các phiên: dời cả hộp theo ô
            bb = m.geom_bodyid[pad]
            m.body_pos[bb][:2] = box_xy0 + (np.array(pe["center"]) - pad_xy0)
            m.geom_pos[pad][:2] = np.array(pe["center"]) - m.body_pos[bb][:2]
            qq = np.zeros(4)
            mujoco.mju_euler2Quat(qq, np.array([0, 0, np.radians(pe["yaw_deg"])]), "xyz")
            m.geom_quat[pad] = qq
        xy0, yaw0 = start[ep]["xy"], start[ep]["yaw_deg"]
        if args.tin_from == "fk":
            g_ = 7 if arm == "left" else 15
            hold_ = np.flatnonzero((q[:, g_] > 0.018) & (q[:, g_] < 0.032) & (eff[:, g_] > 0.8) & (cmd[:, g_] < 0.01))
            if len(hold_):
                sim.reset(qpos16=q[hold_[0]])
                mujoco.mj_forward(m, sim.d)
                tb = m.body(f"openarm_{arm}_hand_tcp").id
                Rt = sim.d.xmat[tb].reshape(3, 3)
                xy0 = sim.d.xpos[tb][:2].tolist()
                yaw0 = float(np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0])))   # trục x_tcp ⟂ phương khép ngón (y_tcp)
        for c, cam in cams.items():
            per = bgdir / "bg_episodes" / f"background_{c}_{ep}_rgb.png"
            cam.set_background(f"_{ep}", per.parent) if per.exists() else cam.set_background("")
        sim.reset(qpos16=q[0])
        set_object(m, sim.d, obj, xy0, yaw0, box_top + half_h + 1e-4)
        mujoco.mj_forward(m, sim.d)
        sim.settle(0.2)
        sq, grip_sim, zmax = [], [], 0.0
        panels, depth_err = [], []
        for t in range(len(q) - 1):
            sim.step(cmd[t])
            qs = sim.qpos16()
            sq.append(qs - q[t + 1])
            zmax = max(zmax, float(sim.d.xpos[obj_body][2] - (box_top + half_h)))
            hold = (q[t + 1, g] > 0.018) & (q[t + 1, g] < 0.032) & (eff[t + 1, g] > 0.8)
            if hold:
                grip_sim.append((qs[g], q[t + 1, g]))
            if t + 1 in real_imgs:
                row = []
                for c, cam in cams.items():
                    rgb, dep = cam.observe(sim.d)
                    rr, rd = real_imgs[t + 1][c]
                    both = (dep > 0) & (rd > 0)
                    depth_err.append(np.median(np.abs(dep[both].astype(float) - rd[both])))
                    dv = lambda x: cv2.applyColorMap(np.clip((x.astype(np.float32) - 200) / 1000 * 255, 0,  # noqa: E731
                                                             255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
                    row += [rr, rgb, dv(rd), dv(dep)]
                panels.append(np.concatenate(row, 1))
        sq = np.array(sq)
        p_end = sim.d.xpos[obj_body].copy()
        pad_c, pad_h = sim.d.geom_xpos[pad], m.geom_size[pad][:2]
        R = sim.d.geom_xmat[pad].reshape(3, 3)
        loc = (p_end - pad_c) @ R
        in_pad = bool(abs(loc[0]) <= pad_h[0] and abs(loc[1]) <= pad_h[1] and p_end[2] < box_top + 0.05)
        gs = np.array(grip_sim) if grip_sim else np.zeros((0, 2))
        r = {"episode": ep, "arm": arm, "qpos_rms_rad": float(np.sqrt((sq[:, cols] ** 2).mean())),
             "qpos_rms_per_joint": np.sqrt((sq[:, cols] ** 2).mean(0)).round(4).tolist(),
             "grip_hold_sim_mm": float(np.median(gs[:, 0]) * 1000) if len(gs) else None,
             "grip_hold_real_mm": float(np.median(gs[:, 1]) * 1000) if len(gs) else None,
             "lifted": zmax > 0.03, "max_lift_m": zmax, "tin_end_sim_xy": p_end[:2].round(4).tolist(),
             "tin_end_real_xy": end[ep]["xy"] if ep in end else None,
             "end_error_mm": float(np.linalg.norm(p_end[:2] - end[ep]["xy"]) * 1000) if ep in end else None,
             "in_pad_sim": in_pad, "depth_median_abs_err_mm": float(np.median(depth_err)) if depth_err else None}
        rows.append(r)
        print(json.dumps(r), flush=True)
        if panels:
            img = np.concatenate(panels, 0)
            cv2.imwrite(str(args.out / f"{ep}.jpg"), cv2.resize(img, None, fx=0.35, fy=0.35)[..., ::-1])
    summary = {"episodes": len(rows),
               "qpos_rms_rad_mean": float(np.mean([r["qpos_rms_rad"] for r in rows])),
               "lift_rate": float(np.mean([r["lifted"] for r in rows])),
               "in_pad_rate": float(np.mean([r["in_pad_sim"] for r in rows])),
               "end_error_mm_median": float(np.median([r["end_error_mm"] for r in rows if r["end_error_mm"] is not None])),
               "grip_hold_mm_sim_vs_real": [float(np.median([r["grip_hold_sim_mm"] for r in rows if r["grip_hold_sim_mm"]])),
                                            float(np.median([r["grip_hold_real_mm"] for r in rows if r["grip_hold_real_mm"]]))]}
    if any(r["depth_median_abs_err_mm"] is not None for r in rows):
        summary["depth_median_abs_err_mm"] = float(np.median([r["depth_median_abs_err_mm"] for r in rows
                                                              if r["depth_median_abs_err_mm"] is not None]))
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
