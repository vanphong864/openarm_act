#!/usr/bin/env python3
"""Kiểm chứng hiệu chuẩn camera: render robot (qpos thật) bằng camera đã hiệu chuẩn, so với ảnh thật.

- Depth: trên pixel sim thấy robot và depth thật hợp lệ, sai số |z_thật − z_sim| (mm). Pixel bị vật khác che
  (người, hộp kẹo) cho sai số lớn; báo trung vị và tỉ lệ pixel lệch < 1 cm / 2 cm.
- RGB: ảnh chồng viền robot sim (tím) lên ảnh thật; viền khớp ở cả RGB và depth nghĩa là depth đã căn theo RGB.

Ví dụ:
  validate_cameras.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --cameras simulation/realdata/cameras_20261013.json --out outputs/sim_20261013/camera_check
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
from openarm_sim.calibrated_camera import add_camera  # noqa: E402
from openarm_sim.robot_surface import ARM16  # noqa: E402

ROBOT = Path(__file__).resolve().parent.parent / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"


def build(calib, hide=()):
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    for g in spec.geoms:
        if g.parent.name in hide:
            g.group = 5                       # không render (nhóm ngoài mặc định)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    for name, c in calib.items():
        add_camera(spec.worldbody, name, c)
    return spec.compile()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", nargs="*", help="mặc định: các episode giữ lại khi hiệu chuẩn")
    ap.add_argument("--frames", type=int, default=6, help="số khung mỗi episode")
    ap.add_argument("--hide-bodies", nargs="*", default=[], help="body không render (vd thân robot che camera ngực)")
    args = ap.parse_args()
    info = json.loads(args.cameras.read_text())
    calib = info["cameras"]
    eps = args.episodes or info["heldout_episodes"]
    m = build(calib, tuple(args.hide_bodies))
    d = mujoco.MjData(m)
    qadr = [m.joint(f"openarm_{s}_joint{i}").qposadr[0] for s in ("left", "right") for i in range(1, 8)]
    fadr = {s: [m.joint(f"openarm_{s}_finger_joint{i}").qposadr[0] for i in (1, 2)] for s in ("left", "right")}
    r = mujoco.Renderer(m, 480, 640)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {c: [] for c in calib}
    for ep in eps:
        with h5py.File(args.raw / f"{ep}.hdf5", "r") as f:
            q = f["observations/qpos"][:]
            ts = np.linspace(0, len(q) - 1, args.frames).astype(int)
            panels = []
            for t in ts:
                d.qpos[:] = 0
                d.qpos[qadr] = q[t, ARM16]
                for s, g in (("left", 7), ("right", 15)):
                    d.qpos[fadr[s]] = q[t, g]
                mujoco.mj_forward(m, d)
                row = []
                for cam in calib:
                    rgb = f[f"observations/images/{cam}_rgb"][t].copy()
                    zr = f[f"observations/images/{cam}_depth"][t].astype(np.float64) * 1e-3
                    r.enable_depth_rendering()
                    r.update_scene(d, cam)
                    zs = r.render().astype(np.float64)
                    r.disable_depth_rendering()
                    r.enable_segmentation_rendering()
                    r.update_scene(d, cam)
                    seg = r.render()[..., 0]
                    r.disable_segmentation_rendering()
                    robot = seg >= 0
                    ok = robot & (zr > 0.2) & (zr < 1.2)
                    # Bỏ pixel robot bị vật khác che (depth thật GẦN hơn sim > 2 cm: hộp kê che cột thân, hộp kẹo,
                    # người). Còn lại: robot nhìn thấy được, hoặc chỗ sim đặt robot mà ảnh thật thấy xuyên qua.
                    vis_ = ok & (zr > zs - 0.02)
                    e = (zr[vis_] - zs[vis_]) * 1000
                    if len(e) > 200:
                        stats[cam].append({"episode": ep, "t": int(t), "pixels": int(len(e)),
                                           "median_abs_mm": float(np.median(np.abs(e))),
                                           "frac_lt_10mm": float((np.abs(e) < 10).mean()),
                                           "frac_lt_20mm": float((np.abs(e) < 20).mean()),
                                           "frac_see_through": float((e > 20).mean())})
                    cnt, _ = cv2.findContours(robot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    vis = rgb.copy()
                    cv2.drawContours(vis, cnt, -1, (255, 0, 255), 1)
                    dz = np.clip((zr - zs) * 1000, -30, 30)
                    dvis = cv2.applyColorMap(((dz + 30) / 60 * 255).astype(np.uint8), cv2.COLORMAP_COOLWARM
                                             if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET)[..., ::-1].copy()
                    dvis[~ok] = (vis[~ok] * 0.35).astype(np.uint8)
                    cv2.drawContours(dvis, cnt, -1, (255, 0, 255), 1)
                    row += [vis, dvis]
                panels.append(np.concatenate(row, 1))
            img = np.concatenate(panels, 0)
            cv2.imwrite(str(args.out / f"{ep}.jpg"), cv2.resize(img, None, fx=0.5, fy=0.5)[..., ::-1])
    summary = {}
    for cam, rows in stats.items():
        if rows:
            summary[cam] = {k: float(np.median([x[k] for x in rows])) for k in ("median_abs_mm", "frac_lt_10mm",
                                                                                 "frac_lt_20mm", "frac_see_through")}
            summary[cam]["frames"] = len(rows)
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "frames": stats}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
