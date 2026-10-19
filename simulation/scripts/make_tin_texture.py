#!/usr/bin/env python3
"""Texture mặt trên hộp kẹo từ ảnh thật: chiếu 4 góc mặt trên (tư thế đo ở khung đầu + camera đã hiệu chuẩn)
vào ảnh RGB thật rồi nắn phẳng. Dùng một episode (ảnh trung vị nhiều episode bị mờ vì tư thế đo lệch vài mm).

Ví dụ:
  make_tin_texture.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --config simulation/config/scene_20261013.yaml --out simulation/realdata/tin_top.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openarm_sim.calibrated_camera import project  # noqa: E402
from replay_real_20261013 import load_cfg  # noqa: E402

PX_PER_M = 4000  # 4 px/mm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default="chest")
    ap.add_argument("--episode", default="episode_20261013_110137", help="episode có nhãn rõ, nằm gọn trong khung")
    ap.add_argument("--gain", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("R", "G", "B"),
                    help="hệ số màu (thật/sim, đo bằng scripts/sim_realism.py): ảnh nguồn ngả màu + đèn sim")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.raw)
    calib = cfg["_measured"]["cameras"]["cameras"][args.camera]
    sx, sy, sz = cfg["objects"]["mentos_tin"]["size"]
    z = cfg["_measured"]["cameras"]["box_top_height_m"] + sz
    W, H = round(sx * PX_PER_M), round(sy * PX_PER_M)
    # góc mặt trên trong hệ hộp kẹo theo thứ tự ảnh texture: (u trái→phải = +x, v trên→dưới = +y)
    local = np.array([[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]])
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    r = {p["episode"]: p for p in cfg["_measured"]["scene"]["tin_poses"]["start"]}[args.episode]
    a = np.radians(r["yaw_deg"])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    uv, depth = project(calib, np.c_[local @ R.T + np.array(r["xy"]), np.full(4, z)])
    if (depth <= 0).any():
        raise SystemExit("Mặt trên hộp kẹo nằm sau camera")
    with h5py.File(args.raw / f"{args.episode}.hdf5", "r") as f:
        rgb = f[f"observations/images/{args.camera}_rgb"][0]
    texture = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(np.float32(uv), dst), (W, H))
    texture = np.clip(texture * np.array(args.gain), 0, 255).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Ảnh nắn ở trên nhìn từ trên xuống với v = +y là ảnh gương; MuJoCo đặt hàng ảnh đầu ở v lớn: lật dọc.
    cv2.imwrite(str(args.out), texture[::-1, :, ::-1])
    print(f"{args.episode} {args.camera}: {W}x{H} px -> {args.out}")


if __name__ == "__main__":
    main()
