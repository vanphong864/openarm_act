#!/usr/bin/env python3
"""Sinh scenes/openarm_chest_cam.xml từ config/scene.yaml."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from openarm_sim.scene import load_config, write_scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(DEFAULT_SCENE))
    args = ap.parse_args()

    cfg = load_config(args.config)
    der = write_scene(cfg, args.out)
    mujoco.MjModel.from_xml_path(args.out)  # đảm bảo file sinh ra load được độc lập

    z0, z1 = cfg["robot"]["column_probe_z"]
    print(f"Mặt trước cột (mesh chân đế, z {z0:.2f}–{z1:.2f}): x = {der.column_front_x:.4f} m")
    print(f"Camera (quang tâm, tâm vỏ):  {der.camera_pos.round(4).tolist()} m")
    b = der.box_center
    print(f"Hộp: tâm {b.round(4).tolist()}, mép gần x = {b[0] - der.box_half[0]:.4f}, "
          f"mặt trên z = {2 * der.box_half[2]:.4f} m")
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
