#!/usr/bin/env python3
"""Sinh scene MuJoCo từ số đo trên dữ liệu (camera, bộ chấp hành, cảnh) + config mỏng.

Điền vào config những gì đo được: hộp kê (độ cao, vùng mặt nhìn thấy), ô giấy (tâm, kích thước, yaw), hộp kẹo
(dài × rộng × dày, tư thế đầu trung vị), phân bố vị trí sinh của hộp kẹo theo từng tay, tư thế home (trung vị khung
đầu của dữ liệu). Ghi XML đứng độc lập và config đã điền (*.resolved.yaml) để runner dùng lại.

Ví dụ:
  build_measured_scene.py --config simulation/config/scene_20261013.yaml \
      --raw "/media/space/New Volume/dataset/2026-10-13" --out simulation/scenes/openarm_20261013.xml
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openarm_sim.measured_scene import load_measured_config, write_measured_scene  # noqa: E402


def episode_arms(raw: Path):
    out = {}
    for p in sorted(raw.glob("*.hdf5")):
        try:
            with h5py.File(p, "r") as f:
                q = f["observations/qpos"][:]
        except OSError:
            continue
        arm = "left" if q[:, 7].min() < 0.035 else ("right" if q[:, 15].min() < 0.035 else None)
        out[p.stem] = (arm, q[0])
    return out


def resolve(cfg: dict, raw: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    sc = cfg["_measured"]["scene"]
    cal = cfg["_measured"]["cameras"]
    h_box = cal["box_top_height_m"]
    s = sc["surface"]
    # Mặt hộp: mặt phẳng ngang ở độ cao đo được; biên = vùng mặt nhìn thấy (gộp hai camera)
    (x0, x1), (y0, y1) = s["x_range_seen"], s["y_range_seen"]
    b = cfg["box"]
    b["center_xy"] = b.get("center_xy") or [(x0 + x1) / 2, (y0 + y1) / 2]
    b["size_x"] = b.get("size_x") or (x1 - x0)
    b["size_y"] = b.get("size_y") or (y1 - y0)
    s.update(normal=[0.0, 0.0, 1.0], offset=h_box, height_at_origin_m=h_box,
             center=[*b["center_xy"], h_box], size_long_m=b["size_x"], size_short_m=b["size_y"])
    s["support_plane"] = {"height_at_origin_m": 0.0,
                          "note": "robot và hộp kê cùng đặt trên bàn gỗ, mặt bàn = gốc chân đế robot (z = 0)"}
    p = sc["pad"]
    cfg["target_pad"].update(center=[*p["center"], h_box], size=p["size"], yaw_deg=p["yaw_deg"],
                             rgba=cfg["target_pad"].get("rgba", [0.93, 0.93, 0.92, 1]))
    tin = sc["tin"]
    o = cfg["objects"]["mentos_tin"]
    # Kích thước ghi trong config (đo bằng depth + độ mở kẹp, xem config) được ưu tiên hơn vết RGB (bị phồng)
    o["size"] = o.get("size") or [tin["length"]["median"], tin["width"]["median"], tin["thickness_m"]]
    arms = episode_arms(raw)
    starts = {"left": [], "right": []}
    for r in sc["tin_poses"]["start"]:
        arm = arms.get(r["episode"], (None,))[0]
        if arm:
            starts[arm].append([*r["xy"], r["yaw_deg"]])
    spawn = {}
    for arm, rows in starts.items():
        a = np.array(rows)
        spawn[arm] = {"x": np.percentile(a[:, 0], [5, 95]).round(4).tolist(),
                      "y": np.percentile(a[:, 1], [5, 95]).round(4).tolist(),
                      "yaw_deg": np.percentile(a[:, 2], [5, 95]).round(1).tolist(),
                      "median": np.median(a, 0).round(4).tolist(), "n": len(a)}
    o["pos_xy"] = spawn["left"]["median"][:2]
    o["yaw_deg"] = spawn["left"]["median"][2]
    home = np.median(np.array([q0 for _, q0 in arms.values()]), 0)
    cfg["task"].update(arm=cfg["task"].get("arm", "left"), spawn_by_arm=spawn,
                       home_qpos=home.round(4).tolist(),
                       place_by_arm={arm: np.median([[*r["xy"]] for r in sc["tin_poses"]["end"]
                                                     if arms.get(r["episode"], (None,))[0] == arm], 0).round(4).tolist()
                                     for arm in ("left", "right")})
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = resolve(load_measured_config(args.config), args.raw)
    out = write_measured_scene(cfg, args.out)
    m = mujoco.MjModel.from_xml_path(str(out))
    resolved = {k: v for k, v in cfg.items() if not k.startswith("_")}
    rpath = args.config.with_suffix(".resolved.yaml")
    rpath.write_text("# SINH TỰ ĐỘNG bởi build_measured_scene.py — đừng sửa tay\n" +
                     yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"xml": str(out), "nq": m.nq, "ncam": m.ncam, "box": cfg["box"], "pad": cfg["target_pad"],
                      "tin": cfg["objects"]["mentos_tin"], "spawn": cfg["task"]["spawn_by_arm"]}, indent=1,
                     default=float))
    print("config đã điền:", rpath)


if __name__ == "__main__":
    main()
