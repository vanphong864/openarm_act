#!/usr/bin/env python3
"""Look at the simulated corrective demos (generate_dart_demos.py): MuJoCo viewer and/or a GIF of the stored camera images.

  --view  replays the demo's labelled joint_commands through the physics in the MuJoCo window, with the tin at the start
          pose recorded in the generation log. The labels are the CLEAN expert commands (the way back to the plan),
          so this shows what the model is taught to do.
  --gif   writes chest | front images exactly as stored in the HDF5 (what the model sees) every 5th frame. These were
          rendered while the arm was really executing the NOISY commands, so they show the drift that was injected
          while recording (demos with "noise 0.000" have none).

  view_sim_demo.py "/media/space/New Volume/sim_demos/2026-10-13_dart_v2" --demo 3 --view --gif outputs/demo3.gif
  view_sim_demo.py "/media/space/New Volume/sim_demos/2026-10-13_dart_v2" --list
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="Folder with the simulated demo HDF5 files and *_generation_log.json")
    ap.add_argument("--raw", type=Path, default=Path("/media/space/New Volume/dataset/2026-10-13"),
                    help="Real dataset (needed for the scene calibration)")
    ap.add_argument("--config", type=Path, default=ROOT / "simulation/config/scene_20261013.yaml")
    ap.add_argument("--list", action="store_true", help="List the demos with their noise level and tin pose")
    ap.add_argument("--demo", default="0", help="Index in the sorted list, or part of the file name")
    ap.add_argument("--view", action="store_true", help="Open the interactive MuJoCo viewer")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed relative to real time (viewer)")
    ap.add_argument("--gif", type=Path, help="Write the stored chest|front images to this GIF")
    args = ap.parse_args()

    files = sorted(args.directory.glob("*.hdf5"))
    files = [f for f in files if not f.name.endswith(".partial.hdf5")]
    if not files:
        raise SystemExit(f"No demo HDF5 files in {args.directory}")
    # The generation log lists every attempt; the written demos are its successful rows, in order, per worker.
    logs = {}
    for path in args.directory.glob("*_generation_log.json"):
        prefix = path.name.removesuffix("_generation_log.json")
        rows = [r for r in json.loads(path.read_text())["demos"] if r["lifted"] and r["in_pad"]]
        logs[prefix] = rows

    def info(path):
        stem = path.stem
        for prefix, rows in logs.items():
            if stem.startswith(prefix + "_"):
                index = int(stem[len(prefix) + 1:].split("_")[0])
                if index < len(rows):
                    return rows[index]
        return None

    if args.list:
        for i, path in enumerate(files):
            r = info(path)
            extra = (f"noise {r['noise_std_rad']:.3f} rad, tin xy {r['tin_xy']}, yaw {r['tin_yaw_deg']:.0f} deg, arm {r['arm']}"
                     if r else "no log row")
            print(f"[{i:2d}] {path.name}  {extra}")
        return
    if args.demo.isdigit() and int(args.demo) < len(files):
        path = files[int(args.demo)]
    else:
        matches = [f for f in files if args.demo in f.name]
        if not matches:
            raise SystemExit(f"No demo matching {args.demo!r}; use --list")
        path = matches[0]
    row = info(path)

    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][:].astype(np.float64)
        commands = f["joint_commands"][:].astype(np.float64)
        fps = float(f.attrs.get("frequency_hz", 50))
        noise = float(f.attrs.get("sim_noise_std_rad", 0.0))
        setup = str(f.attrs.get("sim_setup_episode", ""))
        print(f"{path.name}: {len(qpos)} frames ({len(qpos) / fps:.1f} s), noise {noise:.3f} rad, setup episode {setup}")
        if args.gif:
            from PIL import Image
            frames = [Image.fromarray(np.concatenate([f["observations/images/chest_rgb"][i],
                                                      f["observations/images/front_rgb"][i]], 1)).resize((960, 360))
                      for i in range(0, len(qpos), 5)]
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=round(5000 / fps), loop=0)
            print(f"GIF with the stored images -> {args.gif}")
    if not args.view:
        return
    if row is None:
        raise SystemExit("No generation-log row for this demo (needed for the tin start pose)")

    os.environ["MUJOCO_GL"] = "glfw"
    sys.path.insert(0, str(ROOT / "scripts"))
    from simulate import clean_snap_gui_environment
    clean_snap_gui_environment()
    sys.path[:0] = [str(ROOT / "simulation"), str(ROOT / "simulation/scripts")]
    import mujoco
    import mujoco.viewer
    from openarm_sim.measured_scene import compile_measured
    from openarm_sim.runner import Sim
    from openarm_sim.scene import PAD_GEOM
    from replay_real_20261013 import load_cfg, set_object

    cfg = load_cfg(args.config, args.raw)
    scene = cfg["_measured"]["scene"]
    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    half_h = cfg["objects"]["mentos_tin"]["size"][2] / 2
    pad = m.geom(PAD_GEOM).id
    pad_xy0 = np.array(scene["pad"]["center"][:2])
    per_episode = scene["pad"].get("per_episode", {}).get(setup)
    if per_episode:   # the box and pad were at this position in the setup episode's session
        body = m.geom_bodyid[pad]
        centre = np.array(per_episode["center"])
        m.body_pos[body][:2] += centre - pad_xy0
        m.geom_pos[pad][:2] = centre - m.body_pos[body][:2]
    sim.reset(qpos16=qpos[0])
    set_object(m, sim.d, "mentos_tin", row["tin_xy"], row["tin_yaw_deg"], box_top + half_h + 1e-4)
    mujoco.mj_forward(m, sim.d)
    sim.settle(0.3)
    tin = m.body("mentos_tin").id
    print("Viewer: close the window to stop. Playing the clean labelled commands")
    with mujoco.viewer.launch_passive(m, sim.d) as viewer:
        for t in range(len(commands)):
            tick = time.monotonic()
            if not viewer.is_running():
                break
            sim.step(commands[t])
            viewer.sync()
            time.sleep(max(0.0, 1 / fps / args.speed - (time.monotonic() - tick)))
        lift = float(sim.d.xpos[tin][2] - (box_top + half_h))
        print(f"end of demo; tin at {np.round(sim.d.xpos[tin], 3).tolist()} (height above the box {lift * 1000:.0f} mm)")
        time.sleep(2.0)


if __name__ == "__main__":
    main()
