#!/usr/bin/env python3
"""Play back recorded sim rollouts (rollout_meanflow.py npz files) in the MuJoCo viewer at real speed.

The closed-loop rollout renders two composited cameras and runs the policy every tick, far slower than 50 Hz
on a laptop; this replays the saved MuJoCo state instead, so the motion is smooth. Needs npz files written
with `sim_qpos` (rollout_meanflow.py from 2026-11-03 on).

  view_rollout.py outputs/sim2_run12_view_gpu --raw "/media/space/New Volume/dataset/2026-10-13"
  view_rollout.py outputs/sim2_run12/episode_20261013_110412.npz --speed 0.5 --loop
"""

import argparse
import os
from pathlib import Path
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="Rollout npz files or directories of them")
    ap.add_argument("--raw", type=Path, required=True, help="Dataset dir (scene calibration, as for the rollout)")
    ap.add_argument("--config", type=Path, default=root / "simulation/config/scene_20261013.yaml")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (1 = real time)")
    ap.add_argument("--pause", type=float, default=1.0, help="Seconds to hold the last frame of each episode")
    ap.add_argument("--loop", action="store_true", help="Start again after the last episode")
    args = ap.parse_args()
    if args.speed <= 0:
        ap.error("--speed must be positive")

    sys.path.insert(0, str(root / "scripts"))
    from simulate import clean_snap_gui_environment
    clean_snap_gui_environment()
    os.environ["MUJOCO_GL"] = "glfw"
    sys.path.insert(0, str(root / "simulation"))
    sys.path.insert(0, str(root / "simulation/scripts"))
    import mujoco
    import mujoco.viewer
    import numpy as np
    from openarm_sim.measured_scene import compile_measured
    from openarm_sim.scene import PAD_GEOM
    from replay_real_20261013 import load_cfg

    files = [f for p in args.paths for f in (sorted(p.glob("*.npz")) if p.is_dir() else [p])]
    if not files:
        ap.error("no npz files found")
    recordings = []
    for f in files:
        z = np.load(f)
        if "sim_qpos" not in z.files or not len(z["sim_qpos"]):
            print(f"{f.name}: no sim_qpos (recorded before 2026-11-03); rerun rollout_meanflow.py", flush=True)
            continue
        recordings.append((f.stem, {k: z[k] for k in z.files}))
    if not recordings:
        raise SystemExit("Nothing to play")

    m = compile_measured(load_cfg(args.config, args.raw))
    d = mujoco.MjData(m)
    pad = m.geom(PAD_GEOM).id
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            for name, rec in recordings:
                if rec["sim_qpos"].shape[1] != m.nq:
                    raise ValueError(f"{name}: recorded qpos has {rec['sim_qpos'].shape[1]} values, model has {m.nq}")
                m.body_pos[m.geom_bodyid[pad]] = rec["pad_body_pos"]
                m.geom_pos[pad], m.geom_quat[pad] = rec["pad_geom_pos"], rec["pad_geom_quat"]
                dt = 1 / float(rec["fps"]) / args.speed
                print(f"{name}: {len(rec['sim_qpos'])} steps", flush=True)
                start = time.monotonic()
                for i, q in enumerate(rec["sim_qpos"]):
                    if not viewer.is_running():
                        return
                    d.qpos[:] = q
                    mujoco.mj_forward(m, d)
                    viewer.sync()
                    time.sleep(max(0.0, start + (i + 1) * dt - time.monotonic()))
                time.sleep(args.pause)
            if not args.loop:
                break


if __name__ == "__main__":
    main()
