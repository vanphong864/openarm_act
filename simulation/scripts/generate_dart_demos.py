#!/usr/bin/env python3
"""Corrective (DART-style) pick-and-place demos in the measured 2026-10-13 scene, in the recorder's HDF5 format.

Each demo starts from a REAL training episode's setup (robot first-frame qpos, box/pad position, per-episode
real background and colour), with the tin near its measured start pose (small random offset). The scripted
expert (openarm_sim/policy.py: min-jerk Cartesian plan + IK every tick) drives the recorded arm; the commands
actually EXECUTED carry correlated (Ornstein-Uhlenbeck) joint noise, so the arm drifts off the plan, while the
LABEL written to `joint_commands` is the clean expert command, i.e. the way back. Training on these teaches the
policy to recover from its own drift in closed loop (Laskey et al., DART, 2017). Only demos where the tin ends
in the pad are kept. Images are the same composite cameras used for rollouts (sim robot/tin over the real
background); validation episodes of the given release are never used as setups.

  generate_dart_demos.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --exclude-val-of releases/meanflow_v2_run3/best_release --demos 30 \
      --out "/media/space/New Volume/sim_demos/2026-10-13_dart"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulation"), str(ROOT / "simulation/scripts")]
from openarm_sim.composite_camera import CompositeCamera  # noqa: E402
from openarm_sim.measured_scene import compile_measured  # noqa: E402
from openarm_sim.policy import ScriptedPickPlace  # noqa: E402
from openarm_sim.runner import Sim  # noqa: E402
from openarm_sim.scene import PAD_GEOM  # noqa: E402
from replay_real_20261013 import load_cfg, set_object  # noqa: E402

HOLD_BEFORE, HOLD_AFTER = 25, 25   # control ticks of standing still before/after the expert plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="Real dataset dir (setups, attributes, backgrounds)")
    ap.add_argument("--config", type=Path, default=ROOT / "simulation/config/scene_20261013.yaml")
    ap.add_argument("--exclude-val-of", type=Path, help="Release dir: its validation episodes are never used")
    ap.add_argument("--demos", type=int, default=30, help="Successful demos to write")
    ap.add_argument("--max-attempts", type=int, help="Default: 3 x --demos")
    ap.add_argument("--noise", type=float, nargs=2, default=[0.02, 0.06], metavar=("MIN", "MAX"),
                    help="Stationary std (rad) of the joint noise, drawn per demo")
    ap.add_argument("--clean-fraction", type=float, default=0.2, help="Share of demos executed without noise")
    ap.add_argument("--noise-time", type=float, default=0.3, help="Noise correlation time (s)")
    ap.add_argument("--tin-offset", type=float, default=0.015, help="Std (m) of the tin position offset")
    ap.add_argument("--tin-yaw", type=float, default=20.0, help="Max tin yaw offset (deg)")
    ap.add_argument("--tin-from", choices=["measured", "fitted"], default="fitted",
                    help="Centre of the tin start poses: depth-measured, or the object-feature estimate on the real "
                         "first frame (what the policy's object features see; keeps sim and real features consistent)")
    ap.add_argument("--noisy-grasp", action="store_true",
                    help="Also inject drift while grasping and placing (v2 behaviour; teaches schedule-based closing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="sim_dart", help="File name prefix (distinct per parallel worker)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = load_cfg(args.config, args.raw)
    fps = cfg["task"]["fps"]
    scene = cfg["_measured"]["scene"]
    starts = {r["episode"]: r for r in scene["tin_poses"]["start"]}
    excluded = set()
    if args.exclude_val_of:
        meta = json.loads((args.exclude_val_of / "metadata.json").read_text())
        excluded = {e["id"] for e in meta["episodes"] if e["split"] == "val"}
    setups = [ep for ep in sorted(starts) if ep not in excluded and (args.raw / f"{ep}.hdf5").exists()]
    if not setups:
        raise SystemExit("No usable setup episode")
    print(f"{len(setups)} setup episodes ({len(excluded)} validation episodes excluded)", flush=True)

    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    tin_body = m.body("mentos_tin").id
    tin_geom = m.geom("mentos_tin_geom").id
    tin_half = m.geom_size[tin_geom].copy()
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    pad = m.geom(PAD_GEOM).id
    box_xy0 = m.body_pos[m.geom_bodyid[pad]][:2].copy()
    pad_xy0 = np.array(scene["pad"]["center"][:2])
    background_dir = (args.config.parent / cfg["background_dir"]).resolve()
    names = cfg["_measured"]["cameras"]["cameras"]
    cameras = {c: CompositeCamera.from_config(m, c, background_dir, cfg.get("composite"),
                                              rng=np.random.default_rng(args.seed + 1))
               for c in names}
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from openarm_flow.object_features import ObjectConfig, SceneEstimator
    estimator = SceneEstimator(ObjectConfig.from_measurements())
    theta = 1.0 / (args.noise_time * fps)
    written, attempts, log = 0, 0, []
    max_attempts = args.max_attempts or 3 * args.demos
    while written < args.demos and attempts < max_attempts:
        attempts += 1
        setup = setups[int(rng.integers(len(setups)))]
        started = time.monotonic()
        with h5py.File(args.raw / f"{setup}.hdf5", "r") as f:
            q0 = f["observations/qpos"][0].astype(np.float64)
            arm_side = "left" if f["observations/qpos"][:, 7].min() < 0.035 else "right"
            attrs = {k: f.attrs[k] for k in f.attrs}
            first_rgb = {c: f[f"observations/images/{c}_rgb"][0] for c in names}
        per_episode = scene["pad"].get("per_episode", {}).get(setup)
        body = m.geom_bodyid[pad]
        center = np.array(per_episode["center"]) if per_episode else pad_xy0
        m.body_pos[body][:2] = box_xy0 + (center - pad_xy0)
        m.geom_pos[pad][:2] = center - m.body_pos[body][:2]
        if per_episode:
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.array([0, 0, np.radians(per_episode["yaw_deg"])]), "xyz")
            m.geom_quat[pad] = quat
        for name, camera in cameras.items():
            per_background = background_dir / "bg_episodes" / f"background_{name}_{setup}_rgb.png"
            camera.set_background(f"_{setup}", per_background.parent) if per_background.exists() else camera.set_background("")
            camera.match_color(first_rgb[name])
        base_xy, base_yaw = np.asarray(starts[setup]["xy"], float), starts[setup]["yaw_deg"]
        if args.tin_from == "fitted":
            fitted = estimator.estimate(first_rgb)["tin"]
            if fitted is not None:
                base_xy, base_yaw = np.asarray(fitted[:2]), np.degrees(fitted[2])
        xy = base_xy + np.clip(rng.normal(0, args.tin_offset, 2), -2 * args.tin_offset, 2 * args.tin_offset)
        yaw = base_yaw + rng.uniform(-args.tin_yaw, args.tin_yaw)
        sim.reset(qpos16=q0)
        set_object(m, sim.d, "mentos_tin", xy, yaw, box_top + tin_half[2] + 1e-4)
        mujoco.mj_forward(m, sim.d)
        sim.settle(0.3)
        side = slice(0, 8) if arm_side == "left" else slice(8, 16)
        target = sim.d.geom_xpos[pad].copy()
        target[2] = box_top + 2 * m.geom_size[pad][2] + tin_half[2]
        try:
            expert = ScriptedPickPlace(sim.arms[arm_side], sim.d, sim.d.xpos[tin_body].copy(), np.radians(yaw), target,
                                       fps, rng=rng, obj_half=tuple(tin_half), target_half=m.geom_size[pad][:2].copy(),
                                       obj_body=tin_body)
        except (ValueError, RuntimeError) as exc:
            print(f"attempt {attempts} ({setup}): no plan: {exc}", flush=True)
            continue
        sigma = 0.0 if rng.random() < args.clean_fraction else rng.uniform(*args.noise)
        # Drift only on the way: from just before the descent to the end of the lift, and while lowering onto and
        # releasing over the pad, the arm runs the clean commands. The expert closes the gripper on a fixed schedule,
        # so injected drift there would teach "close when it is time" instead of "close at the tin" (run6).
        tw = expert.times
        quiet_windows = [] if args.noisy_grasp else [(tw[1] - 0.5, tw[4] + 0.2), (tw[5] - 0.2, tw[8])]
        hold = sim.qpos16().copy()   # encoder units: commands that keep the robot where it is
        noise = np.zeros(7)
        # Images go straight to disk frame by frame (a demo is ~3 MB/frame x ~600 frames: too large to hold in RAM).
        partial = args.out / f"{args.prefix}_in_progress.partial.hdf5"
        f = h5py.File(partial, "w")
        images = {}
        qpos_log, command_log = [], []
        lifted = False
        total = HOLD_BEFORE + expert.n_steps + HOLD_AFTER
        try:
            for step in range(total):
                label = hold.copy()
                if step >= HOLD_BEFORE:
                    act = expert(sim.d)                      # model units: 7 joints + gripper stroke
                    label[side] = act
                    label[side.start:side.start + 7] -= sim.offset[side.start:side.start + 7]
                for name, camera in cameras.items():
                    for kind, frame in zip(("rgb", "depth"), camera.observe(sim.d)):
                        key = f"observations/images/{name}_{kind}"
                        if key not in images:
                            images[key] = f.create_dataset(key, shape=(0, *frame.shape), maxshape=(None, *frame.shape),
                                                           dtype=frame.dtype, chunks=(1, *frame.shape), compression="lzf")
                        images[key].resize(len(qpos_log) + 1, axis=0)
                        images[key][len(qpos_log)] = frame
                qpos_log.append(sim.qpos16().astype(np.float32))
                command_log.append(label.astype(np.float32))
                executed = label.copy()
                t_plan = (step - HOLD_BEFORE) / fps
                quiet = any(a <= t_plan <= b for a, b in quiet_windows)
                if sigma and HOLD_BEFORE <= step < HOLD_BEFORE + expert.n_steps and not quiet:
                    noise += -theta * noise + sigma * np.sqrt(2 * theta) * rng.normal(size=7)
                    executed[side.start:side.start + 7] += noise
                elif quiet:
                    noise[:] = 0   # no injected drift while descending/closing/lifting or placing/releasing
                sim.step(executed)
                if not np.isfinite(sim.d.qpos).all():
                    break
                lifted |= bool(sim.d.xpos[tin_body][2] > box_top + tin_half[2] + 0.03)
            final = sim.d.xpos[tin_body].copy()
            local = (final - sim.d.geom_xpos[pad]) @ sim.d.geom_xmat[pad].reshape(3, 3)
            in_pad = bool(abs(local[0]) <= m.geom_size[pad][0] and abs(local[1]) <= m.geom_size[pad][1]
                          and final[2] < box_top + 0.05)
            n = len(qpos_log)
            row = {"attempt": attempts, "setup": setup, "arm": arm_side, "noise_std_rad": round(float(sigma), 4),
                   "tin_xy": np.round(xy, 4).tolist(), "tin_yaw_deg": round(float(yaw), 1), "lifted": lifted,
                   "in_pad": in_pad, "steps": n, "seconds": round(time.monotonic() - started, 1)}
            log.append(row)
            print(json.dumps(row), flush=True)
            if lifted and in_pad:
                for key, value in attrs.items():
                    f.attrs[key] = value
                f.attrs.update(sim=True, complete=True, samples=n, duration_seconds=n / fps, frequency_hz=float(fps),
                               created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               sim_setup_episode=setup, sim_noise_std_rad=float(sigma),
                               sim_generator="simulation/scripts/generate_dart_demos.py")
                qpos = np.stack(qpos_log)
                f.create_dataset("observations/qpos", data=qpos)
                f.create_dataset("action", data=qpos)            # the recorder's `action` is the measured qpos
                f.create_dataset("joint_commands", data=np.stack(command_log))
                f.create_dataset("timestamp_ns", data=np.arange(n, dtype=np.int64) * int(1e9 / fps))
        finally:
            f.close()
        if not (lifted and in_pad):
            partial.unlink()
            continue
        partial.replace(args.out / f"{args.prefix}_{written:03d}_{setup.removeprefix('episode_')}.hdf5")
        written += 1
    for camera in cameras.values():
        camera.close()
    (args.out / f"{args.prefix}_generation_log.json").write_text(json.dumps({"written": written, "attempts": attempts,
                                                              "args": {k: str(v) for k, v in vars(args).items()},
                                                              "demos": log}, indent=1))
    print(f"wrote {written} demos in {attempts} attempts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
