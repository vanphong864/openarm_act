#!/usr/bin/env python3
"""Corrective (DART-style) pick-and-place demos in the measured 2026-10-13 scene, in the recorder's HDF5 format.

Each demo starts from a REAL training episode's setup (robot first-frame qpos, box/pad position, per-episode
real background and colour), with the tin near its measured start pose (small random offset). The scripted
expert (openarm_sim/policy.py: min-jerk Cartesian plan + IK every tick) drives the recorded arm; the commands
actually EXECUTED carry correlated (Ornstein-Uhlenbeck) joint noise, so the arm drifts off the plan, while the
LABEL written to `joint_commands` is the clean expert command, i.e. the way back. Training on these teaches the
policy to recover from its own drift in closed loop (Laskey et al., DART, 2017). Only demos where the tin ends
in the pad, and whose arm never touched the work box on the way to the tin, are kept. Images are the same
composite cameras used for rollouts (sim robot/tin over the real background); validation episodes of the given
release are never used as setups.

Ground truth recorded with each frame (not available on the real robot; for auxiliary losses or detector labels):
  observations/object_pose/mentos_tin   [T, 7] x y z qw qx qy qz of the tin in the sim world frame (the camera
                                        calibration frame);
  observations/object_bbox/<camera>     [T, 5] x0 y0 x1 y1 (inclusive, source pixels) and visible pixel count of
                                        the tin in that camera's image (all -1 / 0 when not visible);
and the pad's pose in the file attributes (sim_pad_pose, sim_pad_half_size).

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
from openarm_sim.arm import GRIP_OPEN, grasp_rot  # noqa: E402
from openarm_sim.composite_camera import CompositeCamera  # noqa: E402
from openarm_sim.measured_scene import compile_measured  # noqa: E402
from openarm_sim.policy import ABOVE, GRASP, LIFTED, PLACE_ABOVE, RETREAT, ScriptedPickPlace  # noqa: E402
from openarm_sim.runner import Sim  # noqa: E402
from openarm_sim.scene import BOX_GEOM, PAD_GEOM  # noqa: E402
from replay_real_20261013 import load_cfg, set_object  # noqa: E402

HOLD_BEFORE, HOLD_AFTER = 25, 25   # control ticks of standing still before/after the expert plan


def pad_quat(xmat):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(xmat, np.float64).reshape(9))
    return quat


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
    ap.add_argument("--near-fraction", type=float, default=0.0,
                    help="Share of demos whose arm starts NEAR the tin instead of at the recorded home pose: fingertips "
                         "--near-height above the grasp point, offset sideways (std --near-xy) and yawed (+-"
                         "--near-yaw), then the expert runs from there. The clean label is the way to the grasp, so "
                         "the policy sees many misaligned states close to the tin and learns to correct them (run9-11 "
                         "approached 40-90 mm off, closed on air or never reached a graspable pose)")
    ap.add_argument("--near-height", type=float, nargs=2, default=[0.025, 0.10], metavar=("MIN", "MAX"),
                    help="TCP height (m) above the grasp point at the start of a near-start demo")
    ap.add_argument("--near-xy", type=float, default=0.025, help="Std (m) of the sideways start offset (clipped at 2 std)")
    ap.add_argument("--near-yaw", type=float, default=15.0, help="Max gripper yaw error (deg) at the start")
    ap.add_argument("--near-approach-s", type=float, default=0.8, help="Duration (s) of the first plan segment")
    ap.add_argument("--arm", choices=["left", "right"],
                    help="Only setups recorded with this arm (the expert fails more often with the right arm, so "
                         "balanced data needs more right-arm attempts)")
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
    if args.arm:
        def recorded_arm(ep):
            try:
                with h5py.File(args.raw / f"{ep}.hdf5", "r") as f:
                    return "left" if f["observations/qpos"][:, 7].min() < 0.035 else "right"
            except OSError:   # unreadable (truncated) file
                return None
        setups = [ep for ep in setups if recorded_arm(ep) == args.arm]
    if not setups:
        raise SystemExit("No usable setup episode")
    print(f"{len(setups)} setup episodes ({len(excluded)} validation episodes excluded)", flush=True)

    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    tin_body = m.body("mentos_tin").id
    tin_geom = m.geom("mentos_tin_geom").id
    tin_half = m.geom_size[tin_geom].copy()
    tin_geoms = np.flatnonzero(m.geom_bodyid == tin_body)   # box + top label
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    pad = m.geom(PAD_GEOM).id
    work_box = m.geom(BOX_GEOM).id
    arm_geoms = {s: {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"openarm_{s}_")}
                 for s in ("left", "right")}
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
        side = slice(0, 8) if arm_side == "left" else slice(8, 16)
        target, hold_arm = None, None

        def build(q_start, approach_s=None, compensate=False):
            """Robot at q_start (encoder units), tin at (xy, yaw), expert planned from there. compensate: the arm
            stands in a stretched pose, so hold it with the gravity-sag-compensated command (as the expert does);
            otherwise the soft servos let it sag ~10 cm during the settle. hold_arm = that command (encoder units)."""
            nonlocal target, hold_arm
            sim.reset(qpos16=q_start)
            set_object(m, sim.d, "mentos_tin", xy, yaw, box_top + tin_half[2] + 1e-4)
            mujoco.mj_forward(m, sim.d)
            hold_arm = None
            if compensate:
                seven = slice(side.start, side.start + 7)
                command = sim.arms[arm_side].sag_compensated(sim.d, np.asarray(q_start, float)[seven] + sim.offset[seven])
                sim.arms[arm_side].set_ctrl(sim.d, np.append(command, GRIP_OPEN))
                hold_arm = command - sim.offset[seven]
                if sim.limiter is not None:      # start the rate limiter at the compensated command: no transient
                    sim.limiter.target[seven] = sim.limiter.command[seven] = command
            sim.settle(0.3)
            target = sim.d.geom_xpos[pad].copy()
            target[2] = box_top + 2 * m.geom_size[pad][2] + tin_half[2]
            return ScriptedPickPlace(sim.arms[arm_side], sim.d, sim.d.xpos[tin_body].copy(), np.radians(yaw), target,
                                     fps, rng=rng, obj_half=tuple(tin_half), target_half=m.geom_size[pad][:2].copy(),
                                     obj_body=tin_body, approach_s=approach_s)

        near, near_err = rng.random() < args.near_fraction, 0.0
        try:
            expert = build(q0)
            if near:
                # Plan once from the recorded home pose to learn the grasp pose, then restart the arm near it.
                grasp_wp = expert.wp[GRASP]
                dxy = np.clip(rng.normal(0, args.near_xy, 2), -2 * args.near_xy, 2 * args.near_xy)
                start_pos = grasp_wp.pos + [dxy[0], dxy[1], rng.uniform(*args.near_height)]
                start_rot = grasp_rot(grasp_wp.yaw + np.radians(rng.uniform(-args.near_yaw, args.near_yaw)), *grasp_wp.tilt)
                q_start, err = sim.arms[arm_side].ik(sim.d, start_pos, start_rot, q_init=expert._q_seed, iters=300)
                if err > 2e-3:
                    print(f"attempt {attempts} ({setup}): near start unreachable ({err * 1000:.1f} mm)", flush=True)
                    continue
                q16 = np.asarray(q0, float).copy()
                q16[side.start:side.start + 7] = q_start - sim.offset[side.start:side.start + 7]   # model -> encoder units
                expert = build(q16, args.near_approach_s, compensate=True)
                near_err = float(np.linalg.norm(sim.arms[arm_side].tcp_pose(sim.d)[0] - start_pos))
                if near_err > 0.015:
                    print(f"attempt {attempts} ({setup}): near start sagged {near_err * 1000:.0f} mm; skipped", flush=True)
                    continue
        except (ValueError, RuntimeError) as exc:
            print(f"attempt {attempts} ({setup}): no plan: {exc}", flush=True)
            continue
        sigma = 0.0 if rng.random() < args.clean_fraction else rng.uniform(*args.noise)
        # Drift only on the way: from just before the descent to the end of the lift, and while lowering onto and
        # releasing over the pad, the arm runs the clean commands. The expert closes the gripper on a fixed schedule,
        # so injected drift there would teach "close when it is time" instead of "close at the tin" (run6).
        tw = expert.times
        quiet_windows = [] if args.noisy_grasp else [(tw[ABOVE] - 0.5, tw[LIFTED] + 0.2),
                                                  (tw[PLACE_ABOVE] - 0.2, tw[RETREAT])]
        hold = sim.qpos16().copy()   # encoder units: commands that keep the robot where it is
        if hold_arm is not None:
            hold[side.start:side.start + 7] = hold_arm
        noise = np.zeros(7)
        # Images go straight to disk frame by frame (a demo is ~3 MB/frame x ~600 frames: too large to hold in RAM).
        partial = args.out / f"{args.prefix}_in_progress.partial.hdf5"
        f = h5py.File(partial, "w")
        images = {}
        qpos_log, command_log, pose_log = [], [], []
        bbox_log = {name: [] for name in cameras}
        lifted, box_contact, keep = False, 0, False
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
                    ys, xs = np.nonzero(np.isin(camera.last_segmentation, tin_geoms))
                    bbox_log[name].append([xs.min(), ys.min(), xs.max(), ys.max(), len(xs)] if len(xs)
                                          else [-1, -1, -1, -1, 0])
                pose_log.append(np.r_[sim.d.xpos[tin_body], sim.d.xquat[tin_body]].astype(np.float32))
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
                if t_plan < tw[ABOVE]:   # on the way to the tin the arm must not touch the work box (scraping its side)
                    box_contact += any({c.geom1, c.geom2} & arm_geoms[arm_side] and work_box in (c.geom1, c.geom2)
                                       for c in sim.d.contact[:sim.d.ncon])
            final = sim.d.xpos[tin_body].copy()
            local = (final - sim.d.geom_xpos[pad]) @ sim.d.geom_xmat[pad].reshape(3, 3)
            in_pad = bool(abs(local[0]) <= m.geom_size[pad][0] and abs(local[1]) <= m.geom_size[pad][1]
                          and final[2] < box_top + 0.05)
            n = len(qpos_log)
            row = {"attempt": attempts, "setup": setup, "arm": arm_side, "noise_std_rad": round(float(sigma), 4),
                   "tin_xy": np.round(xy, 4).tolist(), "tin_yaw_deg": round(float(yaw), 1), "near_start": near, "near_start_err_mm": round(near_err * 1000, 1) if near else None, "lifted": lifted,
                   "in_pad": in_pad, "box_contact_steps": box_contact, "steps": n, "seconds": round(time.monotonic() - started, 1)}
            log.append(row)
            print(json.dumps(row), flush=True)
            keep = lifted and in_pad and not box_contact
            if keep:
                for key, value in attrs.items():
                    f.attrs[key] = value
                f.attrs.update(sim=True, complete=True, samples=n, duration_seconds=n / fps, frequency_hz=float(fps),
                               created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               sim_setup_episode=setup, sim_noise_std_rad=float(sigma), sim_near_start=near,
                               sim_generator="simulation/scripts/generate_dart_demos.py",
                               sim_tin_half_size=tin_half.tolist(), sim_pad_half_size=m.geom_size[pad].tolist(),
                               sim_pad_pose=np.r_[sim.d.geom_xpos[pad], pad_quat(sim.d.geom_xmat[pad])].tolist())
                qpos = np.stack(qpos_log)
                f.create_dataset("observations/qpos", data=qpos)
                f.create_dataset("action", data=qpos)            # the recorder's `action` is the measured qpos
                f.create_dataset("joint_commands", data=np.stack(command_log))
                f.create_dataset("timestamp_ns", data=np.arange(n, dtype=np.int64) * int(1e9 / fps))
                f.create_dataset("observations/object_pose/mentos_tin", data=np.stack(pose_log))
                for name, boxes in bbox_log.items():
                    f.create_dataset(f"observations/object_bbox/{name}", data=np.asarray(boxes, np.int32))
        finally:
            f.close()
        if not keep:
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
