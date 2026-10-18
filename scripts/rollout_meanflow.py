#!/usr/bin/env python3
"""Closed-loop MeanFlow v2 (two cameras) rollout in the measured 2026-10-13 MuJoCo scene.

Per episode the robot starts at the recorded first-frame qpos, the Mentos tin at the pose measured
from the cameras in that frame, and the box/pad at that session's position. The policy sees only
simulated chest/front depth (+RGB if the release uses it) and physics qpos at every 50 Hz tick;
no demonstration commands are used. Chest/front images are composited over the real background
(simulation/openarm_sim/composite_camera.py), so the static scene looks like the recording.

  rollout_meanflow.py --checkpoint outputs/run/best_release \
      --raw "/media/space/New Volume/dataset/2026-10-13" --out outputs/sim_meanflow
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="MeanFlow release dir (model.pt, config.json, metadata.json)")
    ap.add_argument("--raw", type=Path, required=True, help="Dataset dir; used for episode start states and calibration")
    ap.add_argument("--config", type=Path, default=root / "simulation/config/scene_20261013.yaml")
    ap.add_argument("--episodes", nargs="*", help="Episode ids (default: the calibration held-out episodes)")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--seconds", type=float, help="Rollout length; default is the recorded episode length")
    ap.add_argument("--execute-steps", type=int, help="Actions executed per inference (default: policy chunk size)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--view", action="store_true", help="Open the interactive MuJoCo viewer (real-time pacing)")
    ap.add_argument("--out", type=Path, default=root / "outputs/sim_meanflow")
    args = ap.parse_args()
    if args.seconds is not None and args.seconds < 0.02:
        ap.error("--seconds must be positive")

    if args.view:
        os.environ["MUJOCO_GL"] = "glfw"
        sys.path.insert(0, str(root / "scripts"))
        from simulate import clean_snap_gui_environment
        clean_snap_gui_environment()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "simulation"))
    sys.path.insert(0, str(root / "simulation/scripts"))
    import h5py
    import mujoco
    import numpy as np
    import torch
    from PIL import Image
    from openarm_sim.composite_camera import CompositeCamera
    from openarm_sim.measured_scene import compile_measured
    from openarm_sim.runner import Sim
    from openarm_sim.scene import PAD_GEOM
    from replay_real_20261013 import load_cfg, set_object
    from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    policy = OpenArmMeanFlowPolicy(args.checkpoint, args.device, seed=args.seed)
    cfg = load_cfg(args.config, args.raw)
    if int(policy.metadata["fps"]) != cfg["task"]["fps"]:
        raise ValueError("Policy and simulator control rates must match")
    scene = cfg["_measured"]["scene"]
    camera_names = cfg["_measured"]["cameras"]["cameras"]
    wanted = {c.removesuffix("_depth") for c in policy.cameras}
    wanted |= {c.removesuffix("_rgb") for c in policy.rgb_config.cameras} if policy.rgb_config else set()
    if wanted - set(camera_names):
        raise ValueError(f"Policy needs cameras {sorted(wanted)}; the scene provides {camera_names}")
    episodes = args.episodes or cfg["_measured"]["cameras"]["heldout_episodes"]
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    starts = {r["episode"]: r for r in scene["tin_poses"]["start"]}

    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    obj = "mentos_tin"
    obj_body = m.body(obj).id
    half_h = cfg["objects"][obj]["size"][2] / 2
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    pad = m.geom(PAD_GEOM).id
    box_xy0 = m.body_pos[m.geom_bodyid[pad]][:2].copy()
    pad_xy0 = np.array(scene["pad"]["center"][:2])
    background_dir = (args.config.parent / cfg["background_dir"]).resolve()
    cameras = {c: CompositeCamera(m, c, background_dir, rng=np.random.default_rng(args.seed)) for c in camera_names}
    execute_steps = min(args.execute_steps or policy.chunk_size, policy.chunk_size)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(m, sim.d)
    for episode in episodes:
        if viewer is not None and not viewer.is_running():
            break
        if episode not in starts:
            print(f"{episode}: no measured start pose for the tin; skipped", flush=True)
            continue
        with h5py.File(args.raw / f"{episode}.hdf5", "r") as f:
            q0 = f["observations/qpos"][0].astype(np.float64)
            recorded_steps = f["observations/qpos"].shape[0]
            recorded_arm = "left" if f["observations/qpos"][:, 7].min() < 0.035 else "right"
        per_episode = scene["pad"].get("per_episode", {}).get(episode)
        if per_episode:  # box and pad shift between sessions
            body = m.geom_bodyid[pad]
            m.body_pos[body][:2] = box_xy0 + (np.array(per_episode["center"]) - pad_xy0)
            m.geom_pos[pad][:2] = np.array(per_episode["center"]) - m.body_pos[body][:2]
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, np.array([0, 0, np.radians(per_episode["yaw_deg"])]), "xyz")
            m.geom_quat[pad] = quat
        for name, camera in cameras.items():
            per_background = background_dir / "bg_episodes" / f"background_{name}_{episode}_rgb.png"
            camera.set_background(f"_{episode}", per_background.parent) if per_background.exists() else camera.set_background("")
        with h5py.File(args.raw / f"{episode}.hdf5", "r") as f:
            gains = {name: camera.match_color(f[f"observations/images/{name}_rgb"][0]).round(3).tolist()
                     for name, camera in cameras.items()}
        sim.reset(qpos16=q0)
        set_object(m, sim.d, obj, starts[episode]["xy"], starts[episode]["yaw_deg"], box_top + half_h + 1e-4)
        mujoco.mj_forward(m, sim.d)
        sim.settle(0.2)
        policy.reset()
        total = round(args.seconds * sim.fps) if args.seconds else recorded_steps
        pad_center, pad_half, pad_rot = sim.d.geom_xpos[pad].copy(), m.geom_size[pad][:2].copy(), sim.d.geom_xmat[pad].reshape(3, 3).copy()
        states, actions, objects, frames = [], [], [], []
        max_lift, chunk, started = 0.0, None, time.monotonic()
        for step in range(total):
            tick = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            images = {}
            for name, camera in cameras.items():
                rgb, depth = camera.observe(sim.d)
                images[f"{name}_depth"], images[f"{name}_rgb"] = depth, rgb
            policy.observe({"images": images, "state": sim.qpos16()})
            if step % execute_steps == 0:
                chunk = np.asarray(policy.infer()["actions"])
            action = chunk[step % execute_steps]
            states.append(sim.qpos16().copy())
            actions.append(action.copy())
            if not args.no_gif and step % 5 == 0:
                frames.append(Image.fromarray(np.concatenate([images["chest_rgb"], images["front_rgb"]], 1)).resize((640, 240)))
            sim.step(action)
            if not np.isfinite(sim.d.qpos).all():
                raise ValueError("Simulation produced nonfinite joint positions")
            position = sim.d.xpos[obj_body].copy()
            objects.append(position)
            max_lift = max(max_lift, float(position[2] - (box_top + half_h)))
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, 1 / sim.fps - (time.monotonic() - tick)))
        if not states:  # viewer closed before the first step
            break
        final = sim.d.xpos[obj_body].copy()
        local = (final - pad_center) @ pad_rot
        in_pad = bool(abs(local[0]) <= pad_half[0] and abs(local[1]) <= pad_half[1] and final[2] < box_top + 0.05)
        states, actions = np.asarray(states), np.asarray(actions)
        active = slice(0, 7) if recorded_arm == "left" else slice(8, 15)
        grip = 7 if recorded_arm == "left" else 15
        row = {"episode": episode, "recorded_arm": recorded_arm, "steps": len(states),
               "lifted": max_lift > 0.03, "max_lift_m": max_lift, "in_pad": in_pad,
               "tin_end_xy": final[:2].round(4).tolist(),
               "background_color_gain": gains,
               "min_active_gripper_mm": float(states[:, grip].min() * 1000),
               "active_arm_joint_range_rad": float(np.ptp(states[:, active], axis=0).max()),
               "inactive_arm_moved_rad": float(np.ptp(states[:, 8:15] if recorded_arm == "left" else states[:, :7], axis=0).max()),
               "wall_seconds": time.monotonic() - started}
        rows.append(row)
        print(json.dumps(row), flush=True)
        np.savez_compressed(args.out / f"{episode}.npz", qpos=states, actions=actions, object_xyz=np.asarray(objects))
        if frames:
            frames[0].save(args.out / f"{episode}.gif", save_all=True, append_images=frames[1:],
                           duration=round(5000 / sim.fps), loop=0)
    for camera in cameras.values():
        camera.close()
    if viewer is not None:
        viewer.close()
    if not rows:
        raise SystemExit("No episode was rolled out")
    summary = {"checkpoint": str(args.checkpoint), "step": policy.metadata.get("step"), "use_rgb": policy.policy.use_rgb,
               "execute_steps": execute_steps, "seed": args.seed, "episodes": len(rows),
               "lift_rate": float(np.mean([r["lifted"] for r in rows])),
               "in_pad_rate": float(np.mean([r["in_pad"] for r in rows])),
               "note": "closed loop; observations are simulated, actions come only from the policy"}
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
