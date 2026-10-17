#!/usr/bin/env python3
"""Run a FlowPolicy OpenArm release in the imported MuJoCo workspace."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

from simulate import clean_snap_gui_environment


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--policy", choices=["auto", "flowpolicy"], default="auto")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--out", type=Path, default=root / "outputs/mujoco_policy")
    ap.add_argument("--execute-steps", type=int, default=25, help="Actions executed before observing again")
    ap.add_argument("--seed", type=int, default=42, help="Seed for model sampling")
    ap.add_argument("--object-xy", type=float, nargs=2,
                    help="Override only initial task-object XY, preserving Z and orientation")
    ap.add_argument("--initial-qpos", type=float, nargs=16,
                    help="Optional initial robot state; all subsequent states come from physics")
    args = ap.parse_args()
    if args.seconds < 0.02 or args.execute_steps < 1:
        ap.error("--seconds and --execute-steps must be positive")
    clean_snap_gui_environment()
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.view else "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(root / "simulation"))
    sys.path.insert(0, str(root / "src"))
    import mujoco
    import numpy as np
    import torch
    from PIL import Image
    from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE
    from openarm_sim.render import OffscreenRenderer
    from openarm_sim.runner import Sim
    from openarm_sim.scene import load_config, load_scene
    from openarm_sim.sensor import CalibratedCamera
    from openarm_sim.task import PickPlaceTask

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    kind = args.policy
    if kind == "auto":
        metadata_file = args.checkpoint / "metadata.json"
        kind = json.loads(metadata_file.read_text()).get("policy_type") if metadata_file.exists() else None
    if kind != "flowpolicy":
        ap.error("This runner requires a FlowPolicy release; use scripts/rollout_meanflow.py for MeanFlow")
    from openarm_flow.policy import OpenArmFlowPolicy
    policy = OpenArmFlowPolicy(args.checkpoint, device=args.device)
    if set(policy.cameras) - {"chest", "chest_depth"}:
        raise ValueError(f"Unsupported cameras: {policy.cameras}")
    cfg = load_config(DEFAULT_CONFIG)
    if int(policy.metadata["fps"]) != cfg["task"]["fps"]:
        raise ValueError("Policy and simulator control rates must match")
    model, _ = load_scene(DEFAULT_SCENE)
    c = policy.camera
    sim = Sim(model, cfg)
    initial_qpos = None if args.initial_qpos is None else np.asarray(args.initial_qpos)
    if initial_qpos is not None and not np.isfinite(initial_qpos).all():
        ap.error("Initial qpos must be finite")
    sim.reset(qpos16=initial_qpos)
    task = PickPlaceTask(model, cfg)
    if args.object_xy is not None:
        if not np.isfinite(args.object_xy).all():
            ap.error("Object XY must be finite")
        sim.d.qpos[task.obj_qadr:task.obj_qadr + 2] = args.object_xy
        mujoco.mj_forward(model, sim.d)
    initial_state = sim.qpos16().copy()
    initial_object = task.state(sim.d).obj.copy()
    policy.reset()
    torch.manual_seed(args.seed)
    execute_steps = min(args.execute_steps, policy.chunk_size)
    total_steps = round(args.seconds * sim.fps)
    args.out.mkdir(parents=True, exist_ok=True)
    frames, states, actions, predicted_chunks = [], [], [], []
    task_states, object_positions, tcp_positions = [], [], []
    clipped_counts = np.zeros(16, dtype=int)
    minimum_finger_gap = float("inf")
    first_closed_time = None
    grip_index = 15 if cfg["task"]["arm"] == "right" else 7
    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0.2, 0, 0.3]
    overview.distance, overview.azimuth, overview.elevation = 2.0, 140, -20
    started = time.monotonic()
    print(f"Loaded policy on {policy.device}; {total_steps} physics steps at {sim.fps} Hz", flush=True)
    with ExitStack() as stack:
        optical = "color" if policy.cloud.use_color else "depth"
        camera = CalibratedCamera(model, cfg["camera"][optical]["name"], c)
        stack.callback(camera.close)
        renderer = stack.enter_context(OffscreenRenderer(model, 640, 480))
        viewer = None
        if args.view:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, sim.d))
        for step in range(total_steps):
            tick = time.monotonic()
            if viewer and not viewer.is_running():
                break
            rgb, depth = camera.observe(sim.d)
            policy.observe({"images": {"chest": rgb, "chest_depth": depth}, "state": sim.qpos16()})
            if step % execute_steps == 0:
                chunk = np.asarray(policy.infer()["actions"])
                if not np.isfinite(chunk).all():
                    raise ValueError("Policy predicted nonfinite actions")
                predicted_chunks.append(chunk.copy())
                print(f"step {step}/{total_steps}: predicted {chunk.shape}", flush=True)
            action = chunk[step % execute_steps].copy()
            for side, off in (("left", 0), ("right", 8)):
                bounds = sim.arms[side].range
                clipped_counts[off:off + 7] += ((action[off:off + 7] < bounds[:, 0]) |
                                                (action[off:off + 7] > bounds[:, 1]))
                clipped_counts[off + 7] += not 0 <= action[off + 7] <= 0.044
            actions.append(action)
            states.append(sim.qpos16().copy())
            if step % 5 == 0:
                frames.append(Image.fromarray(renderer.rgb(sim.d, overview)))
            sim.step(action)
            if not np.isfinite(sim.d.qpos).all():
                raise ValueError("Simulation produced nonfinite joint positions")
            task_state = task.state(sim.d)
            task_states.append(task_state)
            object_positions.append(task_state.obj.copy())
            tcp_positions.append(task_state.tcp.copy())
            for geom in task.arm.finger_geoms:
                minimum_finger_gap = min(minimum_finger_gap, float(mujoco.mj_geomDistance(
                    model, sim.d, geom, task.obj_geom, 10.0, None)))
            if first_closed_time is None and sim.qpos16()[grip_index] < 0.01:
                first_closed_time = (step + 1) / sim.fps
            if viewer:
                viewer.sync()
                time.sleep(max(0, 1 / sim.fps - (time.monotonic() - tick)))
        Image.fromarray(camera.observe(sim.d)[0]).save(args.out / "chest_final.png")
        Image.fromarray(renderer.rgb(sim.d, overview)).save(args.out / "overview_final.png")
    if not frames:
        print("Viewer closed before any simulation steps were executed")
        return
    frames[0].save(args.out / "rollout.gif", save_all=True, append_images=frames[1:],
                   duration=round(5000 / sim.fps), loop=0)
    np.savez_compressed(args.out / "trajectory.npz", qpos=states, predicted_actions=actions,
                        predicted_chunks=predicted_chunks, final_qpos=sim.qpos16(),
                        object_xyz=object_positions, active_tcp_xyz=tcp_positions,
                        initial_qpos=initial_state, initial_object_xyz=initial_object)
    result = {"checkpoint": str(args.checkpoint.resolve()), "policy_type": kind, "device": policy.device,
              "steps": len(actions), "simulated_seconds": len(actions) / sim.fps,
              "wall_seconds": time.monotonic() - started, "inferences": len(predicted_chunks),
              "execute_steps": execute_steps,
              "seed": args.seed,
              "observation_source": "simulated cameras and physics qpos at every tick",
              "action_source": "policy inference only; no demonstration commands",
              "initial_qpos_source": "explicit initial pose" if args.initial_qpos is not None else "scene home",
              "initial_qpos": initial_state.tolist(),
              "initial_object_xyz": initial_object.tolist(),
              "final_object_xyz": task_states[-1].obj.tolist(),
              "ever_grasped": any(s.grasped for s in task_states),
              "ever_success": any(s.success for s in task_states),
              "final_success": task_states[-1].success,
              "max_lift_m": max(s.lift for s in task_states),
              "minimum_finger_gap_m": minimum_finger_gap,
              "first_active_gripper_under_10mm_time_s": first_closed_time,
              "collision_steps": sum(s.collision for s in task_states),
              "clipped_targets_per_joint": clipped_counts.tolist(),
              "max_joint_change": float(np.max(np.abs(np.asarray(states) - states[0])))}
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved rollout to {args.out}")


if __name__ == "__main__":
    main()
