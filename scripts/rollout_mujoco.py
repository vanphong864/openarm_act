#!/usr/bin/env python3
"""Run an ACT or FlowPolicy OpenArm release in the imported MuJoCo workspace."""

import argparse
from collections import deque
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
    ap.add_argument("--checkpoint", type=Path, default=root / "releases/runpod_smoke")
    ap.add_argument("--policy", choices=["auto", "act", "flowpolicy"], default="auto")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--out", type=Path, default=root / "outputs/mujoco_policy")
    ap.add_argument("--execute-steps", type=int, default=25, help="Actions executed before observing again")
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
    from openarm_sim.sensor import ChestCamera

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    kind = args.policy
    if kind == "auto":
        metadata_file = args.checkpoint / "metadata.json"
        kind = json.loads(metadata_file.read_text()).get("policy_type", "act") if metadata_file.exists() else "act"
    if kind == "flowpolicy":
        from openarm_flow.policy import OpenArmFlowPolicy
        policy = OpenArmFlowPolicy(args.checkpoint, device=args.device)
    else:
        from openarm_act_lerobot.policy import OpenArmPolicy
        policy = OpenArmPolicy(args.checkpoint, device=args.device)
    if set(policy.cameras) - {"chest", "chest_depth"}:
        raise ValueError(f"Unsupported cameras: {policy.cameras}")
    cfg = load_config(DEFAULT_CONFIG)
    if int(policy.metadata["fps"]) != cfg["task"]["fps"]:
        raise ValueError("Policy and simulator control rates must match")
    model, _ = load_scene(DEFAULT_SCENE)
    if kind == "flowpolicy":
        # Match the simulated rays to the checkpoint's pinhole calibration.
        # ChestCamera resizes native 1280x720 to 640x480; only a centered camera is supported here.
        c = policy.camera
        if (c.width, c.height) != (640, 480):
            raise ValueError("Simulation adapter currently requires a 640x480 release calibration")
        # MuJoCo renders a centered pinhole image.  Real D435i intrinsics can
        # have a small principal-point offset and square focal lengths; use
        # fy for the simulator FOV and let the policy keep its recorded
        # intrinsics when projecting the rendered depth.
        if abs(c.cx - 319.5) > 4 or abs(c.cy - 239.5) > 4:
            raise ValueError("Simulation adapter requires a nearly centered principal point")
        model.cam_fovy[model.camera(cfg["camera"]["color"]["name"]).id] = np.degrees(2 * np.arctan(480 / (2*c.fy)))
    sim = Sim(model, cfg)
    sim.reset()
    policy.reset()
    history = deque([sim.qpos16().copy() for _ in range(policy.state_steps)], maxlen=policy.state_steps)
    execute_steps = min(args.execute_steps, policy.chunk_size)
    total_steps = round(args.seconds * sim.fps)
    args.out.mkdir(parents=True, exist_ok=True)
    frames, states, actions, predicted_chunks = [], [], [], []
    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0.2, 0, 0.3]
    overview.distance, overview.azimuth, overview.elevation = 2.0, 140, -20
    started = time.monotonic()
    print(f"Loaded policy on {policy.device}; {total_steps} physics steps at {sim.fps} Hz", flush=True)
    with ExitStack() as stack:
        camera = ChestCamera(model, cfg["camera"]["color"]["name"], out_size=(640, 480))
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
            if kind == "flowpolicy":
                rgb, depth = camera.observe(sim.d)
                policy.observe({"images": {"chest": rgb, "chest_depth": depth}, "state": sim.qpos16()})
            if step % execute_steps == 0:
                if kind == "flowpolicy":
                    chunk = np.asarray(policy.infer()["actions"])
                else:
                    rgb, depth = camera.observe(sim.d)
                    state = history[-1] if policy.state_steps == 1 else np.stack(history)
                    chunk = np.asarray(policy.infer({"images": {"chest": rgb, "chest_depth": depth},
                                                     "state": state})["actions"])
                if not np.isfinite(chunk).all():
                    raise ValueError("Policy predicted nonfinite actions")
                predicted_chunks.append(chunk.copy())
                print(f"step {step}/{total_steps}: predicted {chunk.shape}", flush=True)
            action = chunk[step % execute_steps].copy()
            actions.append(action)
            states.append(sim.qpos16().copy())
            if step % 5 == 0:
                frames.append(Image.fromarray(renderer.rgb(sim.d, overview)))
            sim.step(action)
            if not np.isfinite(sim.d.qpos).all():
                raise ValueError("Simulation produced nonfinite joint positions")
            history.append(sim.qpos16().copy())
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
                        predicted_chunks=predicted_chunks, final_qpos=sim.qpos16())
    result = {"checkpoint": str(args.checkpoint.resolve()), "policy_type": kind, "device": policy.device,
              "steps": len(actions), "simulated_seconds": len(actions) / sim.fps,
              "wall_seconds": time.monotonic() - started, "inferences": len(predicted_chunks),
              "execute_steps": execute_steps,
              "max_joint_change": float(np.max(np.abs(np.asarray(states) - states[0])))}
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved rollout to {args.out}")


if __name__ == "__main__":
    main()
