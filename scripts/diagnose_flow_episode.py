#!/usr/bin/env python3
"""Real held-out observations, matched-pose camera ablation and demonstration replay.

Does not modify weights, simulator limits or HDF5. All camera variants use the
release's unchanged point-cloud preprocessing. Replay uses the default scene's
object placement, which is explicitly NOT registered to the real episode.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "simulation")]
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from openarm_data.episode import EpisodeReader
from openarm_flow.pointcloud import project_depth
from openarm_flow.policy import OpenArmFlowPolicy
from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE
from openarm_sim.render import OffscreenRenderer
from openarm_sim.runner import Sim
from openarm_sim.scene import load_config, load_scene
from openarm_sim.sensor import CalibratedCamera, ChestCamera
from openarm_sim.task import PickPlaceTask

ARMS = [i for i in range(16) if i not in (7, 15)]


def errors(prediction, target, state, mask):
    result = {}
    for name, action in (("policy", prediction), ("hold", np.broadcast_to(state[:, None], target.shape))):
        e = (action - target)[mask]
        result[name + "_arm_rmse_rad"] = float(np.sqrt(np.mean(e[:, ARMS] ** 2)))
        result[name + "_right_arm_rmse_rad"] = float(np.sqrt(np.mean(e[:, 8:15] ** 2)))
        result[name + "_gripper_mae_mm"] = float(np.mean(np.abs(e[:, [7, 15]])) * 1000)
        result[name + "_right_gripper_mae_mm"] = float(np.mean(np.abs(e[:, 15])) * 1000)
    result["valid_action_rows"] = int(mask.sum())
    return result


def predict(policy, clouds, histories, seed):
    # Explicit noise seed; no global RNG effects on camera or point sampling.
    torch.manual_seed(seed)
    parts = []
    with torch.inference_mode():
        for begin in range(0, len(clouds), 8):
            batch = {"point_cloud": torch.from_numpy(clouds[begin:begin + 8]).to(policy.device),
                     "agent_pos": torch.from_numpy(histories[begin:begin + 8]).to(policy.device)}
            parts.append(policy.policy.predict_action(batch)["action"].cpu().numpy())
    return np.concatenate(parts)


def depth_image(depth):
    value = (np.clip((depth.astype(float) - 200) / 1000, 0, 1) * 255).astype("u1")
    value[depth == 0] = 0
    return Image.fromarray(value).convert("RGB")


def object_alignment_sensitivity(model, cfg, real_qpos, commands, first_close_xyz, out, active_arm):
    """One prescribed post-hoc XY placement, not a search over successful poses."""
    sim = Sim(model, cfg)
    task = PickPlaceTask(model, cfg)
    sim.reset(qpos16=real_qpos[0])
    initial = task.state(sim.d).obj.copy()
    # Change ONLY XY. Keep original Z, quaternion, limits, servo and commands.
    sim.d.qpos[task.obj_qadr:task.obj_qadr + 2] = np.asarray(first_close_xyz)[:2]
    mujoco.mj_forward(model, sim.d)
    placed = task.state(sim.d).obj.copy()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.2, 0, .3]
    camera.distance, camera.azimuth, camera.elevation = 2, 140, -20
    states, frames, measured = [], [], []
    with OffscreenRenderer(model, 640, 480) as renderer:
        for i, command in enumerate(commands):
            measured.append(sim.qpos16().copy())
            if i % 10 == 0:
                frames.append(Image.fromarray(renderer.rgb(sim.d, camera)))
            sim.step(command)
            states.append(task.state(sim.d))
        frames.append(Image.fromarray(renderer.rgb(sim.d, camera)))
    frames[0].save(out / "demonstration_replay_grasp_aligned.gif", save_all=True,
                   append_images=frames[1:], duration=200, loop=0)
    frames[-1].save(out / "demonstration_replay_grasp_aligned_final.png")
    error = np.asarray(measured) - real_qpos
    active = slice(8, 15) if active_arm == "right" else slice(0, 7)
    grip_col = 15 if active_arm == "right" else 7
    report = {"method": "Sensitivity test: set ONLY initial object XY to recorded TCP XY at first measured gripper <30mm",
              "limitation": "Post-hoc alignment from recorded robot pose, not independent registration or a policy success test",
              "object_initial_xyz": placed.tolist(), "object_final_xyz": states[-1].obj.tolist(),
              "object_shift_from_default_xy_m": (placed[:2] - initial[:2]).tolist(),
              "object_z_change_m": float(placed[2] - initial[2]),
              "ever_grasped": any(s.grasped for s in states), "ever_success": any(s.success for s in states),
              "final_success": states[-1].success, "max_lift_m": max(s.lift for s in states),
              "collision_steps": sum(s.collision for s in states),
              "active_arm_tracking_rmse_rad": float(np.sqrt(np.mean(error[:, active]**2))),
              "active_gripper_tracking_mae_mm": float(np.mean(np.abs(error[:, grip_col])) * 1000)}
    (out / "object_alignment_sensitivity.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "releases/flow_rgbd_v1")
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/data_openarm_8_10")
    ap.add_argument("--episode", default="episode_20261008_153152")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/flow_rgbd_v1_diagnosis")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--replay-alignment-only", action="store_true",
                    help="Reuse saved diagnostic arrays for the post-hoc object-XY sensitivity test")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.replay_alignment_only:
        previous = json.loads((args.out / "report.json").read_text())
        data = np.load(args.out / "diagnostic_arrays.npz")
        model, _ = load_scene(DEFAULT_SCENE)
        result = object_alignment_sensitivity(
            model, load_config(DEFAULT_CONFIG), data["real_qpos"], data["demo_actions"],
            previous["demonstration_replay"]["recorded_tcp_at_first_grip_under_30mm_xyz"],
            args.out, previous["active_arm"])
        print(json.dumps(result, indent=2), flush=True)
        return
    policy = OpenArmFlowPolicy(args.checkpoint, device=args.device)
    entry = next((e for e in policy.metadata["episodes"] if e["id"] == args.episode), None)
    if entry is None or entry["split"] != "val":
        raise ValueError("Requested episode is not a held-out episode of this release")
    cfg = load_config(DEFAULT_CONFIG)
    if policy.metadata["fps"] != cfg["task"]["fps"]:
        raise ValueError("Control frequency mismatch")
    model, _ = load_scene(DEFAULT_SCENE)
    sim = Sim(model, cfg)
    sim.reset()
    task = PickPlaceTask(model, cfg)
    fps = sim.fps
    path = args.raw / (args.episode + ".hdf5")
    with EpisodeReader(path) as reader, ExitStack() as stack:
        labels = reader.labels(policy.metadata["action_source_requested"], fps)
        if labels.source != entry["action_source"] or len(labels.state) != entry["count"]:
            raise ValueError("Episode labels differ from converted training metadata")
        n, steps, obs_steps = len(labels.state), policy.chunk_size, policy.state_steps
        anchors = np.arange(0, n, steps)
        obs_ids = np.maximum(0, anchors[:, None] + np.arange(-obs_steps + 1, 1))
        target_ids = anchors[:, None] + np.arange(steps)
        mask = target_ids < n
        targets = labels.actions[np.minimum(target_ids, n - 1)]
        histories = labels.state[obs_ids]
        current = labels.state[anchors]
        needed = np.unique(obs_ids)
        cloud_map = {}
        # Per-frame RNG gives identical deterministic preprocessing to each input
        # variant; random valid-point sets need not equal the original conversion.
        for j, i in enumerate(needed):
            source = int(labels.indices[i])
            cloud_map[int(i)] = project_depth(reader.depth(source), policy.camera, policy.cloud,
                                              np.random.default_rng(policy.metadata["seed"] + int(i)),
                                              reader.rgb(source) if policy.cloud.use_color else None)
            if j % 50 == 0:
                print(f"Real frames: {j}/{len(needed)}", flush=True)
        real_clouds = np.stack([np.stack([cloud_map[int(i)] for i in ids]) for ids in obs_ids])
        seeds = [42, 43, 44]
        real_predictions, real_metrics = [], []
        for seed in seeds:
            pred = predict(policy, real_clouds, histories, seed)
            real_predictions.append(pred)
            real_metrics.append({"seed": seed, **errors(pred, targets, current, mask)})
        print("Real observation metrics:", json.dumps(real_metrics), flush=True)

        active_arm = "right" if np.ptp(labels.state[:, 8:15], axis=0).sum() >= np.ptp(labels.state[:, :7], axis=0).sum() else "left"
        grip_col = 15 if active_arm == "right" else 7
        close_ids = np.flatnonzero(labels.state[:, grip_col] < .03)
        first_close = int(close_ids[0]) if len(close_ids) else int(labels.state[:, grip_col].argmin())
        snapshots = sorted(set([0, first_close, int(labels.state[:, grip_col].argmin()), n // 2, n - 1]))
        compare_indices = np.unique(np.linspace(0, len(anchors) - 1, 23).astype(int))
        match_obs_ids = obs_ids[compare_indices]
        camera_name = cfg["camera"]["color" if policy.cloud.use_color else "depth"]["name"]
        camera_id = model.camera(camera_name).id
        original_fovy = float(model.cam_fovy[camera_id])
        calibrated = CalibratedCamera(model, camera_name, policy.camera)
        calibrated_fovy = float(model.cam_fovy[camera_id])
        stack.callback(calibrated.close)
        legacy = ChestCamera(model, cfg["camera"]["color"]["name"], out_size=(640, 480), depth_model=False)
        stack.callback(legacy.close)
        camera_maps = {"calibrated_release_rays": {}, "legacy_color_resize": {}}
        comparison = Image.new("RGB", (6 * 320, len(snapshots) * 260))
        sample_frames = set(np.unique(match_obs_ids).tolist()) | set(snapshots)
        for j, i in enumerate(sorted(sample_frames)):
            sim.reset(qpos16=labels.state[i])
            images = []
            for name, camera, fovy in (("calibrated_release_rays", calibrated, calibrated_fovy),
                                       ("legacy_color_resize", legacy, original_fovy)):
                model.cam_fovy[camera_id] = fovy
                rgb, depth = camera.observe(sim.d)
                camera_maps[name][i] = project_depth(depth, policy.camera, policy.cloud,
                                                    np.random.default_rng(policy.metadata["seed"] + i),
                                                    rgb if policy.cloud.use_color else None)
                images.append((rgb, depth))
            if i in snapshots:
                source = int(labels.indices[i])
                views = [Image.fromarray(reader.rgb(source)), Image.fromarray(images[0][0]), Image.fromarray(images[1][0]),
                         depth_image(reader.depth(source)), depth_image(images[0][1]), depth_image(images[1][1])]
                row = snapshots.index(i)
                names = ["Real RGB", "Sim release rays", "Sim legacy color", "Real depth", "Sim release depth", "Sim legacy depth"]
                for col, (image, title) in enumerate(zip(views, names, strict=True)):
                    comparison.paste(image.resize((320, 240)), (col * 320, row * 260 + 20))
                    ImageDraw.Draw(comparison).text((col * 320 + 3, row * 260 + 3), f"{title}; t={i/fps:.2f}s", fill="white")
            if j % 15 == 0:
                print(f"Matched-pose camera frames: {j}/{len(sample_frames)}", flush=True)
        comparison.save(args.out / "real_vs_sim_cameras.png")
        model.cam_fovy[camera_id] = original_fovy
        paired_metrics = {}
        paired_preds = {}
        variants = {"real_observations": real_clouds[compare_indices]}
        variants.update({name: np.stack([np.stack([clouds[int(i)] for i in ids]) for ids in match_obs_ids])
                         for name, clouds in camera_maps.items()})
        for name, clouds in variants.items():
            pred = predict(policy, clouds, histories[compare_indices], 42)
            paired_preds[name] = pred
            paired_metrics[name] = errors(pred, targets[compare_indices], current[compare_indices], mask[compare_indices])
            # Compare immediate gripper/j2 commands, not padded future targets.
            paired_metrics[name]["predicted_current_gripper_mean_mm"] = float(pred[:, 0, grip_col].mean() * 1000)
            if name != "real_observations":
                paired_metrics[name]["action_change_vs_real_arm_rmse_rad"] = float(np.sqrt(np.mean(
                    (pred - paired_preds["real_observations"])[mask[compare_indices]][:, ARMS] ** 2)))
        print("Paired camera metrics:", json.dumps(paired_metrics), flush=True)

        # Replay the actual command labels at native 50 Hz, starting from the real
        # measured initial qpos. No policy inference, joint-limit edits or IK.
        sim.reset(qpos16=labels.state[0])
        initial_object = task.state(sim.d).obj.copy()
        overview = mujoco.MjvCamera()
        overview.lookat[:] = [.2, 0, .3]
        overview.distance, overview.azimuth, overview.elevation = 2, 140, -20
        renderer = stack.enter_context(OffscreenRenderer(model, 640, 480))
        frames, measured, task_states, clipping, replay_tcp, recorded_tcp = [], [], [], [], [], []
        kinematic = Sim(model, cfg)
        kinematic.reset()
        for i, command in enumerate(labels.actions):
            measured.append(sim.qpos16().copy())
            limits = np.zeros(16, dtype=bool)
            for side, off in (("left", 0), ("right", 8)):
                bounds = sim.arms[side].range
                limits[off:off + 7] = ((command[off:off + 7] < bounds[:, 0]) | (command[off:off + 7] > bounds[:, 1]))
                limits[off + 7] = not 0 <= command[off + 7] <= .044
            clipping.append(limits)
            replay_tcp.append(sim.arms[active_arm].tcp_pose(sim.d)[0])
            kinematic.reset(qpos16=labels.state[i])
            recorded_tcp.append(kinematic.arms[active_arm].tcp_pose(kinematic.d)[0])
            if i % 10 == 0:
                frames.append(Image.fromarray(renderer.rgb(sim.d, overview)))
            sim.step(command)
            task_states.append(task.state(sim.d))
        frames.append(Image.fromarray(renderer.rgb(sim.d, overview)))
        frames[0].save(args.out / "demonstration_replay.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
        frames[-1].save(args.out / "demonstration_replay_final.png")
        measured, replay_tcp, recorded_tcp = np.asarray(measured), np.asarray(replay_tcp), np.asarray(recorded_tcp)
        error = measured - labels.state
        window = slice(max(0, first_close - 50), min(n, first_close + 100))
        active = slice(8, 15) if active_arm == "right" else slice(0, 7)
        replay_report = {"source": labels.source, "initial_state": "recorded first labelled qpos",
                         "object_registration": "default simulated object pose, NOT fitted to real episode",
                         "steps": n, "seconds": n / fps,
                         "arm_tracking_rmse_rad": float(np.sqrt(np.mean(error[:, ARMS] ** 2))),
                         "active_arm_tracking_rmse_rad": float(np.sqrt(np.mean(error[:, active] ** 2))),
                         "gripper_tracking_mae_mm": float(np.mean(np.abs(error[:, [7, 15]])) * 1000),
                         "active_gripper_tracking_mae_mm": float(np.mean(np.abs(error[:, grip_col])) * 1000),
                         "active_tcp_tracking_rmse_m": float(np.sqrt(np.mean(np.sum((replay_tcp-recorded_tcp)**2, axis=1)))),
                         "grasp_window_active_tcp_tracking_rmse_m": float(np.sqrt(np.mean(np.sum((replay_tcp[window]-recorded_tcp[window])**2, axis=1)))),
                         "clipped_targets_per_joint": np.sum(clipping, axis=0).tolist(),
                         "ever_grasped": any(s.grasped for s in task_states), "ever_success": any(s.success for s in task_states),
                         "final_success": task_states[-1].success, "max_lift_m": max(s.lift for s in task_states),
                         "collision_steps": sum(s.collision for s in task_states),
                         "object_initial_xyz": initial_object.tolist(), "object_final_xyz": task_states[-1].obj.tolist(),
                         "recorded_tcp_at_first_grip_under_30mm_xyz": recorded_tcp[first_close].tolist(),
                         "replay_tcp_at_first_grip_under_30mm_xyz": replay_tcp[first_close].tolist()}

        # Per-query immediate action traces expose premature closure on real inputs.
        np.savez_compressed(args.out / "diagnostic_arrays.npz", anchors=anchors, source_indices=labels.indices,
                            query_targets=targets, query_valid=mask, real_predictions=np.stack(real_predictions),
                            real_qpos=labels.state, demo_actions=labels.actions, replay_qpos=measured,
                            replay_tcp=replay_tcp, recorded_tcp=recorded_tcp,
                            paired_query_indices=compare_indices, **{"paired_" + k: v for k, v in paired_preds.items()})
        report = {"episode": args.episode, "split": "val", "checkpoint_step": policy.metadata["step"],
                  "source_file": str(path.resolve()), "source_frames": len(reader), "labelled_frames": n,
                  "labels": {"source": labels.source, "notes": labels.notes}, "fps": fps,
                  "active_arm": active_arm, "query_stride": steps, "query_count": len(anchors),
                  "real_first_gripper_below_30mm_label_time_s": first_close / fps,
                  "real_observations": real_metrics, "paired_camera_queries": len(compare_indices),
                  "paired_camera_metrics": paired_metrics,
                  "camera_variants": {"release_render_fovy_deg": calibrated_fovy, "original_color_fovy_deg": original_fovy,
                                       "note": "Both variants use unchanged release intrinsics to encode point clouds; legacy is an ablation, not verified calibration"},
                  "demonstration_replay": replay_report}
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        sensitivity = object_alignment_sensitivity(model, cfg, labels.state, labels.actions,
                                                   recorded_tcp[first_close], args.out, active_arm)
        print("Object XY sensitivity:", json.dumps(sensitivity), flush=True)


if __name__ == "__main__":
    main()
