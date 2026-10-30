#!/usr/bin/env python3
"""How close are the simulated camera images to the real ones, for the policy?

For each episode the recorded joint positions are replayed KINEMATICALLY in the measured scene (no physics)
from the first labelled frame up to the recorded gripper close (+ margin), with the tin at its measured start
pose. At every frame both the real and the simulated chest/front images are rendered at the same state.

Reported per episode and overall:
  - image: mean RGB of the tin and of the robot (from the simulator's segmentation, same pixels in both images)
    and the median depth difference on robot pixels;
  - policy: two copies of the policy, one fed real images and one fed simulated images (same states, same
    noise), predicted gripper close time and TCP error of each against the recorded commands, and the TCP
    distance between their predictions. If the sim-fed policy behaves like the real-fed one, the simulator is
    good enough for that policy, whatever the images look like.

  sim_realism.py --checkpoint releases/meanflow_v2_run3/best_release \
      --raw "/media/space/New Volume/dataset/2026-10-13" --out outputs/sim_realism_run3
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=root / "simulation/config/scene_20261013.yaml")
    ap.add_argument("--episodes", nargs="*", help="Default: the release's validation episodes")
    ap.add_argument("--stride", type=int, default=4, help="Infer and measure every N frames (all frames are observed)")
    ap.add_argument("--after-close", type=float, default=1.0, help="Seconds replayed after the recorded close")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--channel-swap", action="store_true",
                    help="Also run policies fed sim depth + real RGB and real depth + sim RGB, to find which modality "
                         "makes the sim-fed policy differ")
    ap.add_argument("--paste-test", action="store_true",
                    help="Also run policies fed REAL images with only the sim robot (or only the sim tin) pasted in")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path[:0] = [str(root / "src"), str(root / "simulation"), str(root / "simulation/scripts")]
    import cv2
    import h5py
    import mujoco
    import numpy as np
    import torch
    from PIL import Image
    from openarm_data.episode import EpisodeReader
    from openarm_flow.kinematics import OpenArmKinematics
    from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy
    from openarm_flow.rgb_views import resize_rgb
    from openarm_sim.composite_camera import CompositeCamera
    from openarm_sim.measured_scene import compile_measured
    from openarm_sim.runner import Sim
    from openarm_sim.scene import PAD_GEOM
    from replay_real_20261013 import load_cfg, set_object

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    variants = ("real", "sim") + (("simD_realRGB", "realD_simRGB") if args.channel_swap else ()) \
        + (("real_simRobotRGB", "real_simTinRGB", "sim_realRobotRegionRGB") if args.paste_test else ())
    policies = {name: OpenArmMeanFlowPolicy(args.checkpoint, args.device, seed=args.seed) for name in variants}
    meta = policies["real"].metadata
    fps = meta["fps"]
    episodes = args.episodes or [e["id"] for e in meta["episodes"] if e["split"] == "val"]
    cfg = load_cfg(args.config, args.raw)
    scene = cfg["_measured"]["scene"]
    starts = {r["episode"]: r for r in scene["tin_poses"]["start"]}
    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    tin_geoms = {m.geom(n).id for n in ("mentos_tin_geom", "mentos_tin_label") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    half_h = cfg["objects"]["mentos_tin"]["size"][2] / 2
    pad = m.geom(PAD_GEOM).id
    box_xy0 = m.body_pos[m.geom_bodyid[pad]][:2].copy()
    pad_xy0 = np.array(scene["pad"]["center"][:2])
    background_dir = (args.config.parent / cfg["background_dir"]).resolve()
    names = cfg["_measured"]["cameras"]["cameras"]
    cameras = {c: CompositeCamera.from_config(m, c, background_dir, cfg.get("composite"),
                                              rng=np.random.default_rng(args.seed)) for c in names}
    image_keys = list(policies["real"].cameras) + (list(policies["real"].rgb_config.cameras)
                                                    if policies["real"].rgb_config else [])
    fk = OpenArmKinematics()
    args.out.mkdir(parents=True, exist_ok=True)

    def tcp(chunk, arm):
        return fk(torch.from_numpy(np.asarray(chunk, np.float32)), clamp=True)[0][:, arm].numpy()

    rows, color_acc = [], {k: [] for k in ("tin", "robot")}
    for episode in episodes:
        if episode not in starts:
            print(f"{episode}: no measured tin start pose; skipped", flush=True)
            continue
        per_episode = scene["pad"].get("per_episode", {}).get(episode)
        if per_episode:
            body = m.geom_bodyid[pad]
            m.body_pos[body][:2] = box_xy0 + (np.array(per_episode["center"]) - pad_xy0)
            m.geom_pos[pad][:2] = np.array(per_episode["center"]) - m.body_pos[body][:2]
        with EpisodeReader(args.raw / f"{episode}.hdf5") as reader:
            f = reader.file
            labels = reader.labels(meta.get("action_source_requested", "commands"), fps)
            arm = 0 if labels.actions[:, 7].min() < .03 else 1
            grip = 7 if arm == 0 else 15
            closed = labels.actions[:, grip] < .03
            events = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
            if not len(events):
                print(f"{episode}: no recorded close; skipped", flush=True)
                continue
            close = int(events[0])
            last = min(len(labels.indices) - policies["real"].chunk_size, close + round(args.after_close * fps))
            for name, camera in cameras.items():
                per_background = background_dir / "bg_episodes" / f"background_{name}_{episode}_rgb.png"
                camera.set_background(f"_{episode}", per_background.parent) if per_background.exists() else camera.set_background("")
                camera.match_color(f[f"observations/images/{name}_rgb"][0])
            for policy in policies.values():
                policy.reset()
            close_pred = {name: None for name in variants}
            errors = {**{name: [] for name in variants}, **{f"between_{name}": [] for name in variants if name != "real"}}
            colors = {k: {"real": [], "sim": []} for k in ("tin", "robot")}
            depth_diff = []
            panel = None
            for j in range(last + 1):
                i = int(labels.indices[j])
                real = {k: f[f"observations/images/{k}"][i] for k in image_keys}
                sim.reset(qpos16=labels.state[j])
                set_object(m, sim.d, "mentos_tin", starts[episode]["xy"], starts[episode]["yaw_deg"], box_top + half_h + 1e-4)
                mujoco.mj_forward(m, sim.d)
                simulated, masks, pastes = {}, {}, {}
                for name, camera in cameras.items():
                    rgb, depth = camera.observe(sim.d)
                    simulated[f"{name}_rgb"], simulated[f"{name}_depth"] = rgb, depth
                    seg = camera.last_segmentation
                    tin = np.isin(seg, list(tin_geoms))
                    pastes[name] = (tin, (seg >= 0) & ~tin)  # full (not eroded) visible masks
                    # Eroded masks: the measured poses are a few mm off, so mask edges would mix in the table.
                    erode = lambda x: cv2.erode(x.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # noqa: E731
                    masks[name] = (erode(tin), erode((seg >= 0) & ~tin))
                sources = {"real": real, "sim": simulated,
                           "simD_realRGB": {k: (simulated if k.endswith("depth") else real)[k] for k in image_keys},
                           "realD_simRGB": {k: (real if k.endswith("depth") else simulated)[k] for k in image_keys}}
                sources["sim_realRobotRegionRGB"] = dict(simulated)
                for name in names:  # sim image, but the robot and ~15 px around it from the real image
                    region = cv2.dilate(pastes[name][1].astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
                    sources["sim_realRobotRegionRGB"][f"{name}_rgb"] = np.where(region[..., None], real[f"{name}_rgb"],
                                                                                simulated[f"{name}_rgb"])
                for variant, which in (("real_simRobotRGB", 1), ("real_simTinRGB", 0)):
                    sources[variant] = dict(real)
                    for name in names:
                        mask = pastes[name][which][..., None]
                        sources[variant][f"{name}_rgb"] = np.where(mask, simulated[f"{name}_rgb"], real[f"{name}_rgb"])
                for name, policy in policies.items():
                    policy.observe({"images": {k: sources[name][k] for k in image_keys}, "state": labels.state[j]})
                if j < policies["real"].state_steps - 1 or j % args.stride:
                    continue
                chunks = {name: policy.infer()["actions"] for name, policy in policies.items()}
                target = labels.actions[j:j + policies["real"].chunk_size]
                positions = {name: tcp(c, arm) for name, c in chunks.items()}
                target_position = tcp(target, arm)
                for name in variants:
                    errors[name].extend(np.linalg.norm(positions[name] - target_position, axis=-1).tolist())
                    if close_pred[name] is None and chunks[name][:, grip].min() < .03:
                        close_pred[name] = j
                    if name != "real":
                        errors[f"between_{name}"].extend(np.linalg.norm(positions["real"] - positions[name], axis=-1).tolist())
                for name in names:
                    tin, robot = masks[name]
                    for key, mask in (("tin", tin), ("robot", robot)):
                        if mask.sum() >= 30:
                            colors[key]["real"].append(real[f"{name}_rgb"][mask].mean(0))
                            colors[key]["sim"].append(simulated[f"{name}_rgb"][mask].mean(0))
                    both = robot & (real[f"{name}_depth"] > 0) & (simulated[f"{name}_depth"] > 0)
                    if both.sum() >= 30:
                        depth_diff.append(float(np.median(simulated[f"{name}_depth"][both].astype(float)
                                                          - real[f"{name}_depth"][both])))
                if panel is None and j >= close - round(0.2 * fps):
                    cfg_rgb = policies["real"].rgb_config
                    panel = np.concatenate([np.concatenate([resize_rgb(real[f"{c}_rgb"], cfg_rgb, f"{c}_rgb").transpose(1, 2, 0),
                                                            resize_rgb(simulated[f"{c}_rgb"], cfg_rgb, f"{c}_rgb").transpose(1, 2, 0)], 1)
                                            for c in names], 0) if cfg_rgb else None
        rms = {k: (float(np.sqrt(np.mean(np.square(v))) * 1000) if v else None) for k, v in errors.items()}
        row = {"episode": episode, "arm": "left" if arm == 0 else "right", "real_close_s": close / fps,
               **{f"pred_close_{name}_s": None if close_pred[name] is None else close_pred[name] / fps for name in variants},
               "tcp_rmse_mm_real_images": rms["real"], "tcp_rmse_mm_sim_images": rms["sim"],
               "tcp_rms_mm_sim_vs_real_prediction": rms["between_sim"],
               **{f"tcp_rms_mm_{name}_vs_real_prediction": rms[f"between_{name}"] for name in variants if name not in ("real", "sim")},
               "robot_depth_median_sim_minus_real_mm": float(np.median(depth_diff)) if depth_diff else None}
        for key in ("tin", "robot"):
            if colors[key]["real"]:
                r, s = np.mean(colors[key]["real"], 0), np.mean(colors[key]["sim"], 0)
                row[f"{key}_rgb_real"], row[f"{key}_rgb_sim"] = r.round(1).tolist(), s.round(1).tolist()
                color_acc[key].append((r, s))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if panel is not None:
            Image.fromarray(panel).save(args.out / f"{episode}_before_close.png")
    for camera in cameras.values():
        camera.close()
    if not rows:
        raise SystemExit("No episode measured")

    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None

    closes = {name: [r[f"pred_close_{name}_s"] for r in rows] for name in variants}
    summary = {"checkpoint": str(args.checkpoint), "episodes": len(rows),
               "closes_predicted_real_images": sum(c is not None for c in closes["real"]),
               "closes_predicted_sim_images": sum(c is not None for c in closes["sim"]),
               **{f"closes_predicted_{name}": sum(c is not None for c in closes[name])
                  for name in variants if name not in ("real", "sim")},
               **{f"tcp_rms_mm_{name}_vs_real_prediction": mean(f"tcp_rms_mm_{name}_vs_real_prediction")
                  for name in variants if name not in ("real", "sim")},
               "tcp_rmse_mm_real_images": mean("tcp_rmse_mm_real_images"),
               "tcp_rmse_mm_sim_images": mean("tcp_rmse_mm_sim_images"),
               "tcp_rms_mm_sim_vs_real_prediction": mean("tcp_rms_mm_sim_vs_real_prediction"),
               "robot_depth_median_sim_minus_real_mm": mean("robot_depth_median_sim_minus_real_mm")}
    for key, pairs in color_acc.items():
        if pairs:
            summary[f"{key}_rgb_real"] = np.mean([p[0] for p in pairs], 0).round(1).tolist()
            summary[f"{key}_rgb_sim"] = np.mean([p[1] for p in pairs], 0).round(1).tolist()
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
