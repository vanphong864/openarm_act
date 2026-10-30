#!/usr/bin/env python3
"""Real vs simulated camera images at the same recorded states, without a policy.

Each episode is replayed kinematically (recorded joint positions, tin at its measured start pose) and every
--every seconds both images are compared per camera:
  bg_mae          mean |real - sim| (0-255) on background pixels (away from the simulated robot/tin);
  bg_bad          fraction of those pixels off by more than 40: wrong or smeared background;
  robot/tin rgb   mean colour on the (eroded) simulated robot / tin pixels, real and sim;
  sharpness       mean gradient magnitude around the robot, sim / real (1 = equally sharp);
  noise           high-pass std on flat background, real and sim (the median backgrounds are noise-free);
  shadow          table luminance around the robot, real / sim (< 1: the real robot casts shadows).
A panel (real | sim | |difference| x3, chest and front) is saved per episode at 0.2 s before the recorded close.

  compare_sim_images.py --raw "/media/space/New Volume/dataset/2026-10-13" --out outputs/sim_images
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAL = ("episode_20261013_110412 episode_20261013_112001 episode_20261013_112722 episode_20261013_113152 "
               "episode_20261013_115821 episode_20261013_120946 episode_20261013_121121 episode_20261013_121427 "
               "episode_20261013_121839 episode_20261013_135217 episode_20261013_135509 episode_20261013_135719 "
               "episode_20261013_135842").split()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "simulation/config/scene_20261013.yaml")
    ap.add_argument("--episodes", nargs="*", default=DEFAULT_VAL)
    ap.add_argument("--every", type=float, default=0.5, help="Seconds between compared frames")
    ap.add_argument("--after-close", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--appearance", default="{}",
                    help='JSON overriding the config\'s composite appearance, e.g. \'{"blur_sigma": 1.0}\'')
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "simulation"), str(ROOT / "simulation/scripts")]
    import cv2
    import mujoco
    import numpy as np
    from openarm_data.episode import EpisodeReader
    from openarm_sim.composite_camera import CompositeCamera
    from openarm_sim.measured_scene import compile_measured
    from openarm_sim.runner import Sim
    from replay_real_20261013 import load_cfg, set_object

    cfg = load_cfg(args.config, args.raw)
    scene = cfg["_measured"]["scene"]
    starts = {r["episode"]: r for r in scene["tin_poses"]["start"]}
    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    tin_geoms = [m.geom(n).id for n in ("mentos_tin_geom", "mentos_tin_label")
                 if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
    box_top = cfg["_measured"]["cameras"]["box_top_height_m"]
    half_h = cfg["objects"]["mentos_tin"]["size"][2] / 2
    background_dir = (args.config.parent / cfg["background_dir"]).resolve()
    names = cfg["_measured"]["cameras"]["cameras"]
    composite = {**cfg.get("composite", {}), **json.loads(args.appearance)}
    cameras = {c: CompositeCamera.from_config(m, c, background_dir, composite, rng=np.random.default_rng(args.seed))
               for c in names}
    args.out.mkdir(parents=True, exist_ok=True)
    kernel = lambda n: np.ones((n, n), np.uint8)  # noqa: E731

    def gradient(gray):
        return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    def highpass_std(gray, mask):
        hp = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
        return float(hp[mask].std()) if mask.sum() > 500 else None

    acc = {c: {} for c in names}
    rows = []
    for episode in args.episodes:
        if episode not in starts:
            print(f"{episode}: no measured tin start pose; skipped", flush=True)
            continue
        with EpisodeReader(args.raw / f"{episode}.hdf5") as reader:
            f = reader.file
            labels = reader.labels("commands", 50)
            fps = 50
            grip = 7 if labels.actions[:, 7].min() < .03 else 15
            closed = labels.actions[:, grip] < .03
            events = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
            close = int(events[0]) if len(events) else len(labels.indices) - 1
            last = min(len(labels.indices) - 1, close + round(args.after_close * fps))
            first = {}
            for name, camera in cameras.items():
                per_background = background_dir / "bg_episodes" / f"background_{name}_{episode}_rgb.png"
                camera.set_background(f"_{episode}", per_background.parent) if per_background.exists() \
                    else camera.set_background("")
                first[name] = f[f"observations/images/{name}_rgb"][int(labels.indices[0])]
                camera.match_color(first[name])

            def pose(j):
                sim.reset(qpos16=labels.state[j])
                set_object(m, sim.d, "mentos_tin", starts[episode]["xy"], starts[episode]["yaw_deg"],
                           box_top + half_h + 1e-4)
                mujoco.mj_forward(m, sim.d)

            panel_frame = max(0, close - round(0.2 * fps))
            frames = sorted(set(range(0, last + 1, max(1, round(args.every * fps)))) | {panel_frame})
            stats = {c: {} for c in names}
            panel = []
            for j in frames:
                pose(j)
                i = int(labels.indices[j])
                for name, camera in cameras.items():
                    real = f[f"observations/images/{name}_rgb"][i]
                    simulated, _ = camera.observe(sim.d)
                    seg = camera.last_segmentation
                    dyn = seg >= 0
                    tin = np.isin(seg, tin_geoms)
                    robot = dyn & ~tin
                    # the measured robot pose is a few mm off: compare colours inside, background well outside
                    background = ~(cv2.dilate(dyn.astype(np.uint8), kernel(41)) > 0)
                    diff = np.abs(real.astype(np.int16) - simulated.astype(np.int16)).max(-1)
                    s = stats[name]
                    if background.sum() > 1000:
                        s.setdefault("bg_mae", []).append(float(diff[background].mean()))
                        s.setdefault("bg_bad", []).append(float((diff[background] > 40).mean()))
                    for key, mask in (("robot", robot), ("tin", tin)):
                        inner = cv2.erode(mask.astype(np.uint8), kernel(5)) > 0
                        if inner.sum() >= 30:
                            s.setdefault(f"{key}_rgb_real", []).append(real[inner].mean(0))
                            s.setdefault(f"{key}_rgb_sim", []).append(simulated[inner].mean(0))
                            s.setdefault(f"{key}_std_real", []).append(float(real[inner].std()))
                            s.setdefault(f"{key}_std_sim", []).append(float(simulated[inner].std()))
                    gray_r = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    gray_s = cv2.cvtColor(simulated, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    # the whole neighbourhood of the robot, so both (a few mm apart) outlines are inside
                    around = cv2.dilate(robot.astype(np.uint8), kernel(31)) > 0
                    if robot.sum() > 2000:
                        g_r, g_s = gradient(gray_r)[around].mean(), gradient(gray_s)[around].mean()
                        s.setdefault("sharpness", []).append(float(g_s / max(g_r, 1e-3)))
                    flat = background & (gradient(gray_r) < 20) & (gradient(cv2.GaussianBlur(gray_s, (0, 0), 1.5)) < 20)
                    n_r, n_s = highpass_std(gray_r, flat), highpass_std(gray_s, flat)
                    if n_r is not None:
                        s.setdefault("noise_real", []).append(n_r)
                        s.setdefault("noise_sim", []).append(n_s)
                    ring = (cv2.dilate(robot.astype(np.uint8), kernel(61)) > 0) & ~(cv2.dilate(dyn.astype(np.uint8), kernel(15)) > 0)
                    if ring.sum() > 500:
                        s.setdefault("shadow", []).append(float(gray_r[ring].mean() / max(gray_s[ring].mean(), 1.0)))
                    if j == panel_frame:
                        amp = np.clip(np.abs(real.astype(np.int16) - simulated.astype(np.int16)) * 3, 0, 255).astype(np.uint8)
                        panel.append(np.concatenate([real, simulated, amp], 1))
            cv2.imwrite(str(args.out / f"{episode}.jpg"), cv2.resize(np.concatenate(panel, 0), None, fx=0.5, fy=0.5)[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        row = {"episode": episode}
        for name in names:
            for key, values in stats[name].items():
                value = np.mean(values, 0)
                row[f"{name}_{key}"] = value.round(3).tolist() if np.ndim(value) else round(float(value), 3)
                acc[name].setdefault(key, []).append(value)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {name: {key: (np.mean(v, 0).round(3).tolist() if np.ndim(v[0]) else round(float(np.mean(v)), 3))
                      for key, v in acc[name].items()} for name in names}
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
