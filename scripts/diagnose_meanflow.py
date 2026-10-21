#!/usr/bin/env python3
"""Open-loop diagnosis of a MeanFlow release on REAL recorded frames of held-out episodes.

For sampled frames the policy sees the recorded camera history and qpos exactly as at deployment and
predicts one chunk. Reported per phase (all / approach = 1 s before the recorded close / grasp):
  - TCP error (URDF FK) of the active arm, policy vs a "hold current pose" baseline;
  - predicted vs recorded gripper close time (first frame whose predicted chunk closes the gripper).

  diagnose_meanflow.py --checkpoint releases/meanflow_v2_run2/best_release \
      --raw "/media/space/New Volume/dataset/2026-10-13" --out outputs/diagnose_run2.json
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from openarm_data.episode import EpisodeReader
from openarm_flow.kinematics import OpenArmKinematics
from openarm_flow.meanflow_data import grasp_phase_flags
from openarm_flow.meanflow_policy import HybridMeanFlowPolicy, OpenArmMeanFlowPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--gripper-checkpoint", type=Path, help="Take the grippers from this release (hybrid policy)")
    ap.add_argument("--episodes", nargs="*", help="Default: the release's validation episodes")
    ap.add_argument("--stride", type=int, default=10, help="Evaluate every N-th labelled frame")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    adapter = (HybridMeanFlowPolicy(args.checkpoint, args.gripper_checkpoint, args.device) if args.gripper_checkpoint
               else OpenArmMeanFlowPolicy(args.checkpoint, args.device))
    fps = adapter.metadata["fps"]
    episodes = args.episodes or [e["id"] for e in adapter.metadata["episodes"] if e["split"] == "val"]
    cameras = list(adapter.cameras) + (list(adapter.rgb_config.cameras) if adapter.rgb_config else [])
    fk = OpenArmKinematics()
    errors = {phase: {"policy": [], "hold": []} for phase in ("all", "approach", "after_close")}
    rows = []
    for episode in episodes:
        with EpisodeReader(args.raw / f"{episode}.hdf5") as reader:
            labels = reader.labels(adapter.metadata.get("action_source_requested", "commands"), fps)
            images = {c: reader.file[f"observations/images/{c}"] for c in cameras}
            arm = 0 if labels.actions[:, 7].min() < .03 else 1
            grip = 7 if arm == 0 else 15
            approach, _ = grasp_phase_flags(labels.actions, fps)
            closed = labels.actions[:, grip] < .03
            close_events = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
            real_close = int(close_events[0]) if len(close_events) else None
            predicted_close = None
            wants_close = []   # per sample: (frame, does the chunk close the gripper?)
            grip_err, early, late, flips = [], 0, 0, 0   # approach-window gripper error (mm) and close-timing failures
            previous_open = None
            # Object features: the scene is estimated once from raw frame 0 (as in conversion and deployment),
            # not from whatever frame a sample starts at (the hand may already cover the tin there).
            parts = [getattr(adapter, n) for n in ("arm", "gripper") if hasattr(adapter, n)] or [adapter]
            scenes = [p.estimator.estimate({c: reader.file[f"observations/images/{c}_rgb"][0] for c in p.object_cameras})
                      if getattr(p, "estimator", None) is not None else None for p in parts]
            for k in range(adapter.state_steps - 1, len(labels.indices) - adapter.chunk_size, args.stride):
                adapter.reset()
                for p, scene in zip(parts, scenes):
                    p.scene = scene
                for j in range(k - adapter.state_steps + 1, k + 1):
                    i = int(labels.indices[j])
                    adapter.observe({"images": {c: images[c][i] for c in cameras}, "state": labels.state[j]})
                chunk = adapter.infer()["actions"]
                target = labels.actions[k:k + adapter.chunk_size]
                hold = np.repeat(labels.state[k][None], adapter.chunk_size, 0)
                pos = {name: fk(torch.from_numpy(np.asarray(x, np.float32)), clamp=True)[0][:, arm].numpy()
                       for name, x in (("policy", chunk), ("hold", hold), ("target", target))}
                for name in ("policy", "hold"):
                    d = np.linalg.norm(pos[name] - pos["target"], axis=-1)  # [chunk]
                    errors["all"][name].extend(d.tolist())
                    if approach[k, arm]:
                        errors["approach"][name].extend(d.tolist())
                    if real_close is not None and k >= real_close:
                        errors["after_close"][name].extend(d.tolist())
                if predicted_close is None and chunk[:, grip].min() < .03:
                    predicted_close = k
                if approach[k, arm]:
                    grip_err.extend(np.abs(chunk[:, grip] - target[:, grip]).tolist())
                wants_close.append((k, bool(chunk[:, grip].min() < .03)))
                is_open = chunk[:, grip].min() >= .03
                if previous_open is not None and is_open != previous_open and k < (real_close or 0) + 1:
                    flips += 1   # open/close decision changes between consecutive samples before the real close
                previous_open = is_open
        row = {"episode": episode, "arm": "left" if arm == 0 else "right",
               "real_close_s": None if real_close is None else real_close / fps,
               "predicted_close_s": None if predicted_close is None else predicted_close / fps}
        row["close_error_s"] = (None if None in (real_close, predicted_close)
                                else (predicted_close - real_close) / fps)
        for n in (2, 3, 4):   # first frame at which n consecutive samples all want to close
            first = next((wants_close[i + n - 1][0] for i in range(len(wants_close) - n + 1)
                          if all(w for _, w in wants_close[i:i + n])), None)
            row[f"close_error_confirm{n}_s"] = None if None in (real_close, first) else (first - real_close) / fps
        row["approach_gripper_mae_mm"] = float(np.mean(grip_err) * 1000) if grip_err else None
        row["close_decision_flips_before_close"] = flips
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {}
    for phase, values in errors.items():
        if values["policy"]:
            summary[phase] = {f"{name}_tcp_rmse_mm": float(np.sqrt(np.mean(np.square(v))) * 1000)
                              for name, v in values.items()}
            summary[phase]["samples"] = len(values["policy"])
    offsets = [r["close_error_s"] for r in rows if r["close_error_s"] is not None]
    summary["approach_gripper_mae_mm"] = float(np.mean([r["approach_gripper_mae_mm"] for r in rows
                                                        if r["approach_gripper_mae_mm"] is not None]))
    for n in (2, 3, 4):
        values = [r[f"close_error_confirm{n}_s"] for r in rows if r[f"close_error_confirm{n}_s"] is not None]
        summary[f"confirm{n}"] = {"early_gt_0p3s": sum(v < -0.3 for v in values),
                                  "mean_close_error_s": float(np.mean(values)) if values else None,
                                  "missing": sum(r[f"close_error_confirm{n}_s"] is None for r in rows)}
    summary["early_gt_0p3s"] = sum(e is not None and e < -0.3 for e in offsets)
    summary["early_close_gt_0p5s"] = sum(e is not None and e < -0.5 for e in offsets)
    summary["decision_flips_mean"] = float(np.mean([r["close_decision_flips_before_close"] for r in rows]))
    summary["close_timing_s"] = {"mean": float(np.mean(offsets)) if offsets else None,
                                 "abs_mean": float(np.mean(np.abs(offsets))) if offsets else None,
                                 "missing": sum(r["close_error_s"] is None for r in rows)}
    report = {"checkpoint": str(args.checkpoint), "step": adapter.metadata.get("step"), "stride": args.stride,
              "chunk_size": adapter.chunk_size, "summary": summary, "episodes": rows}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
