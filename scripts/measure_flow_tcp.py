#!/usr/bin/env python3
"""Phase-specific TCP errors from saved held-out predictions and simulation traces.

These are URDF-forward-kinematics proxies, NOT independent measurements of the
real end effector. References and time alignment are separated in the report.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "simulation"), str(ROOT / "src")]

import mujoco
import numpy as np

from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE
from openarm_sim.runner import Sim
from openarm_sim.scene import load_config, load_scene


def tcp_poses(model, cfg, qpos, side, clip_commands=False):
    sim = Sim(model, cfg)
    sim.reset()
    positions, rotations = [], []
    for row in np.asarray(qpos):
        row = row.copy()
        if clip_commands:
            for name, off in (("left", 0), ("right", 8)):
                limits = sim.arms[name].range
                row[off:off + 7] = np.clip(row[off:off + 7], limits[:, 0], limits[:, 1])
                row[off + 7] = np.clip(row[off + 7], 0, .044)
        sim.set_qpos16(row)
        mujoco.mj_kinematics(model, sim.d)
        p, r = sim.arms[side].tcp_pose(sim.d)
        positions.append(p)
        rotations.append(r)
    return np.asarray(positions), np.asarray(rotations)


def stats(error_m, angles_deg, selected):
    e, angle = error_m[selected], angles_deg[selected]
    if not len(e):
        return {"samples": 0}
    distance = np.linalg.norm(e, axis=1) * 1000
    return {"samples": len(e), "tcp_position_rmse_mm": float(np.sqrt(np.mean(distance**2))),
            "tcp_position_median_mm": float(np.median(distance)), "tcp_position_p95_mm": float(np.percentile(distance, 95)),
            "tcp_position_max_mm": float(distance.max()),
            "tcp_xyz_bias_mm": (e.mean(0) * 1000).tolist(),
            "tcp_xy_rmse_mm": float(np.sqrt(np.mean(np.sum(e[:, :2]**2, axis=1))) * 1000),
            "tcp_orientation_median_deg": float(np.median(angle)),
            "tcp_orientation_p95_deg": float(np.percentile(angle, 95))}


def pose_errors(p, r, target_p, target_r):
    relative_trace = np.einsum("nij,nij->n", r, target_r)
    angle = np.degrees(np.arccos(np.clip((relative_trace - 1) / 2, -1, 1)))
    return p - target_p, angle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnosis", type=Path, default=ROOT / "outputs/flow_rgbd_v1_diagnosis")
    ap.add_argument("--closed-loop", type=Path, default=ROOT / "outputs/flow_rgbd_v1_closed_loop_aligned")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/flow_tcp_approach")
    ap.add_argument("--fine-approach-seconds", type=float, default=1.0)
    args = ap.parse_args()
    if args.fine_approach_seconds <= 0:
        ap.error("Approach window must be positive")
    source = json.loads((args.diagnosis / "report.json").read_text())
    data = np.load(args.diagnosis / "diagnostic_arrays.npz")
    model, _ = load_scene(DEFAULT_SCENE)
    cfg = load_config(DEFAULT_CONFIG)
    side, fps = source["active_arm"], source["fps"]
    grip = 15 if side == "right" else 7
    state, command = data["real_qpos"], data["demo_actions"]
    n = len(command)
    closed = np.flatnonzero(command[:, grip] < .03)
    if not len(closed):
        raise ValueError("No recorded command closure below 30 mm; manually annotate a grasp event")
    closing = int(closed[0])
    fine_start = max(0, closing - round(args.fine_approach_seconds * fps))
    phases = {"episode": [0, n], "approach_before_closing": [0, closing],
              "final_approach": [fine_start, closing], "closing_window": [closing, min(n, closing + round(.5 * fps))]}
    measured_p, measured_r = tcp_poses(model, cfg, state, side)
    commanded_p, commanded_r = tcp_poses(model, cfg, command, side, clip_commands=True)
    comparisons = []
    for index, seed in enumerate([r["seed"] for r in source["real_observations"]]):
        prediction = data["real_predictions"][index][data["query_valid"]]
        if prediction.shape != command.shape:
            raise ValueError("Expected non-overlapping chronological chunk predictions covering the whole episode")
        p, r = tcp_poses(model, cfg, prediction, side, clip_commands=True)
        comparisons.append((f"policy_real_inputs_seed_{seed}", "predicted command FK vs recorded command FK; both clipped to controller limits",
                            p, r, commanded_p, commanded_r))
    hold = np.repeat(state[data["anchors"]][:, None], data["query_targets"].shape[1], axis=1)[data["query_valid"]]
    p, r = tcp_poses(model, cfg, hold, side, clip_commands=True)
    comparisons.append(("hold_command_baseline", "hold-current-state command FK vs recorded command FK",
                        p, r, commanded_p, commanded_r))
    replay_p, replay_r = tcp_poses(model, cfg, data["replay_qpos"], side)
    comparisons.append(("demonstration_replay_tracking", "physics replay qpos FK vs measured real qpos FK at identical timestamps",
                        replay_p, replay_r, measured_p, measured_r))
    closed_loop = np.load(args.closed_loop / "trajectory.npz")
    common = min(n, len(closed_loop["qpos"]))
    p, r = tcp_poses(model, cfg, closed_loop["qpos"][:common], side)
    comparisons.append(("policy_closed_loop_time_aligned", "closed-loop physics qpos FK vs measured real qpos FK; includes progress/timing differences",
                        p, r, measured_p[:common], measured_r[:common]))
    args.out.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for name, reference, p, r, target_p, target_r in comparisons:
        error, angles = pose_errors(p, r, target_p, target_r)
        results[name] = {"reference": reference, "phases": {}}
        for phase, (start, end) in phases.items():
            end = min(end, len(error))
            results[name]["phases"][phase] = stats(error, angles, slice(start, end))
        for i, (e, angle) in enumerate(zip(error, angles, strict=True)):
            rows.append({"comparison": name, "label_frame": i, "source_frame": int(data["source_indices"][i]),
                         "time_s": i / fps, "in_final_approach": fine_start <= i < closing,
                         "dx_mm": e[0] * 1000, "dy_mm": e[1] * 1000, "dz_mm": e[2] * 1000,
                         "distance_mm": np.linalg.norm(e) * 1000, "orientation_error_deg": angle})
    with (args.out / "tcp_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"episode": source["episode"], "active_arm": side, "units": "mm and degrees",
              "measurement_status": "URDF FK proxy only; no independently observed real TCP positions available",
              "calibration_status": "not independently calibrated; existing scene and robot geometry unchanged",
              "phase_definition": "fixed from recorded command gripper <30 mm, never from model predictions",
              "closing_label_frame": closing, "closing_source_frame": int(data["source_indices"][closing]),
              "closing_label_time_s": closing / fps,
              "phase_intervals_label_seconds": {k: [a/fps, b/fps] for k, (a, b) in phases.items()},
              "comparisons": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print("Final approach window:", fine_start/fps, closing/fps, "seconds (label time)")
    lines = ["# TCP errors in final approach", "", "URDF FK proxies, not independent real TCP ground truth.", "",
             f"Episode `{source['episode']}`, {side} arm, final approach {fine_start/fps:.2f}–{closing/fps:.2f}s.", "",
             "| Comparison | TCP RMSE mm | P95 mm | Median orientation deg |", "|---|---:|---:|---:|"]
    for name, value in results.items():
        s = value["phases"]["final_approach"]
        print(name, json.dumps(s))
        lines.append(f"| {name} | {s['tcp_position_rmse_mm']:.2f} | {s['tcp_position_p95_mm']:.2f} | {s['tcp_orientation_median_deg']:.2f} |")
    lines += ["", "Policy predictions are compared to demonstration COMMAND FK, not measured state FK (commands lead measured states).",
              "Replay tracking uses measured state FK. Closed-loop time-aligned error also includes different task progress.",
              "Camera/robot geometry uncertainty is not captured by these numbers."]
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
