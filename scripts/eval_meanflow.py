#!/usr/bin/env python3
"""Evaluate a MeanFlow release on its held-out episodes with masked action metrics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, Subset
from openarm_flow.meanflow_data import MeanFlowDataset
from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy
from train_meanflow import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, action="append", required=True, help="Repeat for several zarrs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples-per-episode", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    adapter = OpenArmMeanFlowPolicy(args.checkpoint, args.device)
    data = MeanFlowDataset(args.dataset, adapter.policy.horizon, adapter.state_steps, "val")
    expected = {k: v for k, v in adapter.metadata.items() if k not in ("policy_type", "step")}
    if data.metadata != expected:
        raise ValueError("Dataset metadata differs from the trained release")
    loader = DataLoader(Subset(data, data.validation_indices(args.samples_per_episode)), batch_size=args.batch_size)
    metrics = validate(adapter.policy, loader, adapter.device, args.seed)
    report = {"checkpoint": str(args.checkpoint), "step": adapter.metadata["step"],
              "seed": args.seed, "validation_episodes": len(data.ranges), **metrics}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
