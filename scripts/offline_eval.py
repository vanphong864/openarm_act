#!/usr/bin/env python3
"""A checkpoint on the held-out episodes (split: val in the manifest),
against the hold-still baseline.

    uv run scripts/offline_eval.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/first/checkpoints/040000
"""

import json
import logging

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    from openarm_act_lerobot.evaluate import evaluate

    result = evaluate(cfg)
    summary = {k: result[k] for k in ("checkpoint", "split", "episodes")}
    for name in ("model", "hold"):
        summary[name] = {k: result[name][k] for k in ("chunks", "arm_rmse_rad", "gripper_mae_mm", "arm_rmse_by_step")}
    print(json.dumps(summary, indent=2))
    print(f"full report: {result['output']}")


if __name__ == "__main__":
    main()
