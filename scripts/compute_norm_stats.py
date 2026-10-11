#!/usr/bin/env python3
"""Normalisation stats for state and actions. Run once per dataset, chunk
size and delta setting -- every fine-tuning mode on them shares the result.

    uv run scripts/compute_norm_stats.py
    uv run scripts/compute_norm_stats.py data.delta_actions=true
"""

import logging

import hydra
from omegaconf import DictConfig

from openarm_act_lerobot import paths


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    paths.setup_env(cfg)
    from openarm_act_lerobot.norm_stats import compute_norm_stats

    print(f"norm stats written to {compute_norm_stats(cfg)}")


if __name__ == "__main__":
    main()
