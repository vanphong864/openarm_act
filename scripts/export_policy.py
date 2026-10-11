#!/usr/bin/env python3
"""A trained checkpoint -> a release: weights, processors, config, metadata.

    uv run scripts/export_policy.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/v1/checkpoints/040000
    uv run scripts/export_policy.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/v1 repo_id=<user>/openarm-act
    uv run scripts/serve.py checkpoint=releases/act_dinov3_vitb16_frozen_v1_040000
"""

import logging

import hydra
from omegaconf import DictConfig

from openarm_act_lerobot import paths
from openarm_act_lerobot.export import export


@hydra.main(version_base="1.3", config_path="../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    paths.load_dotenv()
    logging.getLogger().setLevel(logging.INFO)
    print(f"release written to {export(cfg)}")


if __name__ == "__main__":
    main()
