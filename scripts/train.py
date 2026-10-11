#!/usr/bin/env python3
"""Train ACT (DINOv3, DINOv2 or ResNet backbone) on an OpenArm dataset, with LeRobot's training loop.

    uv run scripts/train.py exp_name=first                      # backbone=dinov3_vitb16 finetune=frozen
    uv run scripts/train.py backbone=resnet18 exp_name=first
    uv run scripts/train.py backbone=dinov2_vits14 finetune=partial exp_name=first
    uv run scripts/train.py exp_name=first resume=true          # continue
    uv run scripts/train.py experiment=smoke                    # CPU pipeline check

Checkpoints: checkpoints/<model>_<backbone>_<finetune>/<exp_name>/checkpoints/<step>/,
each with the resolved config (openarm_config.yaml) in pretrained_model/.
"""

import hydra
from omegaconf import DictConfig

from openarm_act_lerobot import paths


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    paths.setup_env(cfg)
    from openarm_act_lerobot.run import train

    print(f"run finished: {train(cfg)}")


if __name__ == "__main__":
    main()
