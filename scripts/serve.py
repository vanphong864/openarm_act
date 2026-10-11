#!/usr/bin/env python3
"""Serve a checkpoint over websocket for openarm_act (RemoteActPolicy).

    uv run scripts/serve.py checkpoint=releases/act_dinov3_vitb16_frozen_first_040000
    uv run scripts/serve.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/first     # its newest step
"""

import logging

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="serve")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    from openarm_act_lerobot.policy import OpenArmPolicy
    from openarm_act_lerobot.server import PolicyServer

    policy = OpenArmPolicy(cfg.checkpoint, device=str(cfg.device))
    logging.info("serving %s (%s) on port %d", policy.directory, policy.run_cfg.name, cfg.port)
    PolicyServer(policy, port=int(cfg.port), metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
