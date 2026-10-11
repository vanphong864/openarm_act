#!/usr/bin/env python3
"""openarm_mc HDF5 episodes -> a LeRobot dataset under paths.lerobot_home.

    uv run scripts/convert_dataset.py                    # data=openarm_rgbd
    uv run scripts/convert_dataset.py data=openarm_rgbd data.fps=25 overwrite=true
    uv run scripts/convert_dataset.py dry_run=true       # what would go in
"""

import hydra
from omegaconf import DictConfig

from openarm_act_lerobot import paths


@hydra.main(version_base="1.3", config_path="../conf", config_name="convert")
def main(cfg: DictConfig) -> None:
    paths.setup_env(cfg)
    from openarm_act_lerobot.convert import convert

    report = convert(cfg)
    if report.get("dry_run"):
        return
    print(f"\n{report['repo_id']}: {len(report['episodes'])} episodes, {report['frames']} frames "
          f"at {report['fps']} Hz -> {report['root']}; {len(report['skipped'])} skipped")


if __name__ == "__main__":
    main()
