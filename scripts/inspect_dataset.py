#!/usr/bin/env python3
"""Report on the raw episodes: data per task, skips, whether joint_commands
can be the labels. Run it first, on the robot PC or wherever the HDF5s are.

    uv run scripts/inspect_dataset.py paths.raw=/path/to/openarm_mc/dataset
"""

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from openarm_act_lerobot.inspect_data import inspect


@hydra.main(version_base="1.3", config_path="../conf", config_name="inspect")
def main(cfg: DictConfig) -> None:
    report = inspect(cfg)
    out = Path(HydraConfig.get().runtime.output_dir) / "inspect.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{report['raw']}: {report['episodes']} episodes, {report['usable']} usable, "
          f"{report['hours']:.2f} h, {report['held_out']} held out")
    for task, count in report["by_task"].items():
        print(f"  {count:4d}  {task}")
    for name, reason in report["skipped"].items():
        print(f"  skip {name}: {reason}")
    print(f"labels: {report['action_sources']}; median command lag "
          f"{report['median_command_lag_steps']} steps")
    print(f"full report: {out}")


if __name__ == "__main__":
    main()
