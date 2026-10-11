#!/usr/bin/env python3
"""Fetch recorded episodes from Hugging Face into data/raw/<repo name>/.

    uv run scripts/download_dataset.py                 # Tuyen062004/data_openarm_8_10
    uv run scripts/download_dataset.py limit=2         # the first two episodes only
    uv run scripts/download_dataset.py repo_id=<user>/<repo> revision=<sha>

Already-downloaded files are skipped. A private repo needs `huggingface-cli
login` (or HF_TOKEN) first.
"""

import hydra
from omegaconf import DictConfig

from openarm_act_lerobot import paths

from openarm_act_lerobot.download import check
from openarm_act_lerobot.download import download


@hydra.main(version_base="1.3", config_path="../conf", config_name="download")
def main(cfg: DictConfig) -> None:
    paths.load_dotenv()
    result = download(cfg)
    print(f"\n{result['repo_id']}@{result['sha'][:8]}: {len(result['files'])} files, "
          f"{result['bytes'] / 1e9:.2f} GB in {result['dir']}")
    if cfg.check:
        from pathlib import Path

        problems = check(Path(result["dir"]), result["files"])
        for name, found in problems.items():
            print(f"  {name}: {'; '.join(found)}")
        print(f"checked {len(result['files'])} episodes: "
              + (f"{len(problems)} with problems" if problems else "all readable and complete"))


if __name__ == "__main__":
    main()
