#!/usr/bin/env python3
"""Train from the Hub dataset in one command, every step only when needed:

    1. wandb      the key is there (before hours of downloading, not after)
    2. backbone   its pretrained weights, fetched now: DINOv3's repos are gated,
                  and a missing access request should stop the run before the
                  episodes download
    3. download   the episodes not under data/raw/<repo name>/ yet
    4. convert    unless data/lerobot/<repo_id> holds exactly these episodes,
                  tasks and settings (a stale or partial one is replaced)
    5. norm stats recomputed (seconds)
    6. train

    uv run scripts/train_pipeline.py exp_name=v1
    uv run scripts/train_pipeline.py exp_name=v1 backbone=dinov2_vitb14
    uv run scripts/train_pipeline.py exp_name=v1 backbone=resnet18           # act_pipeline's backbone
    uv run scripts/train_pipeline.py exp_name=v1 resume=true        # continue a run
    uv run scripts/train_pipeline.py exp_name=try download.limit=10 num_train_steps=2000
    uv run scripts/train_pipeline.py exp_name=v1 download.enable=false   # episodes already in data/raw
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_act_lerobot import cli
from openarm_act_lerobot import paths


def step(title: str) -> None:
    print(f"\n=== {title}", flush=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    paths.setup_env(cfg)
    from openarm_act_lerobot.convert import convert
    from openarm_act_lerobot.download import check
    from openarm_act_lerobot.download import destination
    from openarm_act_lerobot.download import download
    from openarm_act_lerobot.episode import find_episodes
    from openarm_act_lerobot.norm_stats import compute_norm_stats
    from openarm_act_lerobot.run import check_wandb
    from openarm_act_lerobot.run import fetch_backbone
    from openarm_act_lerobot.run import fetch_lingbot_depth
    from openarm_act_lerobot.run import train

    step("1/6 wandb")
    check_wandb(cfg)
    print("off" if not cfg.wandb.enable else f"{cfg.wandb.mode}, project {cfg.wandb.project}")

    step("2/6 backbone")
    where = fetch_backbone(cfg)
    print(f"{cfg.backbone.repo}: {where}" if where else f"not fetched (finetune={cfg.finetune.name}, "
          f"backbone.repo={cfg.backbone.repo})")
    lingbot = fetch_lingbot_depth(cfg)
    if lingbot:
        print(f"LingBot-Depth: {lingbot}")

    step("3/6 episodes")
    download_cfg = OmegaConf.merge({"paths": cfg.paths}, cfg.download)
    folder = destination(download_cfg)
    print(f"{len(find_episodes(folder)) if folder.is_dir() else 0} episodes in {folder}")
    if cfg.download.enable:
        result = download(download_cfg)
        print(f"{result['repo_id']}@{result['sha'][:8]}: {len(result['files'])} episodes, "
              f"{len(result['fetched'])} fetched now")
        if cfg.download.check and result["fetched"]:
            problems = check(Path(result["dir"]), result["fetched"])
            for name, found in problems.items():
                print(f"  {name}: {'; '.join(found)}")
    # Every episode under paths.raw is converted, as with convert_dataset.py;
    # the manifest excludes or holds out the ones not to train on.

    step("4/6 LeRobot dataset")
    convert_cfg = OmegaConf.merge(
        {"paths": cfg.paths, "data": cfg.data, "overwrite": False, "reuse_current": True, "dry_run": False},
        cfg.convert,
    )
    report = convert(convert_cfg)
    print(f"{report['repo_id']}: {len(report['episodes'])} episodes, {report['frames']} frames, "
          f"{len(report['skipped'])} skipped" + (" (already converted)" if report.get("reused") else ""))

    step("5/6 norm stats")
    print(compute_norm_stats(cfg))

    if cfg.prepare_only:
        print(f"\nprepared for {cfg.name} (prepare_only=true): nothing trained")
        return

    step("6/6 train")
    print(f"run finished: {train(cfg)}")


if __name__ == "__main__":
    cli.apply()  # --resnet, --dinov2, --lingbot-depth, ... (openarm_act_lerobot/cli.py)
    main()
