"""A trained checkpoint -> a self-contained release for inference.

Training checkpoints (checkpoints/<run>/<exp>/checkpoints/<step>/) carry the
optimizer state, for resuming. A release keeps what serving needs, and
serves without the Hub (the backbone's weights are in model.safetensors,
a ViT's architecture in config.json; a gated repo is not needed there):

    releases/<name>/
      model.safetensors     the weights (float32, or bfloat16 with dtype=bfloat16)
      config.json           OpenArmACTConfig, the backbone's architecture included
      policy_preprocessor.json / policy_postprocessor.json + their
      *.safetensors         the processors, with the run's norm stats
      openarm_config.yaml   the run's resolved config
      metadata.json         model, backbone, dataset, image size, source checkpoint

serve.py and offline_eval.py take a release directory like a checkpoint.
With `repo_id`, the release is also uploaded to a Hugging Face model repo.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from openarm_act_lerobot import paths
from openarm_act_lerobot.run import MODEL_FILE
from openarm_act_lerobot.run import PRETRAINED_DIR
from openarm_act_lerobot.run import checkpoint_step
from openarm_act_lerobot.run import load_run_config
from openarm_act_lerobot.train_config import policy_metadata

log = logging.getLogger(__name__)
# What a pretrained_model/ holds besides the weights that inference reads.
KEEP = ("config.json", "policy_preprocessor*", "policy_postprocessor*", paths.RUN_CONFIG)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_weights(source: Path, target: Path, dtype: str) -> None:
    if dtype == "float32":
        shutil.copy2(source, target)
        return
    if dtype != "bfloat16":
        raise ValueError(f"dtype must be float32 or bfloat16, not {dtype!r}")
    import torch
    from safetensors.torch import load_file
    from safetensors.torch import save_file

    tensors = load_file(source)
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in tensors.items()}, target)


def export(cfg: DictConfig) -> Path:
    run_cfg, source = load_run_config(cfg.checkpoint)
    step = checkpoint_step(source)
    label = source.parent.name if source.name == PRETRAINED_DIR else source.name  # 020000
    name = cfg.name or f"{run_cfg.name}_{run_cfg.exp_name}_{label}"
    out = paths.resolve(cfg.paths.releases) / name
    if out.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{out} exists; pass overwrite=true to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for pattern in KEEP:
        for file in source.glob(pattern):
            shutil.copy2(file, out / file.name)
    write_weights(source / MODEL_FILE, out / MODEL_FILE, str(cfg.dtype))

    metadata = {
        **policy_metadata(run_cfg),
        "release": name,
        "exp_name": str(run_cfg.exp_name),
        "step": step,
        "source_checkpoint": str(source),
        "backbone_weights": (str(run_cfg.backbone.repo) + (f"@{run_cfg.backbone.revision}"
                                                           if run_cfg.backbone.get("revision") else "")
                             if run_cfg.finetune.load_backbone else None),
        "model_image_size": json.loads((source / "config.json").read_text())["image_size"],
        "dataset": str(run_cfg.data.repo_id),
        "action_source": str(run_cfg.data.action_source),
        "weights_dtype": str(cfg.dtype),
        "exported_at": datetime.datetime.now().astimezone().isoformat(),
        "openarm_act_commit": _git_commit(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("release %s: %.2f GB", out, size / 1e9)

    if cfg.repo_id:
        upload(out, str(cfg.repo_id), private=bool(cfg.private))
    return out


def upload(directory: Path, repo_id: str, private: bool = True) -> None:
    """The release into <repo_id>/<release name>/ of a Hugging Face model repo."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(directory),
        path_in_repo=directory.name,
        commit_message=f"release {directory.name}",
    )
    log.info("uploaded to https://huggingface.co/%s/tree/main/%s", repo_id, directory.name)
