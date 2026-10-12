"""Training, and loading a trained checkpoint back for inference.

Training builds LeRobot's TrainPipelineConfig from the Hydra config
(train_config.py) and runs LeRobot's own loop, lerobot.scripts.lerobot_train.
train, with four things added around it:

    norm stats   this repo's (norm_stats.py) replace the dataset's per-frame
                 ones in the normaliser, so delta actions are scaled right
    delta        with data.delta_actions, processors.py's steps go into the
                 processors: arm joints relative to the state in,
                 absolute targets out
    optimizer    ACT's parameter groups, the backbone at its own learning
                 rate (finetune.lr_backbone), under the Hydra optimizer and
                 schedule; LeRobot gives them only with its policy preset
    run config   openarm_config.yaml is written into every checkpoint's
                 pretrained_model/, which serve, eval and export read

The processors are saved with each checkpoint, steps and stats included, so
inference rebuilds exactly the pipeline training used.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path
from unittest import mock

from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_act_lerobot import paths
from openarm_act_lerobot.norm_stats import load_norm_stats
from openarm_act_lerobot.norm_stats import norm_stats_path
from openarm_act_lerobot.train_config import backbone_source
from openarm_act_lerobot.train_config import build_train_config
from openarm_act_lerobot.train_config import dataset_image_size
from openarm_act_lerobot.train_config import resolve_device
from openarm_act_lerobot.train_config import save_run_config

log = logging.getLogger(__name__)

PRETRAINED_DIR = "pretrained_model"
MODEL_FILE = "model.safetensors"


def run_dir(cfg: DictConfig) -> Path:
    """checkpoints/<model>_<finetune>/<exp_name>: LeRobot's output_dir."""
    return paths.resolve(cfg.paths.checkpoints) / str(cfg.name) / str(cfg.exp_name)


# ------------------------------------------------------------- processors


def with_stats(stats: dict | None, ours: dict) -> dict:
    """The dataset's stats (cameras and all) with state and actions replaced."""
    merged = dict(stats or {})
    merged.update(ours)
    return merged


@contextlib.contextmanager
def openarm_training(cfg: DictConfig, stats: dict | None):
    """LeRobot's train() with this repo's stats, delta steps, optimizer groups and run config."""
    from lerobot.scripts import lerobot_train

    import openarm_act_lerobot.configuration_openarm_act  # noqa: F401  -- registers "openarm_act"

    from openarm_act_lerobot.processors import add_delta_steps
    from openarm_act_lerobot.processors import has_delta_steps
    from openarm_act_lerobot.processors import link_delta_steps

    make_processors = lerobot_train.make_pre_post_processors
    save_checkpoint = lerobot_train.save_checkpoint

    def make_pre_post_processors(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
        if stats is not None:  # a fresh run; a resumed one keeps its checkpoint's stats
            kwargs["dataset_stats"] = with_stats(kwargs.get("dataset_stats"), stats)
            for overrides, step in (("preprocessor_overrides", "normalizer_processor"),
                                    ("postprocessor_overrides", "unnormalizer_processor")):
                if step in kwargs.get(overrides, {}):
                    kwargs[overrides][step]["stats"] = kwargs["dataset_stats"]
        preprocessor, postprocessor = make_processors(policy_cfg, pretrained_path, pretrained_revision, **kwargs)
        if has_delta_steps(preprocessor):  # a resumed run's
            link_delta_steps(preprocessor, postprocessor)
        elif cfg.data.delta_actions:
            add_delta_steps(preprocessor, postprocessor)
        return preprocessor, postprocessor

    def make_optimizer_and_scheduler(train_cfg, policy):
        optimizer = train_cfg.optimizer.build(policy.get_optim_params())
        scheduler = train_cfg.scheduler.build(optimizer, train_cfg.steps) if train_cfg.scheduler else None
        return optimizer, scheduler

    def save_with_run_config(checkpoint_dir, *args, **kwargs):
        save_checkpoint(checkpoint_dir, *args, **kwargs)
        save_run_config(cfg, Path(checkpoint_dir) / PRETRAINED_DIR)

    with (
        mock.patch.object(lerobot_train, "make_pre_post_processors", make_pre_post_processors),
        mock.patch.object(lerobot_train, "make_optimizer_and_scheduler", make_optimizer_and_scheduler),
        mock.patch.object(lerobot_train, "save_checkpoint", save_with_run_config),
    ):
        yield lerobot_train.train


def check_wandb(cfg: DictConfig) -> None:
    """Fail before the dataset and the model load, not after, when wandb
    would ask for a key it cannot get."""
    if not cfg.wandb.enable or cfg.wandb.mode != "online" or os.environ.get("WANDB_API_KEY"):
        return
    netrc = Path(os.environ.get("NETRC", Path.home() / ".netrc"))
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():  # `wandb login`
        return
    raise RuntimeError(
        f"wandb is on but has no API key: paste it into {paths.DOTENV} (WANDB_API_KEY=...), "
        "run `uv run wandb login`, or train with wandb.enable=false"
    )


def fetch_lingbot_depth(cfg: DictConfig) -> str | None:
    """LingBot-Depth's model.pt (1.3 GB, not gated) into the Hugging Face
    cache, when the run uses it as depth encoder or teacher; -> where it is."""
    uses = [group for group in (cfg.depth_encoder, cfg.distill) if group.name == "lingbot_depth"]
    if cfg.depth_encoder.name == "lingbot_depth" and str(cfg.data.get("depth_as")) != "branch":
        uses = [group for group in uses if group is not cfg.depth_encoder]
    if not uses or not cfg.finetune.load_backbone:
        return None
    repo = str(uses[0].repo)
    if Path(paths.resolve(repo)).exists() and repo.startswith(("/", ".", "~")):
        return str(paths.resolve(repo))
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, "model.pt", revision=uses[0].get("revision"))


def fetch_backbone(cfg: DictConfig) -> str | None:
    """The pretrained backbone into its cache, before the dataset and the
    model load; -> where it is. The DINOv3 repos are gated: a refusal says
    what to do about it. None when the run uses no pretrained backbone
    (finetune=dummy) or a local directory."""
    repo, revision = backbone_source(cfg)
    if not cfg.finetune.load_backbone or Path(repo).is_dir():
        return None
    if cfg.backbone.family == "resnet":  # torchvision's, into ~/.cache/torch
        import torch
        import torchvision

        weights = torchvision.models.get_weight(repo)
        weights.get_state_dict(progress=True)
        return str(Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name)

    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    except GatedRepoError as error:
        raise RuntimeError(
            f"{repo} is gated and this Hugging Face account has no access yet: request it on "
            f"https://huggingface.co/{repo} (Meta approves by hand), then put a token of that account in "
            f"{paths.DOTENV} (HF_TOKEN=hf_...). Or use backbone=dinov2_vitb14 (not gated) meanwhile, "
            "or point backbone.repo at a local copy."
        ) from error
    except HfHubHTTPError as error:
        raise RuntimeError(f"cannot fetch {repo}@{revision}: {error}") from error


def train(cfg: DictConfig) -> Path:
    """Train; -> the run directory (checkpoints/<name>/<exp_name>)."""
    check_wandb(cfg)
    stats_file = norm_stats_path(cfg)
    if not stats_file.exists():
        raise FileNotFoundError(
            f"no norm stats at {stats_file}; run scripts/compute_norm_stats.py with the same data/model overrides first"
        )
    out = run_dir(cfg)
    last = out / "checkpoints" / "last"
    if out.exists():  # resume wins over overwrite: never wipe a run asked to continue
        if cfg.resume:
            if not (last / PRETRAINED_DIR).exists():
                raise FileNotFoundError(f"{out} has no checkpoint to resume from; pass overwrite=true")
        elif cfg.overwrite:
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"{out} exists; pass overwrite=true or resume=true")
    resuming = bool(cfg.resume) and (last / PRETRAINED_DIR).exists()

    # Serving may happen where the dataset is not: keep its frame size.
    cfg.data.image_size = dataset_image_size(cfg)
    if cfg.data.image_size is None:
        raise FileNotFoundError(f"no dataset {cfg.data.repo_id}; run scripts/convert_dataset.py first")

    if resuming:
        from lerobot.configs import PreTrainedConfig

        import openarm_act_lerobot.configuration_openarm_act  # noqa: F401  -- an openarm_act checkpoint

        checkpoint = last.resolve()
        policy = PreTrainedConfig.from_pretrained(checkpoint / PRETRAINED_DIR)
        policy.pretrained_path = checkpoint / PRETRAINED_DIR
        policy.device = resolve_device(str(cfg.device))
        train_cfg = build_train_config(cfg, out, policy=policy)
        train_cfg.resume = True
        train_cfg.checkpoint_path = checkpoint
        log.info("resuming from %s", checkpoint)
    else:
        fetch_backbone(cfg)
        fetch_lingbot_depth(cfg)
        train_cfg = build_train_config(cfg, out)
    # The size the backbone sees, resolved from the frames: kept in openarm_config.yaml too.
    cfg.model.image_size = list(train_cfg.policy.image_size)
    # Every path is resolved here; LeRobot's own resolution reads sys.argv.
    train_cfg._resolve_pretrained_from_cli = lambda: None

    stats = None if resuming else load_norm_stats(stats_file)
    log.info("run directory: %s", out)
    with openarm_training(cfg, stats) as lerobot_train:
        lerobot_train(train_cfg)
    return out


# -------------------------------------------------------------- inference


def model_dir(checkpoint: str | Path) -> Path:
    """The directory with model.safetensors and openarm_config.yaml, from a
    release, a pretrained_model/ directory, a step directory, or a run
    directory (its newest step)."""
    path = paths.resolve(checkpoint)
    candidates = [path, path / PRETRAINED_DIR]
    for steps in (path, path / "checkpoints"):  # a run directory, or its checkpoints/
        numbered = sorted((p for p in steps.glob("[0-9]*") if p.name.isdigit()), key=lambda p: int(p.name))
        candidates += [steps / "last" / PRETRAINED_DIR] + [p / PRETRAINED_DIR for p in reversed(numbered)]
    for candidate in candidates:
        if (candidate / MODEL_FILE).exists():
            if not (candidate / paths.RUN_CONFIG).exists():
                raise FileNotFoundError(f"no {paths.RUN_CONFIG} in {candidate}; was it trained by scripts/train.py?")
            return candidate.resolve()
    raise FileNotFoundError(f"{path} is not a checkpoint, a run directory or a release (no {MODEL_FILE})")


def load_run_config(checkpoint: str | Path) -> tuple[DictConfig, Path]:
    directory = model_dir(checkpoint)
    return OmegaConf.load(directory / paths.RUN_CONFIG), directory


def checkpoint_step(directory: Path) -> int | None:
    """The training step a pretrained_model/ came from, if it says."""
    name = directory.parent.name if directory.name == PRETRAINED_DIR else None
    return int(name) if name and name.isdigit() else None
