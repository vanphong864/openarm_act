"""Hydra config -> this repo's OpenArmACTConfig and LeRobot's TrainPipelineConfig.

The Hydra tree (conf/) holds every choice; this is the one place that turns
it into LeRobot's dataclasses, for training, serving and evaluation alike.
LeRobot's command line (draccus) is not used.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def depth_fields(cfg: DictConfig) -> dict:
    """How chest_depth reaches the backbone: data.depth_as=channel (ACT's RGB-D
    input) needs the cameras [chest, chest_depth]."""
    depth_as = str(cfg.data.get("depth_as") or "image")
    cameras = [str(c) for c in cfg.data.cameras]
    if depth_as == "channel" and cameras != [C.CAMERA_RGB, C.CAMERA_DEPTH]:
        raise ValueError(f"data.depth_as=channel needs cameras [chest, chest_depth], got {cameras}")
    return {
        "depth_as": depth_as,
        "rgb_key": C.LEROBOT_IMAGE_PREFIX + C.CAMERA_RGB,
        "depth_key": C.LEROBOT_IMAGE_PREFIX + C.CAMERA_DEPTH,
        "depth_init": str(cfg.data.get("depth_init") or "zero"),
    }


def policy_overrides(cfg: DictConfig) -> dict:
    """The OpenArmACTConfig fields this repo sets, from model + backbone + finetune + data."""
    model, backbone, finetune = cfg.model, cfg.backbone, cfg.finetune
    if finetune.backbone_trainable == "last" and int(finetune.backbone_last_layers) < 1:
        raise ValueError("finetune.backbone_trainable=last needs backbone_last_layers >= 1")
    return {
        "chunk_size": int(model.chunk_size),
        "n_action_steps": int(model.n_action_steps),
        "dim_model": int(model.dim_model),
        "n_heads": int(model.n_heads),
        "dim_feedforward": int(model.dim_feedforward),
        "feedforward_activation": str(model.feedforward_activation),
        "pre_norm": bool(model.pre_norm),
        "n_encoder_layers": int(model.n_encoder_layers),
        "n_decoder_layers": int(model.n_decoder_layers),
        "use_vae": bool(model.use_vae),
        "latent_dim": int(model.latent_dim),
        "n_vae_encoder_layers": int(model.n_vae_encoder_layers),
        "dropout": float(model.dropout),
        "kl_weight": float(model.kl_weight),
        "use_amp": bool(model.use_amp),
        # ACT's optimizer groups; the run's optimizer and schedule come from Hydra.
        "optimizer_lr": float(cfg.lr.peak_lr),
        "optimizer_weight_decay": float(cfg.optimizer.weight_decay),
        "optimizer_lr_backbone": float(finetune.lr_backbone),
        **backbone_fields(cfg),
        "load_backbone_weights": bool(finetune.load_backbone),
        "backbone_trainable": str(finetune.backbone_trainable),
        "backbone_last_layers": int(finetune.backbone_last_layers),
        "train_patch_embedding": bool(finetune.train_patch_embedding),
        "gradient_checkpointing": bool(finetune.gradient_checkpointing),
        "image_mean": tuple(float(v) for v in backbone.image_mean),
        "image_std": tuple(float(v) for v in backbone.image_std),
        **depth_fields(cfg),
        "device": resolve_device(str(cfg.device)),
        "push_to_hub": False,
        "repo_id": None,
        # Filled from the dataset by make_policy: chest (+ chest_depth), 16 joints.
        "input_features": {},
        "output_features": {},
    }


def backbone_source(cfg: DictConfig) -> tuple[str, str | None]:
    """(repo, revision) of the pretrained backbone: for the ViTs a Hub repo at
    backbone.revision or a local directory (absolute, or relative to the
    repository); for ResNet the torchvision weights' name."""
    repo = str(cfg.backbone.repo)
    if cfg.backbone.family == "resnet":
        return repo, None
    if repo.startswith(("/", ".", "~")) or paths.resolve(repo).is_dir():
        return str(paths.resolve(repo)), None
    return repo, cfg.backbone.get("revision")


def backbone_fields(cfg: DictConfig) -> dict:
    """The OpenArmACTConfig fields that pick the backbone (conf/backbone/)."""
    family = str(cfg.backbone.family)
    repo, revision = backbone_source(cfg)
    if family == "resnet":
        return {
            "backbone_type": family,
            "vision_backbone": str(cfg.backbone.arch),
            "pretrained_backbone_weights": repo,
            "replace_final_stride_with_dilation": bool(cfg.backbone.get("dilation", False)),
            "backbone_repo": None,
            "backbone_revision": None,
        }
    return {"backbone_type": family, "backbone_repo": repo, "backbone_revision": revision}


def model_image_size(cfg: DictConfig, patch_size: int) -> tuple[int, int]:
    """[height, width] the backbone sees: model.image_size, else the dataset's
    frame scaled to about model.image_tokens image tokens of patch_size
    pixels (the backbone's stride), aspect ratio kept. For 640x480 frames
    and 300 tokens: DINOv3 (16) 320x240, DINOv2 (14) 280x210, ResNet (32)
    640x480 as act_pipeline. For 424x240: DINOv3 368x208 (23x13)."""
    if cfg.model.get("image_size"):
        size = tuple(int(v) for v in cfg.model.image_size)
        if any(v % patch_size for v in size):
            raise ValueError(f"model.image_size {list(size)} must be multiples of the patch size {patch_size}")
        return size
    frame = cfg.data.get("image_size") or dataset_image_size(cfg)
    if not frame:
        raise FileNotFoundError(f"no dataset {cfg.data.repo_id} to take the frame size from; convert it first, "
                                "or set model.image_size")
    height, width = (int(v) for v in frame)
    scale = math.sqrt(int(cfg.model.image_tokens) * patch_size**2 / (height * width))
    return tuple(max(1, round(v * scale / patch_size)) * patch_size for v in (height, width))


def backbone_config_for(cfg: DictConfig) -> dict | None:
    """A ViT's architecture: the pretrained repo's config.json, or (finetune
    without load_backbone) the config class's defaults with
    finetune.backbone_config on top. None for ResNet (torchvision's)."""
    from openarm_act_lerobot.backbones import vit_config

    family = str(cfg.backbone.family)
    if family == "resnet":
        return None
    overrides = OmegaConf.to_container(cfg.finetune.get("backbone_config") or {}, resolve=True)
    if not cfg.finetune.load_backbone:
        return vit_config(family, overrides=overrides)
    return vit_config(family, *backbone_source(cfg), overrides)


def build_policy_config(cfg: DictConfig):
    """-> OpenArmACTConfig for the composed config."""
    from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig

    config = OpenArmACTConfig(**policy_overrides(cfg))
    config.backbone_config = backbone_config_for(cfg)
    config.image_size = model_image_size(cfg, config.backbone_stride)
    return config


def build_train_config(cfg: DictConfig, output_dir: Path, policy=None):
    """-> lerobot.configs.train.TrainPipelineConfig for the composed config;
    `policy`: a checkpoint's config to resume with instead of a new one."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.default import WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim import AdamWConfig
    from lerobot.optim import CosineDecayWithWarmupSchedulerConfig
    from lerobot.transforms import ImageTransformsConfig

    from openarm_act_lerobot.video import video_backend

    if cfg.lr.warmup_steps >= cfg.lr.decay_steps:
        raise ValueError(f"lr.warmup_steps ({cfg.lr.warmup_steps}) must be below lr.decay_steps ({cfg.lr.decay_steps})")
    if cfg.model.n_action_steps > cfg.model.chunk_size:
        raise ValueError(f"model.n_action_steps ({cfg.model.n_action_steps}) exceeds model.chunk_size")

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=str(cfg.data.repo_id),
            image_transforms=ImageTransformsConfig(enable=bool(cfg.data.image_transforms)),
            video_backend=video_backend(),
            # The model normalises images itself (the backbone's mean/std; depth as 0..1).
            use_imagenet_stats=False,
        ),
        policy=policy if policy is not None else build_policy_config(cfg),
        output_dir=Path(output_dir),
        job_name=f"{cfg.name}_{cfg.exp_name}",  # the wandb run's name
        seed=int(cfg.seed),
        num_workers=int(cfg.num_workers),
        batch_size=int(cfg.batch_size),
        steps=int(cfg.num_train_steps),
        log_freq=int(cfg.log_interval),
        save_freq=int(cfg.save_interval),
        env_eval_freq=0,
        use_policy_training_preset=False,
        optimizer=AdamWConfig(
            lr=float(cfg.lr.peak_lr),
            betas=tuple(float(b) for b in cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
            grad_clip_norm=float(cfg.optimizer.grad_clip_norm),
        ),
        scheduler=CosineDecayWithWarmupSchedulerConfig(
            peak_lr=float(cfg.lr.peak_lr),
            decay_lr=float(cfg.lr.decay_lr),
            num_warmup_steps=int(cfg.lr.warmup_steps),
            num_decay_steps=int(cfg.lr.decay_steps),
        ),
        wandb=WandBConfig(
            enable=bool(cfg.wandb.enable),
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            mode=str(cfg.wandb.mode),
            notes=cfg.wandb.notes,
            disable_artifact=not bool(cfg.wandb.upload_checkpoints),
        ),
    )


def policy_metadata(cfg: DictConfig) -> dict:
    """What the server tells a client when it connects (openarm_act_lerobot_client
    reads it; the keys are the ones openarm_smolvla's server sends, too)."""
    return {
        "robot": "openarm_v1_bimanual",
        "joint_names": list(C.JOINT_NAMES),
        "action_dim": C.NUM_JOINTS,
        "action_horizon": int(cfg.model.chunk_size),
        "fps": int(cfg.data.fps),
        "model": str(cfg.model.name),
        "backbone": str(cfg.backbone.name),
        "finetune": str(cfg.finetune.name),
        "cameras": [str(c) for c in cfg.data.cameras],
        "delta_actions": bool(cfg.data.delta_actions),
        "depth_as": str(cfg.data.get("depth_as") or "image"),  # channel: one RGB-D image, as act_pipeline
        # openarm_act refuses a camera that gives other frames than these.
        "image_size": dataset_image_size(cfg),
    }


def _dataset_meta_dir(cfg: DictConfig) -> Path:
    home = os.environ.get("HF_LEROBOT_HOME") or str(paths.resolve(cfg.paths.lerobot_home))
    return Path(home) / str(cfg.data.repo_id) / "meta"


def dataset_image_size(cfg: DictConfig) -> list[int] | None:
    """[height, width] of the dataset's frames: data.image_size, else what the
    converter wrote into the LeRobot dataset's meta/info.json; None if neither."""
    if cfg.data.get("image_size"):
        return [int(v) for v in cfg.data.image_size]
    info = _dataset_meta_dir(cfg) / "info.json"
    if not info.exists():
        return None
    camera = str(cfg.data.cameras[0])
    shape = json.loads(info.read_text())["features"][C.LEROBOT_IMAGE_PREFIX + camera]["shape"]
    return [int(shape[0]), int(shape[1])]


def save_run_config(cfg: DictConfig, directory: Path) -> Path:
    path = Path(directory) / paths.RUN_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path
