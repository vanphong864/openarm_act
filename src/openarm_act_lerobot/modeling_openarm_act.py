"""ACT with the backbone and depth handling configuration_openarm_act names.

Changes to LeRobot's ACTPolicy:

    backbone        ResNet (ACT's own, with the trainable rules of
                    backbones.py), DINOv2 or DINOv3; with depth_as=branch
                    wrapped in backbones.RGBDFusion, a depth encoder beside
                    it. Its parameters are `model.backbone.*`.
    first layer     with depth_as=channel, ResNet's conv1 / the ViT's patch
                    embedding takes four channels (act_pipeline's conv1_adapter).
    prepare_images  each camera resized to config.image_size and normalised
                    with the backbone's mean/std; the depth camera fused into
                    the RGB one per depth_as.
    optimizer       the pretrained backbone at optimizer_lr_backbone; what
                    trains from scratch (ACT, the depth CNN, projections, the
                    distillation head) at optimizer_lr.
    state history   with n_state_steps > 1, the last n joint states [B, n, 16]
                    flattened into ACT's one state token (and the CVAE's):
                    its two state projections take 16 n inputs.
    distillation    with distill=lingbot_depth, in training only: the
                    transformer encoder's image tokens, through a linear
                    head, predict LingBot-Depth's features of the same
                    patches; loss + distill_weight * (1 - cosine). The teacher
                    is not part of the model: not saved, never run at inference.
"""

from __future__ import annotations

import contextvars

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE
from openarm_act_lerobot.backbones import LingBotDepthEncoder
from openarm_act_lerobot.backbones import ResNetBackbone
from openarm_act_lerobot.backbones import RGBDFusion
from openarm_act_lerobot.backbones import ViTBackbone
from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig

# Set while a checkpoint is being loaded: the backbone is built empty, the
# checkpoint's weights fill it, nothing is fetched (Hub or torchvision).
_LOADING_CHECKPOINT = contextvars.ContextVar("openarm_act_loading_checkpoint", default=False)
# Batch keys prepare_images adds for the distillation teacher.
TEACHER_RGB = "openarm.teacher_rgb"
TEACHER_DEPTH = "openarm.teacher_depth"


class OpenArmACTPolicy(ACTPolicy):
    config_class = OpenArmACTConfig
    name = "openarm_act"

    def __init__(self, config: OpenArmACTConfig, **kwargs):
        fresh = config.load_backbone_weights and not _LOADING_CHECKPOINT.get()
        # ACT builds its ResNet here, with torchvision's weights when it is given
        # some: only a fresh ResNet run gets them.
        weights = config.pretrained_backbone_weights
        if config.backbone_type != "resnet" or not fresh:
            config.pretrained_backbone_weights = None
        try:
            super().__init__(config, **kwargs)
        finally:
            config.pretrained_backbone_weights = weights
        if config.n_state_steps > 1 and config.robot_state_feature:
            width = config.robot_state_feature.shape[0] * config.n_state_steps
            self.model.encoder_robot_state_input_proj = nn.Linear(width, config.dim_model)
            if config.use_vae:
                self.model.vae_encoder_robot_state_input_proj = nn.Linear(width, config.dim_model)
        if not config.image_features:
            return
        if any(int(v) % config.backbone_stride for v in config.image_size):
            raise ValueError(f"image_size {list(config.image_size)} is not a multiple of the "
                             f"{config.backbone_type} stride {config.backbone_stride}")
        if config.backbone_type == "resnet":
            backbone = ResNetBackbone(self.model.backbone, config)
        else:
            backbone = ViTBackbone(config, load_weights=fresh)
            self.model.encoder_img_feat_input_proj = nn.Conv2d(backbone.hidden_size, config.dim_model, kernel_size=1)
        if config.depth_as == "branch":
            backbone = RGBDFusion(backbone, config, load_weights=fresh)
        self.model.backbone = backbone
        self.register_buffer("image_mean", torch.tensor(config.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(config.image_std).view(1, 3, 1, 1), persistent=False)

        if config.distill != "none":
            dim = int(config.lingbot_depth_config["dim_out"])
            self.distill_head = nn.Sequential(nn.LayerNorm(config.dim_model), nn.Linear(config.dim_model, dim))
            self._encoder_out = None
            self.model.encoder.register_forward_hook(self._keep_encoder_out)

    # ------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """A checkpoint holds the whole network: the backbone is built empty and
        the checkpoint fills it, with nothing downloaded."""
        token = _LOADING_CHECKPOINT.set(True)
        try:
            return super().from_pretrained(*args, **kwargs)
        finally:
            _LOADING_CHECKPOINT.reset(token)

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """Every tensor or none: a layer left at its random weights would drive
        the arm with a network nobody trained."""
        missing, unexpected = model.load_state_dict(load_file(model_file, device="cpu"), strict=False)
        if missing or unexpected:
            raise ValueError(f"{model_file} does not fit this network: {len(missing)} missing keys {missing[:5]}, "
                             f"{len(unexpected)} unexpected {unexpected[:5]}")
        return model

    # ----------------------------------------------------------- optimizer

    def get_optim_params(self) -> list[dict]:
        """Two groups: the pretrained backbone's trainable weights at
        optimizer_lr_backbone, everything else at optimizer_lr. A depth
        encoder inside model.backbone trains from scratch: the second kind."""
        backbone, rest = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            pretrained = name.startswith("model.backbone.") and not name.startswith("model.backbone.depth.")
            (backbone if pretrained else rest).append(param)
        groups = [{"params": rest}]
        if backbone:
            groups.append({"params": backbone, "lr": self.config.optimizer_lr_backbone})
        return groups

    # -------------------------------------------------------------- images

    def depth_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The depth camera as (metres or 0..1 per depth_format, valid mask), [B, 1, H, W]."""
        config = self.config
        depth = batch[config.depth_key][:, :1].float()
        if config.depth_format == "raw":
            valid = (depth > config.depth_valid_min_m).to(depth.dtype)
            return depth * valid, valid
        return depth, torch.ones_like(depth)  # gray8: no way to tell a missing reading

    def prepare_images(self, batch: dict) -> dict:
        """Each camera ACT encodes, resized to image_size and fused per depth_as;
        with distillation, the teacher's RGB and depth too."""
        config = self.config
        batch = dict(batch)
        has_depth = config.depth_key in batch
        if (config.depth_as in ("channel", "branch") or config.distill != "none") and not has_depth:
            raise KeyError(f"no {config.depth_key} in the batch: this model reads RGB-D")
        size = tuple(config.image_size)

        def resize(x, mode="bilinear"):
            if tuple(x.shape[-2:]) == size:
                return x
            # Bilinear on half-pixel centres, as act_pipeline (and cv2) resize;
            # raw depth and its mask nearest, so no reading is averaged with a real one.
            return F.interpolate(x, size=size, mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))

        depth = valid = None
        if has_depth and config.depth_as != "image":
            mode = "nearest" if config.depth_format == "raw" else "bilinear"
            depth, valid = (resize(v, mode) for v in self.depth_inputs(batch))
        if config.distill != "none" and self.training:  # the teacher: RGB 0..1, depth in m (0 = no reading)
            batch[TEACHER_RGB] = resize(batch[config.rgb_key][:, :3])
            batch[TEACHER_DEPTH] = depth
        for key in config.image_features:
            image = resize(batch[key][:, :3])
            if key == config.rgb_key and config.depth_as == "branch":
                batch[key] = torch.cat([image, depth.to(image.dtype), valid.to(image.dtype)], dim=1)  # normalised in RGBDFusion
                continue
            image = (image - self.image_mean.to(image.dtype)) / self.image_std.to(image.dtype)
            if key == config.rgb_key and config.depth_as == "channel":
                if config.depth_format == "raw":  # act_pipeline's 0..1 over depth_min..max_m, 0 without a reading
                    span = config.depth_max_m - config.depth_min_m
                    depth = (depth.clamp(config.depth_min_m, config.depth_max_m) - config.depth_min_m) / span * valid
                image = torch.cat([image, depth.to(image.dtype)], dim=1)
            batch[key] = image
        return batch

    # ------------------------------------------------------------ teacher

    def _keep_encoder_out(self, module, args, output):
        self._encoder_out = output

    @property
    def teacher(self) -> LingBotDepthEncoder:
        """LingBot-Depth, built on first use and kept outside the module tree:
        not saved with the model, not moved by .to(), not in the optimizer."""
        teacher = self.__dict__.get("_teacher")
        if teacher is None:
            # Its weights from the Hub (or lingbot_depth_repo) whenever it is
            # built: a checkpoint does not hold them. Off only for random test models.
            teacher = LingBotDepthEncoder(self.config, load_weights=self.config.load_backbone_weights)
            self.__dict__["_teacher"] = teacher
        return teacher

    def distill_loss(self, batch: dict) -> torch.Tensor:
        """1 - cosine between the head's prediction from each image token of
        the transformer encoder and the teacher's feature of that patch."""
        grid = tuple(int(v) // self.config.backbone_stride for v in self.config.image_size)
        start = self.model.encoder_1d_feature_pos_embed.num_embeddings  # latent, state: then the first camera
        tokens = self._encoder_out[start : start + grid[0] * grid[1]].transpose(0, 1)  # [B, h*w, dim_model]
        prediction = self.distill_head(tokens)
        teacher = self.teacher.to(prediction.device)
        target = teacher(batch[TEACHER_RGB], batch[TEACHER_DEPTH], grid).flatten(2).transpose(1, 2)
        return (1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()

    # ------------------------------------------------------------- forward

    def prepare_state(self, batch: dict) -> dict:
        """The state history [B, n, 16] -> [B, 16 n], oldest first."""
        n = self.config.n_state_steps
        state = batch.get(OBS_STATE)
        if n == 1 or state is None:
            return batch
        if state.ndim != 3 or state.shape[1] != n:
            raise ValueError(f"this model reads the last {n} joint states, [B, {n}, D]; got {tuple(state.shape)}")
        return {**batch, OBS_STATE: state.flatten(1)}

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        batch = self.prepare_images(self.prepare_state(batch))
        loss, loss_dict = super().forward(batch)
        if self.config.distill != "none" and self.training:
            distill = self.distill_loss(batch)
            loss = loss + self.config.distill_weight * distill
            loss_dict["distill_loss"] = distill.item()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.eval()  # before prepare_images: no teacher inputs at inference
        return super().predict_action_chunk(self.prepare_images(self.prepare_state(batch)))
