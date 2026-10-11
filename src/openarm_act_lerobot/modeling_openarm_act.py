"""ACT with the backbone configuration_openarm_act names, reading RGB-D as act_pipeline.

Three changes to LeRobot's ACTPolicy:

    backbone        ResNet (ACT's own, with the trainable rules of
                    backbones.py), DINOv2 or DINOv3. Its parameters are
                    `model.backbone.*`, so ACT's optimizer groups give them
                    optimizer_lr_backbone.
    first layer     with depth_as=channel, ResNet's conv1 / the ViT's patch
                    embedding takes four channels (act_pipeline's conv1_adapter).
    prepare_images  each camera resized to config.image_size and normalised
                    with the backbone's mean/std; with depth_as=channel the
                    depth camera's first channel (0..1, as act_pipeline scales
                    depth) is stacked onto the RGB camera first.
"""

from __future__ import annotations

import contextvars

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy
from openarm_act_lerobot.backbones import ResNetBackbone
from openarm_act_lerobot.backbones import ViTBackbone
from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig

# Set while a checkpoint is being loaded: the backbone is built empty, the
# checkpoint's weights fill it, nothing is fetched (Hub or torchvision).
_LOADING_CHECKPOINT = contextvars.ContextVar("openarm_act_loading_checkpoint", default=False)


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
        if not config.image_features:
            return
        if any(int(v) % config.backbone_stride for v in config.image_size):
            raise ValueError(f"image_size {list(config.image_size)} is not a multiple of the "
                             f"{config.backbone_type} stride {config.backbone_stride}")
        if config.backbone_type == "resnet":
            self.model.backbone = ResNetBackbone(self.model.backbone, config)
        else:
            backbone = ViTBackbone(config, load_weights=fresh)
            self.model.backbone = backbone
            self.model.encoder_img_feat_input_proj = nn.Conv2d(backbone.hidden_size, config.dim_model, kernel_size=1)
        self.register_buffer("image_mean", torch.tensor(config.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(config.image_std).view(1, 3, 1, 1), persistent=False)

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

    def get_optim_params(self) -> list[dict]:
        """ACT's two groups (the backbone at optimizer_lr_backbone), without an
        empty one when the backbone is frozen."""
        return [group for group in super().get_optim_params() if group["params"]]

    def prepare_images(self, batch: dict) -> dict:
        """Each camera ACT encodes: resized to image_size, normalised as the
        backbone expects; the depth camera fused in as channel 4 with depth_as=channel."""
        config = self.config
        batch = dict(batch)
        if config.depth_as_channel and config.depth_key not in batch:
            raise KeyError(f"no {config.depth_key} in the batch: this model reads RGB-D")
        for key in config.image_features:
            image = batch[key]
            rgb = (image[:, :3] - self.image_mean.to(image.dtype)) / self.image_std.to(image.dtype)
            if config.depth_as_channel and key == config.rgb_key:
                rgb = torch.cat([rgb, batch[config.depth_key][:, :1].to(rgb.dtype)], dim=1)
            if tuple(rgb.shape[-2:]) != tuple(config.image_size):
                # Bilinear on half-pixel centres, as act_pipeline (and cv2) resize.
                rgb = F.interpolate(rgb, size=tuple(config.image_size), mode="bilinear", align_corners=False)
            batch[key] = rgb
        return batch

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        return super().forward(self.prepare_images(batch))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return super().predict_action_chunk(self.prepare_images(batch))
