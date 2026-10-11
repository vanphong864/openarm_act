"""ACT's vision backbones: ResNet (ACT's own), DINOv2, DINOv3.

Each takes a normalised image [B, 3|4, H, W], H and W multiples of its
stride, and returns ACT's {"feature_map": [B, C, H / stride, W / stride]},
which ACT projects to dim_model with a 1x1 conv and gives its 2D sinusoidal
position embedding. For the ViTs the map is their patch tokens (CLS and
register tokens dropped), after the final layer norm.

The same rules for all three (configuration_openarm_act):

    4 channels   the first layer -- ResNet's conv1, the ViT's patch
                 embedding -- is widened from 3 to 4 input channels; the 4th
                 channel's weights start per depth_init
    trainable    backbone_trainable none | last | all; the widened first
                 layer trains anyway with train_patch_embedding (act_pipeline
                 trains its conv1_adapter and freezes the rest of ResNet-18)
    no graph     a backbone with nothing to train runs under no_grad
"""

from __future__ import annotations

import contextlib
import logging

import torch
from torch import nn

log = logging.getLogger(__name__)

# The ViT config fields that define the network, kept in the checkpoint's
# config (OpenArmACTConfig.backbone_config).
VIT_CONFIG_KEYS = {
    "dinov3": (
        "patch_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "hidden_act", "attention_dropout", "initializer_range", "layer_norm_eps", "rope_theta", "image_size",
        "num_channels", "query_bias", "key_bias", "value_bias", "proj_bias", "mlp_bias", "layerscale_value",
        "drop_path_rate", "use_gated_mlp", "num_register_tokens",
        "pos_embed_shift", "pos_embed_jitter", "pos_embed_rescale",
    ),
    "dinov2": (
        "patch_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio", "hidden_act",
        "hidden_dropout_prob", "attention_probs_dropout_prob", "initializer_range", "layer_norm_eps",
        "image_size", "num_channels", "qkv_bias", "layerscale_value", "drop_path_rate", "use_swiglu_ffn",
        "use_mask_token",
    ),
}  # fmt: skip
# DINOv3 jitters and rescales its RoPE coordinates in training mode, a
# pretraining augmentation. Off here: ACT trains and serves the same geometry.
NO_ROPE_AUGMENTATION = {"pos_embed_shift": None, "pos_embed_jitter": None, "pos_embed_rescale": None}


def vit_classes(family: str):
    """(config class, model class) of a ViT family, from transformers."""
    import transformers

    names = {"dinov3": ("DINOv3ViTConfig", "DINOv3ViTModel"), "dinov2": ("Dinov2Config", "Dinov2Model")}
    if family not in names:
        raise ValueError(f"no ViT family {family!r}; there are {sorted(names)}")
    return tuple(getattr(transformers, name) for name in names[family])


def vit_config(family: str, repo: str | None = None, revision: str | None = None, overrides: dict | None = None):
    """The ViT config's fields, as OpenArmACTConfig.backbone_config holds them:
    the repo's config.json (a few hundred bytes; a gated repo needs HF_TOKEN),
    or the config class's defaults when no repo is given, with `overrides` on top."""
    config_class, _ = vit_classes(family)
    config = config_class.from_pretrained(repo, revision=revision) if repo else config_class()
    values = {key: getattr(config, key) for key in VIT_CONFIG_KEYS[family]}
    values.update({k: v for k, v in (overrides or {}).items() if k in VIT_CONFIG_KEYS[family]})
    if family == "dinov3":
        values.update(NO_ROPE_AUGMENTATION)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}


def widen_weight(weight: torch.Tensor, depth_init: str) -> torch.Tensor:
    """[out, 3, k, k] -> [out, 4, k, k], the 4th channel per depth_init."""
    if depth_init == "mean":
        depth = weight.mean(dim=1, keepdim=True)
    else:
        depth = torch.zeros_like(weight[:, :1])
    return torch.cat([weight, depth], dim=1)


def widen_conv(old: nn.Conv2d, depth_init: str) -> nn.Conv2d:
    """A copy of a 3-channel first conv that takes a 4th (depth) channel."""
    if old.in_channels != 3:
        raise ValueError(f"cannot widen a {old.in_channels}-channel conv to 4")
    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(widen_weight(old.weight, depth_init))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new.to(device=old.weight.device, dtype=old.weight.dtype)


class Backbone(nn.Module):
    """What the three backbones share: the first layer, the blocks that the
    `last` setting counts back from, and the trainable rules."""

    stride: int
    hidden_size: int

    def first_conv(self) -> nn.Conv2d:
        raise NotImplementedError

    def set_first_conv(self, conv: nn.Conv2d) -> None:
        raise NotImplementedError

    def blocks(self) -> list[nn.Module]:
        """In order; `last` trains the final backbone_last_layers of them."""
        raise NotImplementedError

    def head(self) -> list[nn.Module]:
        """What trains along with the last blocks (a ViT's final norm)."""
        return []

    def configure(self, config) -> None:
        if config.image_channels == 4 and self.first_conv().in_channels == 3:
            self.set_first_conv(widen_conv(self.first_conv(), config.depth_init))
        self.requires_grad_(config.backbone_trainable == "all")
        if config.backbone_trainable == "last" and config.backbone_last_layers > 0:
            for module in (*self.blocks()[-int(config.backbone_last_layers) :], *self.head()):
                module.requires_grad_(True)
        if config.depth_as_channel and config.train_patch_embedding:
            self.first_conv().requires_grad_(True)

    def features(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        trains = any(p.requires_grad for p in self.parameters())
        with contextlib.nullcontext() if trains else torch.no_grad():
            return {"feature_map": self.features(image)}


class ResNetBackbone(Backbone):
    """ACT's own: the torchvision ResNet LeRobot builds (FrozenBatchNorm2d,
    up to layer4), wrapped to follow the trainable rules."""

    def __init__(self, body: nn.Module, config):
        super().__init__()
        self.body = body  # IntermediateLayerGetter(resnet, {"layer4": "feature_map"})
        self.stride = config.backbone_stride
        last = self.body.layer4[-1]  # a bottleneck block (resnet50+) ends in conv3
        self.hidden_size = int((last.conv3 if hasattr(last, "conv3") else last.conv2).out_channels)
        self.configure(config)

    def first_conv(self) -> nn.Conv2d:
        return self.body.conv1

    def set_first_conv(self, conv: nn.Conv2d) -> None:
        self.body.conv1 = conv

    def blocks(self) -> list[nn.Module]:
        return [self.body.layer1, self.body.layer2, self.body.layer3, self.body.layer4]

    def features(self, image: torch.Tensor) -> torch.Tensor:
        return self.body(image)["feature_map"]


class ViTBackbone(Backbone):
    """DINOv2 or DINOv3 from transformers: patch tokens as a feature map."""

    def __init__(self, config, load_weights: bool):
        super().__init__()
        self.family = config.backbone_type
        config_class, model_class = vit_classes(self.family)
        if config.backbone_config is None:
            raise ValueError("backbone_config is not set (train_config.build_policy_config fills it)")
        vit_config = config_class(**config.backbone_config)
        if load_weights:
            log.info("%s weights from %s@%s", self.family, config.backbone_repo, config.backbone_revision)
            self.vit = model_class.from_pretrained(
                config.backbone_repo, revision=config.backbone_revision, config=vit_config, dtype=torch.float32
            )
        else:
            self.vit = model_class(vit_config)
        self.stride = int(vit_config.patch_size)
        self.hidden_size = int(vit_config.hidden_size)
        # CLS, then DINOv3's register tokens, then the patches.
        self.num_prefix_tokens = 1 + int(getattr(vit_config, "num_register_tokens", 0) or 0)
        self.configure(config)
        if config.gradient_checkpointing:
            self.vit.gradient_checkpointing_enable()

    def first_conv(self) -> nn.Conv2d:
        patch = self.vit.embeddings.patch_embeddings
        return patch if self.family == "dinov3" else patch.projection

    def set_first_conv(self, conv: nn.Conv2d) -> None:
        if self.family == "dinov3":
            self.vit.embeddings.patch_embeddings = conv
        else:
            self.vit.embeddings.patch_embeddings.projection = conv
            self.vit.embeddings.patch_embeddings.num_channels = conv.in_channels  # it checks the input against this

    def blocks(self) -> list[nn.Module]:
        return list(self.vit.model.layer if self.family == "dinov3" else self.vit.encoder.layer)

    def head(self) -> list[nn.Module]:
        return [self.vit.norm if self.family == "dinov3" else self.vit.layernorm]

    def features(self, image: torch.Tensor) -> torch.Tensor:
        height, width = (int(v) // self.stride for v in image.shape[-2:])
        tokens = self.vit(pixel_values=image).last_hidden_state[:, self.num_prefix_tokens :]
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)
