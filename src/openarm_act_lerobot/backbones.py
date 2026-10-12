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


# ------------------------------------------------------------ depth branch


def normalise_depth(depth_m: torch.Tensor, valid: torch.Tensor, config) -> torch.Tensor:
    """Metres -> about zero-mean, unit-spread over depth_min_m..depth_max_m,
    as the RGB channels after ImageNet normalisation; 0 where there is no reading."""
    span = config.depth_max_m - config.depth_min_m
    scaled = (depth_m.clamp(config.depth_min_m, config.depth_max_m) - config.depth_min_m) / span
    return (scaled - 0.5) / 0.25 * valid


class DepthEncoder(nn.Module):
    """A small CNN from [normalised depth, valid] to the RGB backbone's token
    grid: stride-2 convs, then pooled to (h, w) and projected to its width.
    The projection starts at zero, so a fresh model reads RGB only."""

    def __init__(self, out_channels: int, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, width = [], 2
        for c in channels:
            layers += [nn.Conv2d(width, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.ReLU(inplace=True)]
            width = c
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Conv2d(width, out_channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, depth: torch.Tensor, valid: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        x = self.convs(torch.cat([depth, valid], dim=1))
        return self.proj(nn.functional.adaptive_avg_pool2d(x, grid))


def lingbot_depth_state(repo: str, revision: str | None) -> dict:
    """The encoder's weights from LingBot-Depth's model.pt (a Hub repo or a
    file), keys as LingBotDepthEncoder names them. The published checkpoint
    calls the depth patch embedding depth_patch_embed, lingbot-vla-v2's code
    depth_mask_patch_embed: either loads."""
    from pathlib import Path

    path = Path(repo)
    if path.is_dir():
        path = path / "model.pt"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "model.pt", revision=revision)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key.removeprefix("encoder.").replace("depth_mask_patch_embed", "depth_patch_embed"): value
        for key, value in checkpoint["model"].items()
        if key.startswith("encoder.")
    }


class LingBotDepthEncoder(nn.Module):
    """LingBot-Depth's RGB-D ViT-L/14 (lingbot_depth/, Apache 2.0), frozen.

    rgb [B, 3, H, W] in 0..1 and depth [B, 1, H, W] in metres (0 = no
    reading) -> its features [B, 1024, h, w] on the grid asked for: it resizes
    the image to (14 h, 14 w) itself, normalises RGB with ImageNet's stats and
    takes log depth. Its random masking of depth tokens, a pretraining
    augmentation, is off. Always in eval mode, always without a graph.
    """

    def __init__(self, config, load_weights: bool):
        super().__init__()
        from openarm_act_lerobot.lingbot_depth.rgbd_encoder import DINOv2_RGBD_Encoder

        self.encoder = DINOv2_RGBD_Encoder(**config.lingbot_depth_config)
        if load_weights:
            log.info("LingBot-Depth weights from %s@%s", config.lingbot_depth_repo, config.lingbot_depth_revision)
            state = lingbot_depth_state(config.lingbot_depth_repo, config.lingbot_depth_revision)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(f"LingBot-Depth checkpoint does not fit its encoder: missing {missing[:5]}, "
                                 f"unexpected {unexpected[:5]}")
        self.out_channels = int(config.lingbot_depth_config["dim_out"])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, rgb: torch.Tensor, depth_m: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        with torch.no_grad():
            features, *_ = self.encoder(rgb.float(), depth_m.float(), int(grid[0]), int(grid[1]),
                                        remap_depth_in="log", enable_depth_mask=False)
        return features


class LingBotDepthBranch(nn.Module):
    """LingBot-Depth's features, projected to the RGB backbone's width by a
    trained 1x1 conv that starts at zero."""

    def __init__(self, out_channels: int, config, load_weights: bool):
        super().__init__()
        self.lingbot = LingBotDepthEncoder(config, load_weights)
        self.proj = nn.Conv2d(self.lingbot.out_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, rgb: torch.Tensor, depth_m: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        return self.proj(self.lingbot(rgb, depth_m, grid))


class RGBDFusion(Backbone):
    """depth_as=branch: the RGB backbone and a depth encoder side by side,
    their feature maps added. Takes [rgb 0..1 (3) | depth m (1) | valid (1)]
    at the model's image size; normalises RGB for the backbone itself."""

    def __init__(self, rgb: Backbone, config, load_weights: bool):
        super().__init__()
        self.rgb = rgb
        self.stride, self.hidden_size = rgb.stride, rgb.hidden_size
        self.kind = config.depth_encoder
        if self.kind == "cnn":
            self.depth = DepthEncoder(rgb.hidden_size, config.depth_encoder_channels)
        else:
            self.depth = LingBotDepthBranch(rgb.hidden_size, config, load_weights)
        self.config = config
        self.register_buffer("mean", torch.tensor(config.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(config.image_std).view(1, 3, 1, 1), persistent=False)

    def first_conv(self) -> nn.Conv2d:
        return self.rgb.first_conv()

    def blocks(self) -> list[nn.Module]:
        return self.rgb.blocks()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb, depth_m, valid = image[:, :3], image[:, 3:4], image[:, 4:5]
        rgb_map = self.rgb((rgb - self.mean.to(rgb.dtype)) / self.std.to(rgb.dtype))["feature_map"]
        grid = tuple(rgb_map.shape[-2:])
        if self.kind == "cnn":
            depth_map = self.depth(normalise_depth(depth_m, valid, self.config), valid, grid)
        else:
            depth_map = self.depth(rgb, depth_m * valid, grid)
        return {"feature_map": rgb_map + depth_map.to(rgb_map.dtype)}
