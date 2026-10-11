"""The three backbones inside ACT: depth channel, trainable rules, image
preparation, and loading a checkpoint without the Hub."""

import pytest
import torch
from lerobot.configs import FeatureType
from lerobot.configs import PolicyFeature

from openarm_act_lerobot.backbones import vit_classes
from openarm_act_lerobot.backbones import vit_config
from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig
from openarm_act_lerobot.modeling_openarm_act import OpenArmACTPolicy

SMALL_VIT = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 2,
             "num_register_tokens": 4}
RGB, DEPTH = "observation.images.chest", "observation.images.chest_depth"


def make_config(family="dinov3", depth_as="channel", **kwargs) -> OpenArmACTConfig:
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 60, 80))
    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)), RGB: image}
    if depth_as:
        inputs[DEPTH] = image
    fields = {
        "input_features": inputs,
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        "device": "cpu", "dim_model": 32, "n_heads": 2, "dim_feedforward": 64, "n_encoder_layers": 1,
        "n_vae_encoder_layers": 1, "chunk_size": 10, "n_action_steps": 10,
        "backbone_type": family, "depth_as": depth_as or "image", "load_backbone_weights": False,
    }
    if family == "resnet":
        fields |= {"vision_backbone": "resnet18", "image_size": (64, 96)}
    else:
        fields |= {"backbone_config": vit_config(family, overrides=SMALL_VIT),
                   "image_size": (48, 64) if family == "dinov3" else (42, 56)}
    return OpenArmACTConfig(**(fields | kwargs))


def pretrained_vit(tmp_path, family):
    """A small 'pretrained' ViT saved as a Hub repo would hold it."""
    config_class, model_class = vit_classes(family)
    torch.manual_seed(1)
    model = model_class(config_class(**vit_config(family, overrides=SMALL_VIT)))
    model.save_pretrained(tmp_path / family)
    return tmp_path / family, model


@pytest.mark.parametrize("family", ["dinov3", "dinov2"])
@pytest.mark.parametrize("depth_init", ["zero", "mean"])
def test_vit_patch_embedding_widens_pretrained_weights(tmp_path, family, depth_init):
    repo, pretrained = pretrained_vit(tmp_path, family)
    config = make_config(family, backbone_repo=str(repo), load_backbone_weights=True, depth_init=depth_init)
    policy = OpenArmACTPolicy(config)
    backbone = policy.model.backbone
    old = pretrained.embeddings.patch_embeddings
    old = (old if family == "dinov3" else old.projection).weight
    weight = backbone.first_conv().weight
    assert weight.shape[1] == 4 and weight.requires_grad  # trainable, as ACT's conv1_adapter
    torch.testing.assert_close(weight[:, :3], old)
    expected = old.mean(1) if depth_init == "mean" else torch.zeros_like(old[:, 0])
    torch.testing.assert_close(weight[:, 3], expected)
    torch.testing.assert_close(backbone.blocks()[0].mlp.down_proj.weight if family == "dinov3"
                               else backbone.blocks()[0].mlp.fc2.weight,
                               pretrained.state_dict()["model.layer.0.mlp.down_proj.weight" if family == "dinov3"
                                                       else "encoder.layer.0.mlp.fc2.weight"])
    # The rest frozen, the image projection ACT's own.
    assert not any(p.requires_grad for p in backbone.blocks()[0].parameters())
    assert policy.model.encoder_img_feat_input_proj.in_channels == 32


@pytest.mark.parametrize("family", ["dinov3", "dinov2", "resnet"])
def test_feature_map_and_trainable_rules(family):
    config = make_config(family, depth_as=None)
    policy = OpenArmACTPolicy(config)
    backbone = policy.model.backbone
    height, width = config.image_size
    out = backbone(torch.randn(2, 3, height, width))["feature_map"]
    assert out.shape == (2, backbone.hidden_size, height // backbone.stride, width // backbone.stride)
    # RGB and frozen: nothing trains, no graph, ACT alone in the optimizer.
    assert not out.requires_grad
    assert not any(p.requires_grad for p in backbone.parameters())
    assert len(policy.get_optim_params()) == 1

    last = OpenArmACTPolicy(make_config(family, depth_as=None, backbone_trainable="last", backbone_last_layers=1))
    blocks = last.model.backbone.blocks()
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert not any(p.requires_grad for p in blocks[0].parameters())
    assert not last.model.backbone.first_conv().weight.requires_grad
    groups = last.get_optim_params()
    assert len(groups) == 2 and groups[1]["lr"] == config.optimizer_lr_backbone

    everything = OpenArmACTPolicy(make_config(family, depth_as=None, backbone_trainable="all"))
    assert all(p.requires_grad for p in everything.model.backbone.parameters())


@pytest.mark.parametrize("family", ["dinov3", "resnet"])
def test_images_are_fused_resized_and_normalised(family):
    config = make_config(family)
    policy = OpenArmACTPolicy(config)
    rgb, depth = torch.rand(2, 3, 60, 80), torch.rand(2, 3, 60, 80)
    batch = policy.prepare_images({RGB: rgb, DEPTH: depth, "observation.state": torch.zeros(2, 16)})
    image = batch[RGB]
    assert image.shape == (2, 4, *config.image_size)
    mean, std = torch.tensor(config.image_mean).view(1, 3, 1, 1), torch.tensor(config.image_std).view(1, 3, 1, 1)
    resize = torch.nn.functional.interpolate  # noqa: N806
    expected = resize((rgb - mean) / std, size=config.image_size, mode="bilinear", align_corners=False)
    torch.testing.assert_close(image[:, :3], expected)
    torch.testing.assert_close(image[:, 3], resize(depth[:, :1], size=config.image_size, mode="bilinear")[:, 0])
    assert list(config.image_features) == [RGB]  # depth travels inside the RGB image
    with pytest.raises(KeyError, match="chest_depth"):
        policy.prepare_images({RGB: rgb})

    # Training: L1 + KL, with the CVAE.
    policy.train()
    loss, parts = policy.forward({RGB: rgb, DEPTH: depth, "observation.state": torch.randn(2, 16),
                                  "action": torch.randn(2, 10, 16), "action_is_pad": torch.zeros(2, 10, dtype=bool)})
    assert loss.requires_grad and set(parts) == {"l1_loss", "kld_loss"}
    assert policy.predict_action_chunk({RGB: rgb, DEPTH: depth, "observation.state": torch.zeros(2, 16)}).shape == (
        2, 10, 16)


def test_a_checkpoint_loads_without_the_hub(tmp_path):
    """A run trained on pretrained weights loads from its own file: the repo
    it named is never contacted, and a file that does not fit is refused."""
    config = make_config("dinov3", backbone_repo="facebook/no-such-repo", load_backbone_weights=False)
    policy = OpenArmACTPolicy(config)
    policy.config.load_backbone_weights = True  # as a trained run's config.json says
    policy.save_pretrained(tmp_path / "ckpt")

    loaded = OpenArmACTPolicy.from_pretrained(tmp_path / "ckpt")
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    from safetensors.torch import load_file
    from safetensors.torch import save_file

    weights = load_file(tmp_path / "ckpt" / "model.safetensors")
    weights.pop(next(k for k in weights if "backbone" in k))
    save_file(weights, tmp_path / "ckpt" / "model.safetensors")
    with pytest.raises(ValueError, match="does not fit"):
        OpenArmACTPolicy.from_pretrained(tmp_path / "ckpt")
