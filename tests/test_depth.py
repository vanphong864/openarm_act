"""Depth handling: training flags, raw depth in the dataset, the depth branch
(CNN, LingBot-Depth), distillation, and train -> serve with each."""

import json
import uuid

import numpy as np
import pytest
import torch
from hydra import compose
from hydra import initialize_config_dir

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths
from openarm_act_lerobot.cli import expand_flags

SMALL_LINGBOT = {"backbone": "dinov2_vits14", "dim_out": 64}


def compose_cfg(name="train", overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(paths.CONF_DIR)):
        return compose(config_name=name, overrides=list(overrides))


def flags(*args):
    return expand_flags(["script", *args])[1:]


# -------------------------------------------------------------------- flags


def test_flags_become_overrides():
    assert flags("--resnet") == ["backbone=resnet18"]
    assert flags("--dinov2", "exp_name=x") == ["exp_name=x", "backbone=dinov2_vitb14"]
    assert flags("--backbone", "dinov3_vits16") == ["backbone=dinov3_vits16"]
    assert flags("--backbone=resnet50") == ["backbone=resnet50"]
    assert flags("--lingbot-depth") == ["data=openarm_rgbd_branch", "depth_encoder=lingbot_depth"]
    assert flags("--lingbot-depth-distill", "--resnet") == [
        "data=openarm_rgbd_branch", "distill=lingbot_depth", "backbone=resnet18"]
    assert flags("--depth-branch", "--lingbot-depth") == ["data=openarm_rgbd_branch", "depth_encoder=lingbot_depth"]
    # What the command line says itself wins over what a flag implies.
    assert flags("--lingbot-depth-distill", "data=openarm_rgbd_branch", "data.depth_as=none") == [
        "data=openarm_rgbd_branch", "data.depth_as=none", "distill=lingbot_depth"]
    assert flags("--resnet", "backbone=resnet34") == ["backbone=resnet34"]
    assert flags("--cfg", "job", "-m") == ["--cfg", "job", "-m"]  # Hydra's own


@pytest.mark.parametrize("args", [["--resent"], ["--resnet", "--dinov2"], ["--lingbot-depth", "--lingbot-depth-distill"],
                                  ["--backbone"]])
def test_bad_flags_stop(args):
    with pytest.raises(SystemExit):
        flags(*args)


@pytest.mark.parametrize(("args", "name", "depth_as", "encoder", "distill"), [
    ([], "act_dinov3_vitb16_frozen", "channel", "cnn", "none"),
    (["--resnet", "--depth-branch"], "act_resnet18_frozen", "branch", "cnn", "none"),
    (["--dinov2", "--lingbot-depth"], "act_dinov2_vitb14_frozen_lingbotdepth", "branch", "lingbot_depth", "none"),
    (["--lingbot-depth-distill"], "act_dinov3_vitb16_frozen_distill", "branch", "cnn", "lingbot_depth"),
    (["--lingbot-depth-distill", "data.depth_as=none"], "act_dinov3_vitb16_frozen_distill", "none", "cnn",
     "lingbot_depth"),
])
def test_flags_compose_into_a_run(args, name, depth_as, encoder, distill):
    from openarm_act_lerobot.train_config import policy_overrides

    cfg = compose_cfg(overrides=flags(*args, "device=cpu"))
    assert cfg.name == name
    fields = policy_overrides(cfg)
    assert (fields["depth_as"], fields["depth_encoder"], fields["distill"]) == (depth_as, encoder, distill)
    assert fields["depth_format"] == ("gray8" if depth_as == "channel" else "raw")
    if "lingbot_depth" in (encoder, distill):
        assert fields["lingbot_depth_repo"] == "robbyant/lingbot-depth-pretrain-vitl-14"


def test_configs_refuse_what_cannot_work():
    from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig

    with pytest.raises(ValueError, match="depth_format=raw"):
        OpenArmACTConfig(depth_as="branch", depth_format="gray8")
    with pytest.raises(ValueError, match="depth_format=raw"):
        OpenArmACTConfig(depth_as="channel", depth_format="gray8", distill="lingbot_depth")
    with pytest.raises(ValueError, match="already the depth encoder"):
        OpenArmACTConfig(depth_as="branch", depth_format="raw", depth_encoder="lingbot_depth", distill="lingbot_depth")
    from openarm_act_lerobot.train_config import policy_overrides

    with pytest.raises(ValueError, match="chest_depth"):
        policy_overrides(compose_cfg(overrides=["data=openarm_rgb", "distill=lingbot_depth"]))


# ---------------------------------------------------------------- raw depth


def _convert_raw(tmp_path, episode_factory, extra=()):
    from openarm_act_lerobot.convert import convert

    episode_factory("2026-10-08/episode_a", n=80)
    episode_factory("2026-10-08/episode_b", n=60, seed=1)
    overrides = [f"paths.raw={tmp_path / 'raw'}", f"paths.assets={tmp_path / 'assets'}",
                 f"paths.checkpoints={tmp_path / 'checkpoints'}", "data=openarm_rgbd_branch",
                 f"data.manifest={tmp_path / 'none.yaml'}", f"data.name=test_{uuid.uuid4().hex[:12]}",
                 "data.val_fraction=0", *extra]
    return convert(compose_cfg("convert", overrides)), overrides


@pytest.mark.parametrize("video", [False, True])
def test_raw_depth_round_trip(tmp_path, episode_factory, video):
    """The dataset gives back the recorder's depth in metres; missing readings
    stay below depth_valid_min_m; serving builds the very same tensor."""
    from lerobot.datasets import LeRobotDataset

    from openarm_act_lerobot.episode import EpisodeReader
    from openarm_act_lerobot.images import depth_to_metres

    report, overrides = _convert_raw(tmp_path, episode_factory, [f"data.video={str(video).lower()}"])
    cfg = compose_cfg(overrides=overrides)
    dataset = LeRobotDataset(str(cfg.data.repo_id), depth_output_unit="m")
    assert dataset.meta.depth_keys == ["observation.images.chest_depth"]
    with EpisodeReader(tmp_path / "raw/2026-10-08/episode_a.hdf5") as ep:
        raw = ep.depth(5).copy()
    sample = np.asarray(dataset[5]["observation.images.chest_depth"])
    served = depth_to_metres(raw)
    assert sample.shape == served.shape == (1, 240, 424)
    tolerance = 1.5e-3 if video else 1e-6  # video: 12-bit log quantisation
    valid = raw > 100
    np.testing.assert_allclose(sample[0][valid], served[0][valid], atol=tolerance)
    assert (sample[0][raw == 0] < 0.1).all() and (served[0][raw == 0] < 0.1).all()


# ------------------------------------------------------------- depth branch


def make_config(family="dinov3", **kwargs):
    from lerobot.configs import FeatureType
    from lerobot.configs import PolicyFeature

    from openarm_act_lerobot.backbones import vit_config
    from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig

    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 60, 80))
    depth = PolicyFeature(type=FeatureType.VISUAL, shape=(1, 60, 80))
    fields = {
        "input_features": {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
                           "observation.images.chest": image, "observation.images.chest_depth": depth},
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        "device": "cpu", "dim_model": 32, "n_heads": 2, "dim_feedforward": 64, "n_encoder_layers": 1,
        "n_vae_encoder_layers": 1, "chunk_size": 10, "n_action_steps": 10, "backbone_type": family,
        "load_backbone_weights": False, "depth_as": "branch", "depth_format": "raw",
        "lingbot_depth_config": {"backbone": "dinov2_vits14", "intermediate_layers": 1, "dim_out": 64,
                                 "strict": False, "depth_emb_mode": "conv_1c", "img_depth_fuse_mode": "cat_token"},
    }
    small = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
             "num_register_tokens": 4}
    if family == "resnet":
        fields |= {"vision_backbone": "resnet18", "image_size": (64, 96)}
    else:
        fields |= {"backbone_config": vit_config(family, overrides=small),
                   "image_size": (48, 64) if family == "dinov3" else (42, 56)}
    return OpenArmACTConfig(**(fields | kwargs))


def batch(n=2, holes=True):
    rng = torch.Generator().manual_seed(0)
    depth = 0.3 + torch.rand(n, 1, 60, 80, generator=rng)
    if holes:
        depth[:, :, :10, :10] = 0.0
    return {"observation.images.chest": torch.rand(n, 3, 60, 80, generator=rng),
            "observation.images.chest_depth": depth, "observation.state": torch.randn(n, 16, generator=rng)}


@pytest.mark.parametrize("family", ["dinov3", "dinov2", "resnet"])
@pytest.mark.parametrize("encoder", ["cnn", "lingbot_depth"])
def test_depth_branch_starts_as_rgb_and_learns_depth(family, encoder):
    from openarm_act_lerobot.backbones import RGBDFusion
    from openarm_act_lerobot.modeling_openarm_act import OpenArmACTPolicy

    torch.manual_seed(0)
    policy = OpenArmACTPolicy(make_config(family, depth_encoder=encoder)).eval()
    fusion = policy.model.backbone
    assert isinstance(fusion, RGBDFusion) and fusion.first_conv().in_channels == 3  # RGB backbone as pretrained
    assert list(policy.config.image_features) == ["observation.images.chest"]
    b = batch()
    image = policy.prepare_images(b)["observation.images.chest"]
    assert image.shape == (2, 5, *policy.config.image_size)  # rgb | depth | valid
    assert image[:, 4].min() == 0 and image[:, 4].max() == 1
    out = fusion(image)["feature_map"]
    rgb_only = fusion.rgb((image[:, :3] - fusion.mean) / fusion.std)["feature_map"]
    torch.testing.assert_close(out, rgb_only)  # the depth branch starts at zero

    # The depth branch trains from scratch (ACT's lr); the RGB backbone is frozen.
    groups = policy.get_optim_params()
    names = {id(p): n for n, p in policy.named_parameters()}
    assert len(groups) == 1 and any(names[id(p)].startswith("model.backbone.depth.") for p in groups[0]["params"])
    assert not any(p.requires_grad for p in fusion.rgb.parameters())
    if encoder == "lingbot_depth":
        assert not any(p.requires_grad for p in fusion.depth.lingbot.parameters())
        fusion.train()
        assert not fusion.depth.lingbot.training  # always eval: no depth masking, no drop path

    with torch.no_grad():
        fusion.depth.proj.weight.normal_(std=0.1)
    a = policy.predict_action_chunk(b)
    far = dict(b, **{"observation.images.chest_depth": b["observation.images.chest_depth"] + 0.4})
    assert not torch.allclose(a, policy.predict_action_chunk(far))


def test_lingbot_depth_checkpoint_keys(tmp_path):
    """The published checkpoint's depth_patch_embed and lingbot-vla-v2's
    depth_mask_patch_embed both load, strictly; anything less is refused."""
    from openarm_act_lerobot.backbones import LingBotDepthEncoder

    config = make_config(lingbot_depth_repo=str(tmp_path))
    source = LingBotDepthEncoder(config, load_weights=False)
    state = {f"encoder.{k}": v for k, v in source.encoder.state_dict().items()}
    renamed = {k.replace("depth_patch_embed", "depth_mask_patch_embed"): v for k, v in state.items()}
    for weights in (state, renamed):
        torch.save({"model": weights, "model_config": {"encoder": config.lingbot_depth_config}}, tmp_path / "model.pt")
        loaded = LingBotDepthEncoder(config, load_weights=True)
        for key, value in source.encoder.state_dict().items():
            torch.testing.assert_close(loaded.encoder.state_dict()[key], value)
    del state["encoder.backbone.depth_patch_embed.proj.weight"]
    torch.save({"model": state}, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="does not fit"):
        LingBotDepthEncoder(config, load_weights=True)


def test_published_lingbot_depth_loads(tmp_path):
    """The real checkpoint (when it is in the Hugging Face cache): every encoder
    weight loads, and a real-sized frame gives finite features on the grid."""
    from huggingface_hub import try_to_load_from_cache

    from openarm_act_lerobot.backbones import LingBotDepthEncoder
    from openarm_act_lerobot.configuration_openarm_act import LINGBOT_DEPTH_REPO
    from openarm_act_lerobot.configuration_openarm_act import LINGBOT_DEPTH_REVISION
    from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig

    if not isinstance(try_to_load_from_cache(LINGBOT_DEPTH_REPO, "model.pt", revision=LINGBOT_DEPTH_REVISION), str):
        pytest.skip("LingBot-Depth's model.pt is not in the Hugging Face cache")
    encoder = LingBotDepthEncoder(OpenArmACTConfig(), load_weights=True)
    rgb, depth = torch.rand(1, 3, 240, 320), 0.3 + torch.rand(1, 1, 240, 320)
    features = encoder(rgb, depth, (15, 20))
    assert features.shape == (1, 1024, 15, 20) and torch.isfinite(features).all()


# ---------------------------------------------------------------- distill


@pytest.mark.parametrize("depth_as", ["branch", "none"])
def test_distillation_trains_the_encoder_tokens(depth_as):
    from openarm_act_lerobot.modeling_openarm_act import OpenArmACTPolicy

    torch.manual_seed(0)
    policy = OpenArmACTPolicy(make_config(depth_as=depth_as, distill="lingbot_depth"))
    policy.train()
    b = batch() | {"action": torch.randn(2, 10, 16), "action_is_pad": torch.zeros(2, 10, dtype=torch.bool)}
    loss, parts = policy.forward(b)
    assert 0 < parts["distill_loss"] <= 2  # 1 - cosine
    loss.backward()
    assert policy.distill_head[1].weight.grad.abs().sum() > 0
    assert policy.model.encoder.layers[0].linear1.weight.grad.abs().sum() > 0  # ACT's encoder learns from the teacher
    assert all(p.grad is None for p in policy.teacher.parameters())

    # The teacher is not part of the model: not saved, not run at inference.
    assert not any("teacher" in k or "lingbot" in k for k in policy.state_dict())
    assert any(k.startswith("distill_head.") for k in policy.state_dict())
    policy.__dict__.pop("_teacher")
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 10, 16)
    assert "_teacher" not in policy.__dict__
    policy.eval()
    _, parts = policy.forward(b)
    assert "distill_loss" not in parts


# ------------------------------------------------- end to end, on a CPU


@pytest.mark.parametrize("args", [["--depth-branch"], ["--resnet", "--lingbot-depth"], ["--lingbot-depth-distill"]])
def test_train_resume_export_serve(tmp_path, episode_factory, monkeypatch, args):
    """LeRobot's loop with each depth flag: train, resume, export, then serve
    from the release alone (no Hub) a frame as the robot's client sends it."""
    from openarm_act_lerobot.export import export
    from openarm_act_lerobot.norm_stats import compute_norm_stats
    from openarm_act_lerobot.policy import OpenArmPolicy
    from openarm_act_lerobot.run import train

    _, converted = _convert_raw(tmp_path, episode_factory, ["data.video=false"])
    overrides = flags(*args, *converted, "experiment=smoke", "num_train_steps=4", "save_interval=2")
    cfg = compose_cfg(overrides=overrides)
    compute_norm_stats(cfg)
    run = train(cfg)
    if "--lingbot-depth-distill" in args:
        weights = json.loads((run / "checkpoints/000004/training_state/optimizer_param_groups.json").read_text())
        assert len(weights) == 2  # ACT + the trainable bit of the backbone
    train(compose_cfg(overrides=[*overrides, "num_train_steps=6", "resume=true", "overwrite=false"]))
    import subprocess
    import sys

    report = subprocess.run([sys.executable, str(paths.REPO_ROOT / "scripts/inspect_checkpoint.py"), f"checkpoint={run}"],
                            capture_output=True, text=True, check=True).stdout
    assert "model.backbone.depth.proj.weight: norm" in report and "R " in report
    release = export(compose_cfg("export", [f"checkpoint={run}", f"paths.releases={tmp_path / 'releases'}"]))

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    policy = OpenArmPolicy(release, device="cpu")
    assert policy.metadata["depth_format"] == "raw" and policy.metadata["cameras"] == ["chest", "chest_depth"]
    rng = np.random.default_rng(0)
    obs = {"images": {"chest": rng.integers(0, 256, (240, 424, 3), dtype=np.uint8),
                      "chest_depth": rng.integers(0, 1500, (240, 424), dtype=np.uint16)},
           "state": rng.normal(size=16).astype(np.float32)}
    actions = policy.infer(obs)["actions"]
    assert actions.shape == (50, 16) and np.isfinite(actions).all()
    assert "_teacher" not in policy.policy.__dict__
    assert C.NUM_JOINTS == actions.shape[1]
