"""Hydra -> LeRobot configs, images, conversion, norm stats, delta steps,
small ACTs through train -> save -> load -> infer, and the client against the server."""

import contextlib
import json
import socket
import threading
import time
import uuid

import numpy as np
import pytest
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths

FAMILIES = {"dinov3": "dinov3_vitb16", "dinov2": "dinov2_vits14", "resnet": "resnet18"}


def compose_cfg(name="train", overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(paths.CONF_DIR)):
        return compose(config_name=name, overrides=list(overrides))


# ------------------------------------------------------------------ config


@pytest.mark.parametrize("backbone", sorted(p.stem for p in (paths.CONF_DIR / "backbone").glob("*.yaml")))
def test_every_backbone_composes(backbone):
    from openarm_act_lerobot.train_config import policy_overrides

    cfg = compose_cfg(overrides=[f"backbone={backbone}", "exp_name=t", "device=cpu"])
    assert cfg.name == f"act_{backbone}_frozen"
    overrides = policy_overrides(cfg)
    family = cfg.backbone.family
    assert overrides["backbone_type"] == family
    if family == "resnet":
        assert overrides["vision_backbone"] == backbone and overrides["pretrained_backbone_weights"].startswith("ResNet")
    else:
        assert overrides["backbone_repo"].startswith("facebook/dinov") and len(overrides["backbone_revision"]) == 40


@pytest.mark.parametrize("finetune", ["frozen", "partial", "full", "dummy"])
@pytest.mark.parametrize("data", ["openarm_rgb", "openarm_rgb_depth", "openarm_rgbd"])
def test_every_combination_composes(finetune, data):
    from openarm_act_lerobot.train_config import policy_overrides

    cfg = compose_cfg(overrides=[f"finetune={finetune}", f"data={data}", "exp_name=t", "device=cpu"])
    assert cfg.name == f"act_dinov3_vitb16_{finetune}"
    labels = "qpos" if data == "openarm_rgbd" else "commands"
    assert cfg.data.repo_id == f"openarm/{data}_{labels}_50hz"
    assert list(cfg.data.cameras) == (["chest"] if data == "openarm_rgb" else ["chest", "chest_depth"])
    overrides = policy_overrides(cfg)
    assert overrides["chunk_size"] == 50 and overrides["device"] == "cpu"
    assert overrides["depth_as"] == ("channel" if data == "openarm_rgbd" else "image")
    assert overrides["backbone_trainable"] == {"frozen": "none", "partial": "last"}.get(finetune, "all")
    assert overrides["load_backbone_weights"] == (finetune != "dummy")
    assert overrides["input_features"] == {}  # taken from the dataset


@pytest.mark.parametrize(
    ("backbone", "frame", "expected"),
    [
        ("dinov3_vitb16", [480, 640], (240, 320)),  # 15 x 20 = 300 tokens, as act_pipeline's ResNet-18
        ("dinov2_vitb14", [480, 640], (210, 280)),
        ("resnet18", [480, 640], (480, 640)),  # act_pipeline's input, unresized
        ("dinov3_vitb16", [240, 424], (208, 368)),  # 13 x 23, the aspect ratio kept
    ],
)
def test_image_size_follows_the_backbone(backbone, frame, expected):
    from openarm_act_lerobot.train_config import model_image_size

    cfg = compose_cfg(overrides=[f"backbone={backbone}", f"data.image_size=[{frame[0]},{frame[1]}]"])
    stride = {"dinov3": 16, "dinov2": 14, "resnet": 32}[cfg.backbone.family]
    assert model_image_size(cfg, stride) == expected
    assert model_image_size(compose_cfg(overrides=["model.image_size=[224,224]"]), 16) == (224, 224)
    with pytest.raises(ValueError, match="multiples"):
        model_image_size(compose_cfg(overrides=["model.image_size=[230,224]"]), 16)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_smoke_builds_a_train_config(tmp_path, family):
    from openarm_act_lerobot.configuration_openarm_act import OpenArmACTConfig
    from openarm_act_lerobot.train_config import build_train_config

    cfg = compose_cfg(overrides=["experiment=smoke", f"backbone={FAMILIES[family]}", "data.image_size=[240,424]"])
    train_cfg = build_train_config(cfg, tmp_path / "run")
    policy = train_cfg.policy
    assert isinstance(policy, OpenArmACTConfig) and policy.type == "openarm_act"
    assert policy.backbone_type == family and not policy.load_backbone_weights
    if family == "resnet":
        assert policy.backbone_config is None and policy.backbone_stride == 32
    else:
        assert policy.backbone_config["hidden_size"] == 64 and policy.backbone_config["num_hidden_layers"] == 2
    assert all(v % policy.backbone_stride == 0 for v in policy.image_size)
    assert train_cfg.steps == 10 and train_cfg.batch_size == 2
    assert train_cfg.scheduler.num_decay_steps == 10
    assert not train_cfg.dataset.use_imagenet_stats  # the model normalises images itself
    assert train_cfg.dataset.repo_id == "openarm/openarm_rgbd_qpos_50hz"  # the default data


def test_norm_stats_shared_across_finetuning_and_backbones():
    from openarm_act_lerobot.norm_stats import norm_stats_path

    frozen = compose_cfg(overrides=["finetune=frozen"])
    assert norm_stats_path(frozen) == norm_stats_path(compose_cfg(overrides=["finetune=full"]))
    assert norm_stats_path(frozen) == norm_stats_path(compose_cfg(overrides=["backbone=resnet18"]))
    assert norm_stats_path(compose_cfg(overrides=["data.delta_actions=true"])) != norm_stats_path(frozen)
    assert norm_stats_path(compose_cfg(overrides=["data.fps=25"])) != norm_stats_path(frozen)


def test_dotenv_and_the_wandb_key_check(tmp_path, monkeypatch):
    from openarm_act_lerobot.run import check_wandb

    env = tmp_path / ".env"
    env.write_text("# comment\nWANDB_API_KEY='abc123'\nWANDB_ENTITY=\nexport OPENARM_TEST_SET=shell\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("OPENARM_TEST_SET", "shell wins")
    assert paths.load_dotenv(env) == ["WANDB_API_KEY"]
    assert __import__("os").environ["WANDB_API_KEY"] == "abc123"
    assert __import__("os").environ["OPENARM_TEST_SET"] == "shell wins"
    check_wandb(compose_cfg())  # the key is there

    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setenv("NETRC", str(tmp_path / "no_netrc"))
    with pytest.raises(RuntimeError, match="no API key"):
        check_wandb(compose_cfg())
    check_wandb(compose_cfg(overrides=["wandb.mode=offline"]))
    check_wandb(compose_cfg(overrides=["experiment=smoke"]))  # wandb off


def test_fetch_backbone_skips_what_needs_no_download(tmp_path):
    from openarm_act_lerobot.run import fetch_backbone

    assert fetch_backbone(compose_cfg(overrides=["finetune=dummy"])) is None
    assert fetch_backbone(compose_cfg(overrides=[f"backbone.repo={tmp_path}"])) is None  # a local copy


# ------------------------------------------------------------------ images


def test_depth_encoding_is_acts():
    from openarm_act_lerobot.images import depth_to_image

    depth = np.array([[0, 200, 700, 1200, 5000]], dtype=np.uint16)
    image = depth_to_image(depth)
    assert image.shape == (1, 5, 3) and image.dtype == np.uint8
    # ACT: clip(m, 0.2, 1.2) - 0.2, over 1.0 -> 0, 0, 0.5, 1, 1
    np.testing.assert_array_equal(image[0, :, 0], [0, 0, 128, 255, 255])


def test_image_layouts_agree():
    from openarm_act_lerobot.images import parse_image
    from openarm_act_lerobot.images import to_model_input

    rgb = np.random.default_rng(0).integers(0, 256, (240, 424, 3), dtype=np.uint8)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # as LeRobot decodes
    np.testing.assert_array_equal(parse_image(chw), rgb)
    np.testing.assert_allclose(to_model_input(rgb), chw)
    assert parse_image(np.zeros((240, 424), dtype=np.uint16)).shape == (240, 424, 3)


# -------------------------------------------------------------- conversion


def _convert(tmp_path, episode_factory, data="openarm_rgb", extra=()):
    from openarm_act_lerobot.convert import convert

    episode_factory("2026-10-08/episode_a", n=80, task="pick the cup")
    episode_factory("2026-10-08/episode_b", n=60, seed=1, commands="no_grippers")
    episode_factory("2026-10-09/episode_c", n=60)
    manifest = tmp_path / "episodes.yaml"
    manifest.write_text("defaults: {task: default task}\nrules:\n  - {match: '2026-10-09/*', split: val}\n")
    overrides = [
        f"paths.raw={tmp_path / 'raw'}",
        f"paths.assets={tmp_path / 'assets'}",
        f"paths.checkpoints={tmp_path / 'checkpoints'}",
        f"data={data}",
        f"data.manifest={manifest}",
        f"data.name=test_{uuid.uuid4().hex[:12]}",  # one dataset per test, in the session's HF_LEROBOT_HOME
        *extra,
    ]
    return convert(compose_cfg("convert", overrides)), overrides


@pytest.mark.parametrize("data", ["openarm_rgb", "openarm_rgb_depth", "openarm_rgbd"])
def test_convert_and_read_back(tmp_path, episode_factory, data):
    from lerobot.datasets import LeRobotDataset

    from openarm_act_lerobot.episode import EpisodeReader
    from openarm_act_lerobot.images import depth_to_image

    report, overrides = _convert(tmp_path, episode_factory, data, ["data.video=false", "data.val_fraction=0"])
    assert [e["id"] for e in report["episodes"]] == ["2026-10-08/episode_a", "2026-10-08/episode_b"]
    assert report["skipped"] == [{"id": "2026-10-09/episode_c", "reason": "split val"}]
    cfg = compose_cfg(overrides=[*overrides, "data.val_fraction=0"])
    source = str(cfg.data.action_source)  # openarm_rgbd: qpos, the others: commands
    # commands: episode_b's grippers fall back to next_qpos and lose the last row
    frames = 80 + (60 if source == "qpos" else 59)
    assert report["frames"] == frames

    dataset = LeRobotDataset(str(cfg.data.repo_id), delta_timestamps={"action": [i / 50 for i in range(50)]})
    assert len(dataset) == frames and dataset.num_episodes == 2
    sample = dataset[0]
    assert sample["task"] == "pick the cup"
    first = tmp_path / "raw/2026-10-08/episode_a.hdf5"
    with EpisodeReader(first) as ep:
        labels = ep.labels(source)
        rgb0, depth0 = ep.rgb(0), ep.depth(0)
    np.testing.assert_allclose(np.asarray(sample["observation.state"]), labels.state[0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(sample["action"]), labels.actions[:50], atol=1e-6)
    image = (np.asarray(sample["observation.images.chest"]) * 255).round().astype(np.uint8).transpose(1, 2, 0)
    np.testing.assert_array_equal(image, rgb0)  # PNG frames are lossless
    if data != "openarm_rgb":
        depth = np.asarray(sample["observation.images.chest_depth"])
        np.testing.assert_array_equal((depth * 255).round().astype(np.uint8).transpose(1, 2, 0), depth_to_image(depth0))


def test_max_episodes_limits_the_conversion(tmp_path, episode_factory):
    report, _ = _convert(tmp_path, episode_factory, extra=["data.video=false", "data.max_episodes=1"])
    assert [e["id"] for e in report["episodes"]] == ["2026-10-08/episode_a"]
    assert {"id": "2026-10-08/episode_b", "reason": "beyond data.max_episodes (1)"} in report["skipped"]


def test_convert_reuses_a_current_dataset_only(tmp_path, episode_factory):
    from openarm_act_lerobot.convert import convert

    first, overrides = _convert(tmp_path, episode_factory, extra=["data.video=false"])
    again = compose_cfg("convert", [*overrides, "data.video=false", "reuse_current=true"])
    assert convert(again).get("reused")
    with pytest.raises(FileExistsError):  # neither flag: refuse
        convert(compose_cfg("convert", [*overrides, "data.video=false"]))
    episode_factory("2026-10-08/episode_d", n=60, seed=2)  # a new episode arrives
    report = convert(again)
    assert not report.get("reused") and len(report["episodes"]) == 3
    (tmp_path / "episodes.yaml").write_text("defaults: {task: another task}\n")  # a task is reworded
    report = convert(again)
    assert not report.get("reused") and report["episodes"][1]["task"] == "another task"
    assert convert(again).get("reused")


def test_episodes_without_a_task_get_the_fallback(tmp_path, episode_factory):
    from openarm_act_lerobot.convert import convert

    episode_factory("2026-10-08/episode_a")
    overrides = [f"paths.raw={tmp_path / 'raw'}", f"data.manifest={tmp_path / 'none.yaml'}", "data.video=false",
                 "dry_run=true", "data.val_fraction=0"]
    report = convert(compose_cfg("convert", overrides))
    assert report["episodes"][0]["task"] == "openarm teleop"  # ACT reads no prompt
    with pytest.raises(ValueError, match="no task"):  # unless the run asks for tasks
        convert(compose_cfg("convert", [*overrides, "data.require_task=true"]))


@pytest.mark.parametrize("delta", [False, True])
def test_norm_stats_cover_whole_chunks(tmp_path, episode_factory, delta):
    from openarm_act_lerobot.norm_stats import compute_norm_stats
    from openarm_act_lerobot.norm_stats import read_episodes

    _, overrides = _convert(tmp_path, episode_factory, extra=["data.video=false"])
    cfg = compose_cfg(overrides=[*overrides, f"data.delta_actions={str(delta).lower()}", "model.chunk_size=10"])
    stats = json.loads(compute_norm_stats(cfg).read_text())
    episodes = read_episodes(_root(cfg))
    targets = []
    for state, action in episodes:
        for t in range(len(action)):
            for k in range(10):
                if t + k < len(action):
                    target = action[t + k].copy()
                    if delta:
                        target[list(C.ARM_JOINTS)] -= state[t][list(C.ARM_JOINTS)]
                    targets.append(target)
    targets = np.asarray(targets)
    np.testing.assert_allclose(stats["action"]["mean"], targets.mean(0), atol=1e-6)
    np.testing.assert_allclose(stats["action"]["std"], targets.std(0), atol=1e-6)
    states = np.concatenate([s for s, _ in episodes])
    np.testing.assert_allclose(stats["observation.state"]["mean"], states.mean(0), atol=1e-6)


def _root(cfg):
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / str(cfg.data.repo_id)


# ------------------------------------------------------------ delta steps


@pytest.mark.parametrize("state_shape", [(4, 16), (4, 1, 16)])  # ACT's batches; SmolVLA's
def test_delta_steps_round_trip(state_shape):
    import torch
    from lerobot.processor import NormalizerProcessorStep
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor import UnnormalizerProcessorStep
    from lerobot.processor.converters import batch_to_transition
    from lerobot.processor.converters import policy_action_to_transition
    from lerobot.processor.converters import transition_to_batch
    from lerobot.processor.converters import transition_to_policy_action

    from openarm_act_lerobot.processors import AbsoluteActionsStep
    from openarm_act_lerobot.processors import DeltaActionsStep
    from openarm_act_lerobot.processors import add_delta_steps
    from openarm_act_lerobot.processors import link_delta_steps

    pre = PolicyProcessorPipeline(
        steps=[NormalizerProcessorStep(features={}, norm_map={})],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline(
        steps=[UnnormalizerProcessorStep(features={}, norm_map={})],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    add_delta_steps(pre, post)
    assert isinstance(pre.steps[0], DeltaActionsStep) and isinstance(post.steps[1], AbsoluteActionsStep)

    rng = torch.Generator().manual_seed(0)
    state = torch.randn(*state_shape, generator=rng)
    chunk = torch.randn(4, 50, 16, generator=rng)
    out = pre({C.LEROBOT_STATE: state, C.LEROBOT_ACTION: chunk.clone()})
    arm, grippers = list(C.ARM_JOINTS), list(C.GRIPPERS)
    flat = state.reshape(4, 1, 16)
    torch.testing.assert_close(out[C.LEROBOT_ACTION][..., arm], chunk[..., arm] - flat[:, :, arm])
    torch.testing.assert_close(out[C.LEROBOT_ACTION][..., grippers], chunk[..., grippers])
    torch.testing.assert_close(post(out[C.LEROBOT_ACTION]), chunk)

    # Serialised and loaded back, the link is gone until link_delta_steps.
    loaded = AbsoluteActionsStep(**post.steps[1].get_config())
    post.steps[1] = loaded
    with pytest.raises(RuntimeError, match="not linked"):
        post(out[C.LEROBOT_ACTION])
    link_delta_steps(pre, post)
    torch.testing.assert_close(post(out[C.LEROBOT_ACTION]), chunk)


# ------------------------------------------------- small models, end to end


def _save_dummy_checkpoint(tmp_path, episode_factory, delta=False, data="openarm_rgbd", family="dinov3"):
    """A random tiny ACT saved the way LeRobot's training saves one."""
    import torch
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_policy
    from lerobot.policies import make_pre_post_processors

    from openarm_act_lerobot.norm_stats import compute_norm_stats
    from openarm_act_lerobot.norm_stats import load_norm_stats
    from openarm_act_lerobot.processors import add_delta_steps
    from openarm_act_lerobot.run import with_stats
    from openarm_act_lerobot.train_config import build_policy_config
    from openarm_act_lerobot.train_config import dataset_image_size
    from openarm_act_lerobot.train_config import save_run_config

    _, overrides = _convert(tmp_path, episode_factory, data, extra=["data.video=false"])
    cfg = compose_cfg(overrides=[*overrides, "experiment=smoke", f"backbone={FAMILIES[family]}",
                                 f"data.delta_actions={str(delta).lower()}"])
    stats = load_norm_stats(compute_norm_stats(cfg))
    meta = LeRobotDatasetMetadata(str(cfg.data.repo_id))
    torch.manual_seed(0)
    policy = make_policy(build_policy_config(cfg), ds_meta=meta)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=with_stats(meta.stats, stats))
    if delta:
        add_delta_steps(pre, post)
    out = tmp_path / "checkpoints/run/checkpoints/000010/pretrained_model"
    policy.save_pretrained(out)
    (out.parents[1] / "last").symlink_to("000010")  # as LeRobot's update_last_checkpoint
    pre.save_pretrained(out)
    post.save_pretrained(out)
    cfg.data.image_size = dataset_image_size(cfg)
    save_run_config(cfg, out)
    return out, cfg


@pytest.mark.parametrize(("family", "delta", "data"), [
    ("dinov3", False, "openarm_rgbd"),
    ("dinov3", True, "openarm_rgb"),
    ("dinov2", False, "openarm_rgb_depth"),
    ("resnet", True, "openarm_rgbd"),
])
def test_saved_model_serves_through_openarm_policy(tmp_path, episode_factory, family, delta, data):
    import torch

    from openarm_act_lerobot.policy import OpenArmPolicy

    out, cfg = _save_dummy_checkpoint(tmp_path, episode_factory, delta, data, family)
    assert json.loads((out / "config.json").read_text())["type"] == "openarm_act"
    policy = OpenArmPolicy(out.parents[2], device="cpu")  # the run directory: its newest step
    assert type(policy.policy).__name__ == "OpenArmACTPolicy" and policy.policy.config.backbone_type == family
    assert policy.directory == out.resolve()
    assert policy.metadata["step"] == 10 and policy.metadata["image_size"] == [240, 424]
    assert policy.metadata["backbone"] == FAMILIES[family]

    rng = np.random.default_rng(0)
    obs = {"images": {"chest": rng.integers(0, 256, (240, 424, 3), dtype=np.uint8),
                      "chest_depth": rng.integers(0, 1500, (240, 424), dtype=np.uint16)},
           "state": rng.normal(size=16).astype(np.float32)}
    actions = policy.infer(obs)["actions"]
    assert actions.shape == (50, 16) and actions.dtype == np.float32 and np.all(np.isfinite(actions))
    np.testing.assert_array_equal(policy.infer(obs)["actions"], actions)  # z = 0: deterministic

    # By hand: the normaliser's stats, delta or not.
    with torch.inference_mode():
        batch = policy.preprocessor(policy.batch(obs))
        raw = policy.policy.predict_action_chunk(batch)[0]
    stats = json.loads((paths.resolve(cfg.paths.assets) / f"h50_{'delta' if delta else 'absolute'}"
                        / str(cfg.data.repo_id) / "stats.json").read_text())["action"]
    expected = raw.numpy() * np.asarray(stats["std"]) + np.asarray(stats["mean"])
    if delta:
        expected[:, list(C.ARM_JOINTS)] += obs["state"][list(C.ARM_JOINTS)]
    np.testing.assert_allclose(actions, expected, rtol=1e-4, atol=1e-4)

    if data != "openarm_rgb":
        with pytest.raises(KeyError, match="chest_depth"):
            policy.infer(dict(obs, images={"chest": obs["images"]["chest"]}))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_depth_reaches_the_network(tmp_path, episode_factory, family):
    """data=openarm_rgbd: the client's raw RGB + depth + qpos in, and the
    depth channel changes the chunk once its weights are not zero."""
    from openarm_act_lerobot.policy import OpenArmPolicy

    out, _ = _save_dummy_checkpoint(tmp_path, episode_factory, data="openarm_rgbd", family=family)
    policy = OpenArmPolicy(out, device="cpu")
    assert policy.metadata["depth_as"] == "channel" and policy.metadata["cameras"] == ["chest", "chest_depth"]
    conv = policy.policy.model.backbone.first_conv()
    assert conv.in_channels == 4
    rng = np.random.default_rng(0)
    obs = {"images": {"chest": rng.integers(0, 256, (240, 424, 3), dtype=np.uint8),
                      "chest_depth": rng.integers(0, 1500, (240, 424), dtype=np.uint16)},
           "state": rng.normal(size=16).astype(np.float32)}
    flat = dict(obs, images=dict(obs["images"], chest_depth=np.full((240, 424), 300, np.uint16)))
    import torch

    with torch.no_grad():
        conv.weight[:, 3] = 0.0
    np.testing.assert_allclose(policy.infer(obs)["actions"], policy.infer(flat)["actions"], atol=1e-5)
    with torch.no_grad():
        conv.weight[:, 3] += 1.0
    assert not np.allclose(policy.infer(obs)["actions"], policy.infer(flat)["actions"])


def test_export_then_serve_offline(tmp_path, episode_factory, monkeypatch):
    from openarm_act_lerobot.export import export
    from openarm_act_lerobot.policy import OpenArmPolicy

    out, cfg = _save_dummy_checkpoint(tmp_path, episode_factory)
    export_cfg = compose_cfg("export", [f"checkpoint={out.parents[1]}", f"paths.releases={tmp_path / 'releases'}",
                                        "dtype=bfloat16"])
    release = export(export_cfg)
    assert release.name == "act_dinov3_vitb16_dummy_smoke_000010"
    metadata = json.loads((release / "metadata.json").read_text())
    assert metadata["step"] == 10 and metadata["weights_dtype"] == "bfloat16"
    assert metadata["backbone"] == "dinov3_vitb16" and metadata["backbone_weights"] is None
    from safetensors import safe_open

    with safe_open(release / "model.safetensors", "pt") as weights:
        assert {weights.get_slice(k).get_dtype() for k in weights.keys()} == {"BF16"}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # everything from the release itself
    policy = OpenArmPolicy(release, device="cpu")
    obs = {"images": {"chest": np.zeros((240, 424, 3), dtype=np.uint8),
                      "chest_depth": np.zeros((240, 424), dtype=np.uint16)},
           "state": np.zeros(16, dtype=np.float32)}
    assert policy.infer(obs)["actions"].shape == (50, 16)
    assert OmegaConf.load(release / paths.RUN_CONFIG).name == "act_dinov3_vitb16_dummy"


@pytest.mark.parametrize("family", ["dinov3", "resnet"])
def test_training_runs_and_resumes(tmp_path, episode_factory, family):
    """LeRobot's own training loop with this repo's additions, on a CPU."""
    from safetensors.torch import load_file

    from openarm_act_lerobot.norm_stats import compute_norm_stats
    from openarm_act_lerobot.run import train

    _, overrides = _convert(tmp_path, episode_factory, "openarm_rgbd", extra=["data.video=false"])
    overrides = [*overrides, "experiment=smoke", f"backbone={FAMILIES[family]}", "finetune.backbone_trainable=none",
                 "num_train_steps=4", "save_interval=2"]
    cfg = compose_cfg(overrides=overrides)
    compute_norm_stats(cfg)
    run = train(cfg)
    step = run / "checkpoints" / "000004"
    saved = OmegaConf.load(step / "pretrained_model" / paths.RUN_CONFIG)
    assert saved.data.image_size == [240, 424] and len(saved.model.image_size) == 2
    # Two optimizer groups: ACT's, and the backbone's (its widened first layer)
    # at finetune.lr_backbone.
    groups = json.loads((step / "training_state" / "optimizer_param_groups.json").read_text())
    assert [g["initial_lr"] for g in groups] == [1e-4, 1e-5]
    first = load_file(step / "pretrained_model" / "model.safetensors")

    train(compose_cfg(overrides=[*overrides, "num_train_steps=6", "resume=true", "overwrite=false"]))
    assert (run / "checkpoints" / "000006").is_dir()
    later = load_file(run / "checkpoints" / "000006" / "pretrained_model" / "model.safetensors")
    frozen = [k for k in first if "backbone" in k and ("layer2" in k or ".layer.1." in k)]
    assert frozen and all((first[k] == later[k]).all() for k in frozen)  # the frozen backbone did not move
    assert not all((first[k] == later[k]).all() for k in first if k.startswith("model.decoder"))


# ------------------------------------------------------------------ client


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _EchoPolicy:
    """Stands in for OpenArmPolicy: remembers the observation, returns a
    chunk that runs past V1's limits."""

    def __init__(self):
        self.seen = None

    def infer(self, obs):
        self.seen = obs
        actions = np.repeat(np.asarray(obs["state"], dtype=np.float32)[None], 50, axis=0)
        actions[:, 3] = 5.0  # left_j4 beyond 2.44 rad
        return {"actions": actions}


def _serve(policy, metadata):
    from openarm_act_lerobot.server import PolicyServer

    port = _free_port()
    server = PolicyServer(policy, host="127.0.0.1", port=port, metadata=metadata)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


def _connect(port, **kwargs):
    from openarm_act_lerobot_client.policy import RemoteActPolicy

    for _ in range(50):
        try:
            return RemoteActPolicy(checkpoint=f"ws://127.0.0.1:{port}", **kwargs)
        except ConnectionError:
            time.sleep(0.1)
    raise AssertionError("the server never came up")


@pytest.mark.parametrize("data", ["openarm_rgb", "openarm_rgb_depth", "openarm_rgbd"])
def test_client_against_a_server(data):
    from openarm_act_lerobot.train_config import policy_metadata

    cfg = compose_cfg(overrides=[f"data={data}", "data.image_size=[240,424]"])
    policy = _EchoPolicy()
    client = _connect(_serve(policy, policy_metadata(cfg)))
    assert client.chunk_size == 50 and client.image_size == (240, 424)

    rgb = np.zeros((240, 424, 3), dtype=np.uint8)
    depth = np.full((240, 424), 600, dtype=np.uint16)
    qpos = np.zeros(16, dtype=np.float32)
    actions = client.predict(rgb, depth, qpos)
    assert actions.shape == (50, 16)
    assert np.all(actions[:, 3] <= 2.443461 + 1e-6)  # clipped to V1
    assert ("chest_depth" in policy.seen["images"]) == (data != "openarm_rgb")
    assert policy.seen["images"]["chest"].dtype == np.uint8 and "prompt" not in policy.seen
    assert client.server_ms is not None
    assert "RemoteActPolicy" in client.describe() and "dinov3_vitb16" in client.describe()


def test_client_sees_server_errors_and_reconnects():
    from openarm_act_lerobot.train_config import policy_metadata

    class Failing(_EchoPolicy):
        calls = 0

        def infer(self, obs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("first call fails")
            return super().infer(obs)

    cfg = compose_cfg(overrides=["data.image_size=[240,424]"])
    client = _connect(_serve(Failing(), policy_metadata(cfg)))
    args = (np.zeros((240, 424, 3), np.uint8), np.zeros((240, 424), np.uint16), np.zeros(16, np.float32))
    with pytest.raises(RuntimeError, match="first call fails"):
        client.predict(*args)
    assert client.predict(*args).shape == (50, 16)  # a new connection


def test_client_refuses_quickly_without_a_server():
    from openarm_act_lerobot_client.policy import RemoteActPolicy
    from openarm_act_lerobot_client.policy import parse_server

    assert parse_server("gpu:9000") == ("gpu", 9000)
    assert parse_server("ws://gpu") == ("gpu", 8000)
    with pytest.raises(ConnectionError):
        RemoteActPolicy(checkpoint=f"127.0.0.1:{_free_port()}", connect_timeout=1.0)


def test_msgpack_matches_openpi_encoding():
    from openarm_act_lerobot_client import msgpack_numpy

    data = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.uint16(7), "s": "text"}
    back = msgpack_numpy.unpackb(msgpack_numpy.packb(data))
    np.testing.assert_array_equal(back["a"], data["a"])
    assert back["b"] == 7 and back["b"].dtype == np.uint16 and back["s"] == "text"
    with pytest.raises(ValueError):
        msgpack_numpy.packb({"x": np.array([object()])})
