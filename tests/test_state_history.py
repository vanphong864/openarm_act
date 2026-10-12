"""The last n joint states as input (model.state_steps): dataset, model,
serving, offline evaluation and the robot's client."""

import numpy as np
import pytest
import torch

from openarm_act_lerobot import constants as C
from test_pipeline import _connect
from test_pipeline import _convert
from test_pipeline import _EchoPolicy
from test_pipeline import _serve
from test_pipeline import compose_cfg


def test_state_history_padding():
    from openarm_act_lerobot.evaluate import state_history

    states = np.arange(5)[:, None] * np.ones((1, 16))
    np.testing.assert_array_equal(state_history(states, 3, 1), states[3])
    np.testing.assert_array_equal(state_history(states, 1, 4)[:, 0], [0, 0, 0, 1])  # the first repeats, oldest first
    np.testing.assert_array_equal(state_history(states, 4, 3)[:, 0], [2, 3, 4])


def test_dataset_yields_the_state_history_only(tmp_path, episode_factory):
    """Training reads the last n states at the dataset's rate, padded with the
    first at an episode's start, and still one frame per camera."""
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.scripts import lerobot_train  # noqa: F401  -- what openarm_training patches

    from openarm_act_lerobot.episode import EpisodeReader
    from openarm_act_lerobot.run import openarm_training
    from openarm_act_lerobot.train_config import build_policy_config

    _, overrides = _convert(tmp_path, episode_factory, "openarm_rgbd", extra=["data.video=false"])
    cfg = compose_cfg(overrides=[*overrides, "experiment=smoke", "model.state_steps=4"])
    policy_cfg = build_policy_config(cfg)
    assert policy_cfg.n_state_steps == 4
    meta = LeRobotDatasetMetadata(str(cfg.data.repo_id))
    with openarm_training(cfg, None):
        from lerobot.datasets import factory

        timestamps = factory.resolve_delta_timestamps(policy_cfg, meta)
    assert timestamps["observation.state"] == [-0.06, -0.04, -0.02, 0.0]
    assert "observation.images.chest" not in timestamps
    dataset = LeRobotDataset(str(cfg.data.repo_id), delta_timestamps=timestamps)
    with EpisodeReader(tmp_path / "raw/2026-10-08/episode_a.hdf5") as ep:
        states = ep.labels("qpos").state
    for t, expected in ((10, states[7:11]), (1, states[[0, 0, 0, 1]])):
        sample = dataset[t]
        np.testing.assert_allclose(np.asarray(sample["observation.state"]), expected, atol=1e-6)
        assert sample["observation.images.chest"].shape[0] == 3  # one frame


def test_model_reads_the_flattened_history():
    from test_depth import make_config

    from openarm_act_lerobot.modeling_openarm_act import OpenArmACTPolicy

    policy = OpenArmACTPolicy(make_config("resnet", depth_as="channel", depth_format="gray8", n_state_steps=10))
    assert policy.model.encoder_robot_state_input_proj.in_features == 160
    assert policy.model.vae_encoder_robot_state_input_proj.in_features == 160
    rng = torch.Generator().manual_seed(0)
    batch = {"observation.images.chest": torch.rand(2, 3, 60, 80, generator=rng),
             "observation.images.chest_depth": torch.rand(2, 1, 60, 80, generator=rng),
             "observation.state": torch.randn(2, 10, 16, generator=rng)}
    policy.train()
    loss, _ = policy.forward(batch | {"action": torch.randn(2, 10, 16), "action_is_pad": torch.zeros(2, 10, dtype=bool)})
    loss.backward()
    assert policy.model.encoder_robot_state_input_proj.weight.grad[:, :16].abs().sum() > 0  # the oldest state matters
    policy.eval()
    a = policy.predict_action_chunk(batch)
    moved = dict(batch, **{"observation.state": batch["observation.state"].clone()})
    moved["observation.state"][:, 0] += 1.0  # only the oldest state differs
    assert not torch.allclose(a, policy.predict_action_chunk(moved))
    with pytest.raises(ValueError, match="last 10 joint states"):
        policy.predict_action_chunk(dict(batch, **{"observation.state": torch.randn(2, 16)}))


def test_train_serve_and_evaluate_with_history(tmp_path, episode_factory):
    from openarm_act_lerobot.evaluate import evaluate
    from openarm_act_lerobot.norm_stats import compute_norm_stats
    from openarm_act_lerobot.policy import OpenArmPolicy
    from openarm_act_lerobot.run import train

    _, overrides = _convert(tmp_path, episode_factory, "openarm_rgbd",
                            extra=["data.video=false", "data.val_fraction=0.5"])
    cfg = compose_cfg(overrides=[*overrides, "experiment=smoke", "model.state_steps=10", "data.delta_actions=true",
                                 "num_train_steps=4", "save_interval=4"])
    compute_norm_stats(cfg)
    run = train(cfg)
    policy = OpenArmPolicy(run, device="cpu")
    assert policy.state_steps == 10 and policy.metadata["state_steps"] == 10
    rng = np.random.default_rng(0)
    obs = {"images": {"chest": rng.integers(0, 256, (240, 424, 3), dtype=np.uint8),
                      "chest_depth": rng.integers(0, 1500, (240, 424), dtype=np.uint16)},
           "state": rng.normal(size=(10, 16)).astype(np.float32)}
    actions = policy.infer(obs)["actions"]
    assert actions.shape == (50, 16) and np.isfinite(actions).all()
    with pytest.raises(ValueError, match="last 10 joint states"):
        policy.infer(dict(obs, state=obs["state"][-1]))
    result = evaluate(compose_cfg("eval", [f"checkpoint={run}", "every=20", f"paths.raw={tmp_path / 'raw'}"]))
    assert result["episodes"] >= 1 and np.isfinite(result["model"]["arm_rmse_rad"])


def test_client_sends_the_history():
    from openarm_act_lerobot.train_config import policy_metadata

    cfg = compose_cfg(overrides=["data.image_size=[240,424]", "model.state_steps=10"])
    policy = _EchoPolicy()
    policy.infer = lambda obs: (setattr(policy, "seen", obs) or
                                {"actions": np.zeros((50, C.NUM_JOINTS), np.float32)})
    client = _connect(_serve(policy, policy_metadata(cfg)))
    assert client.state_steps == 10 and client.fps == 50 and "last 10 joint states" in client.describe()
    rgb, depth = np.zeros((240, 424, 3), np.uint8), np.zeros((240, 424), np.uint16)
    history = np.arange(160, dtype=np.float32).reshape(10, 16)
    client.predict(rgb, depth, history)
    np.testing.assert_array_equal(policy.seen["state"], history)
    with pytest.raises(ValueError, match="last 10 joint states"):
        client.predict(rgb, depth, history[-1])

    one = _connect(_serve(_EchoPolicy(), policy_metadata(compose_cfg(overrides=["data.image_size=[240,424]"]))))
    assert one.state_steps == 1
    assert one.predict(rgb, depth, history).shape == (50, 16)  # a history to a one-state model: its newest
