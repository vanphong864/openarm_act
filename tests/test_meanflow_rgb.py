"""RGB branch: conversion, dataset batches, gradients, release round trip and misuse errors."""

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("zarr")

from test_meanflow import model_config
from openarm_flow.meanflow import MeanFlowPolicy
from openarm_flow.meanflow_data import MeanFlowDataset, convert_depth
from openarm_flow.depth_views import DepthConfig
from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy, save_meanflow_release
from openarm_flow.rgb_views import RGBConfig, resize_rgb


def episode(path, offset=0):
    n = 6
    rng = np.random.default_rng(offset)
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, frequency_hz=50, depth_scale=.001)
        state = np.repeat((np.arange(n, dtype="f4") + offset)[:, None], 16, axis=1)
        state[:, [7, 15]] *= 0.001
        f.create_dataset("observations/qpos", data=state)
        for c, d in (("chest", 500), ("front", 800)):
            f.create_dataset(f"observations/images/{c}_depth", data=np.full((n, 32, 32), d, dtype="u2"))
            f.create_dataset(f"observations/images/{c}_rgb", data=rng.integers(0, 256, (n, 32, 32, 3), dtype="u1"))


@pytest.fixture
def rgb_converted(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5")
    episode(raw / "b.hdf5", 100)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    out = tmp_path / "rgb.zarr"
    metadata = convert_depth(raw, out, DepthConfig(height=16, width=16), manifest=manifest, action_source="next_qpos",
                             rgb=RGBConfig(height=16, width=16))
    return out, metadata, raw


def test_resize_rgb_block_mean_and_validation():
    cfg = RGBConfig(height=16, width=16)
    raw = np.zeros((32, 32, 3), "u1")
    raw[:2, :2] = (100, 150, 200)
    out = resize_rgb(raw, cfg)
    assert out.shape == (3, 16, 16) and out.dtype == np.uint8
    assert out[:, 0, 0].tolist() == [100, 150, 200] and out[:, 1, 1].tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        resize_rgb(raw.astype("f4"), cfg)


def test_dataset_yields_uint8_rgb(rgb_converted):
    path, metadata, _ = rgb_converted
    assert metadata["rgb"]["cameras"] == ["chest_rgb", "front_rgb"]
    data = MeanFlowDataset(path, 4, 2)
    item = data[0]
    assert item["obs"]["rgb"].shape == (2, 2, 3, 16, 16) and item["obs"]["rgb"].dtype == torch.uint8
    assert item["obs"]["depth"].shape == (2, 2, 2, 16, 16)


def test_rgb_model_trains_and_uses_rgb(rgb_converted, tmp_path):
    torch.manual_seed(0)
    path, metadata, raw = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    config = {**model_config(), "use_rgb": True}
    model = MeanFlowPolicy(**config)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    loss, _ = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.rgb_cnn[0].weight.grad.norm() > 0
    model.eval()
    changed = {**batch["obs"], "rgb": 255 - batch["obs"]["rgb"]}
    assert not torch.allclose(model.encode(batch["obs"]), model.encode(changed))
    with pytest.raises(ValueError, match="rgb"):
        model.encode({k: v for k, v in batch["obs"].items() if k != "rgb"})
    noise = torch.randn(2, 4, 16)
    prediction = model.predict_action(batch["obs"], noise=noise)
    save_meanflow_release(tmp_path / "release", model, config, metadata, 1)
    loaded = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    torch.testing.assert_close(loaded.policy.predict_action(batch["obs"], noise=noise)["action"], prediction["action"])
    with h5py.File(raw / "a.hdf5", "r") as f:
        images = {c: f[f"observations/images/{c}"][0] for c in (*loaded.cameras, *loaded.rgb_config.cameras)}
        observation = {"images": images, "state": f["observations/qpos"][0]}
    loaded.observe(observation)
    np.testing.assert_array_equal(loaded.history[-1]["rgb"], data[0]["obs"]["rgb"][-1].numpy())
    assert loaded.infer()["actions"].shape == (3, 16)


def test_depth_only_model_rejects_rgb_mismatch(rgb_converted):
    path, _, _ = rgb_converted
    batch = next(iter(torch.utils.data.DataLoader(MeanFlowDataset(path, 4, 2), batch_size=2)))
    model = MeanFlowPolicy(**model_config())  # use_rgb=False ignores the extra key
    model.set_statistics(MeanFlowDataset(path, 4, 2).statistics())
    assert torch.isfinite(model.compute_loss(batch)[0])


def test_spatial_softmax_localizes_and_trains(rgb_converted, tmp_path):
    from openarm_flow.meanflow import SpatialSoftmaxHead
    head = SpatialSoftmaxHead(4, 3, 8)
    with torch.no_grad():
        head.keypoints.weight.zero_()
        head.keypoints.weight[:, 0] = 20.0  # every keypoint follows channel 0
        head.log_temperature.fill_(0.0)
    x = torch.zeros(2, 4, 12, 16)
    x[0, 0, 3, 4] = 1.0
    x[1, 0, 9, 13] = 1.0
    weights = head.keypoints(x).flatten(2).softmax(-1).view(2, 3, 12, 16)
    ys, xs = torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 16)
    px = (weights.sum(2) * xs).sum(-1)[:, 0]
    py = (weights.sum(3) * ys).sum(-1)[:, 0]
    assert px[0] < -0.3 and px[1] > 0.5 and py[0] < -0.3 and py[1] > 0.5

    path, metadata, _ = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    config = {**model_config(), "use_rgb": True, "spatial_softmax": True, "keypoints": 8,
              "rgb_channel_gain_jitter": .1, "state_noise_std": .01}
    model = MeanFlowPolicy(**config)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    loss, _ = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.rgb_cnn[-1].keypoints.weight.grad.norm() > 0
    model.eval()
    first = model.encode(batch["obs"])
    torch.testing.assert_close(first, model.encode(batch["obs"]))  # augmentation is train-only
    save_meanflow_release(tmp_path / "release", model, config, metadata, 1)
    loaded = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    noise = torch.randn(2, 4, 16)
    torch.testing.assert_close(loaded.policy.predict_action(batch["obs"], noise=noise)["action"],
                               model.predict_action(batch["obs"], noise=noise)["action"])
