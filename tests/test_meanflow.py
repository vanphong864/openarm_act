"""MeanFlow identity, dual-depth contracts, episode leakage and release integration."""

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("zarr")

from openarm_flow.depth_views import DepthConfig, encode_depth, resize_depth
from openarm_flow.meanflow import MeanFlowPolicy, dispersive_loss, meanflow_target
from openarm_flow.meanflow_data import MeanFlowDataset, convert_depth
from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy, save_meanflow_release


def episode(path, offset=0):
    n = 6
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, frequency_hz=50, depth_scale=.001)
        state = np.repeat((np.arange(n, dtype="f4") + offset)[:, None], 16, axis=1)
        state[:, [7, 15]] *= 0.001
        f.create_dataset("observations/qpos", data=state)
        for c, d in (("chest_depth", 500), ("front_depth", 800)):
            f.create_dataset(f"observations/images/{c}", data=np.full((n, 32, 32), d, dtype="u2"))


@pytest.fixture
def converted(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5")
    episode(raw / "b.hdf5", 100)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    out = tmp_path / "depth.zarr"
    metadata = convert_depth(raw, out, DepthConfig(height=16, width=16), manifest=manifest,
                             action_source="next_qpos")
    return out, metadata, raw


def model_config():
    return dict(horizon=4, n_obs_steps=2, n_action_steps=3, down_dims=[16, 32], time_dim=16,
                feature_dim=16, state_dim=16, pixel_dropout=0, view_dropout=0, depth_noise_std=0)


def test_masked_depth_pooling_and_camera_order():
    cfg = DepthConfig(height=16, width=16)
    raw = np.full((32, 32), 500, dtype="u2")
    raw[::2, ::2] = 0
    reduced = resize_depth(raw, cfg)
    np.testing.assert_array_equal(reduced, 500)
    raw[:2, :2] = 1200  # clipped far plane is invalid
    reduced = resize_depth(raw, cfg)
    assert reduced[0, 0] == 0
    encoded = encode_depth(np.stack([reduced, reduced]), cfg)
    assert encoded.shape == (2, 2, 16, 16)
    assert (encoded[:, :, 0, 0] == 0).all()
    with pytest.raises(ValueError, match="Too few"):
        resize_depth(np.zeros((32, 32), dtype="u2"), cfg)
    with pytest.raises(ValueError, match="uint16"):
        resize_depth(raw.astype("f4"), cfg)


def test_meanflow_jvp_identity_and_stop_gradient():
    # Analytic field u=2*z+3*t+5*r+c: total derivative = 2*v+3, r is FIXED.
    def velocity(z, t, r, c):
        return 2*z + 3*t[:, None, None] + 5*r[:, None, None] + c[:, None, :]
    z = torch.randn(2, 4, 16, requires_grad=True)
    v = torch.randn_like(z)
    t, r = torch.tensor([.8, .6]), torch.tensor([.2, .6])
    cond = torch.randn(2, 16, requires_grad=True)
    actual = meanflow_target(velocity, z, t, r, v, cond)
    torch.testing.assert_close(actual, v - (t-r)[:, None, None] * (2*v+3))
    torch.testing.assert_close(actual[1], v[1])
    assert not actual.requires_grad


def test_episode_boundaries_masks_and_train_only_stats(converted):
    path, meta, _ = converted
    train, val = MeanFlowDataset(path, 4, 2), MeanFlowDataset(path, 4, 2, "val")
    assert len(train) == len(val) == 5
    first, last = train[0], train[4]
    assert first["obs"]["agent_pos"][:, 0].tolist() == [0, 0]
    assert first["action"][:, 0].tolist() == [1, 2, 3, 4]  # current target, no past prefix
    assert last["action"][:, 0].tolist() == [5, 5, 5, 5]
    assert last["action_valid"].tolist() == [True, False, False, False]
    assert first["obs"]["depth"].shape == (2, 2, 2, 16, 16)
    assert first["obs"]["depth"][0, 0, 0].mean() < first["obs"]["depth"][0, 1, 0].mean()
    statistics = train.statistics()
    assert statistics["state_mean"][0] == 2
    # All valid offsets (5+4+3+2)=14 targets: residual sum (5+8+9+8)=30.
    assert statistics["action_mean"][0] == pytest.approx(30 / 14)
    assert np.isfinite(statistics["action_scale"]).all()
    with pytest.raises(ValueError, match="training split"):
        val.statistics()
    assert val.validation_indices(2) == [0, 4]


def test_several_zarrs_read_as_one_dataset(converted, tmp_path):
    path, meta, raw = converted
    other = tmp_path / "other.zarr"   # same episodes again, with the states offset by 1000
    episode(raw / "a.hdf5", 1000)
    convert_depth(raw, other, DepthConfig(height=16, width=16), manifest=tmp_path / "manifest.yaml",
                  action_source="next_qpos")
    single, both = MeanFlowDataset(path, 4, 2), MeanFlowDataset([path, other], 4, 2)
    assert single.metadata == meta and "sources" not in single.metadata
    assert len(both) == 2 * len(single) and len(both.metadata["episodes"]) == 4
    assert both[0]["obs"]["agent_pos"][0, 0] == 0 and both[5]["obs"]["agent_pos"][0, 0] == 1000
    assert both[9]["action_valid"].tolist() == [True, False, False, False]
    assert both.statistics()["state_mean"][0] == pytest.approx(502)
    assert MeanFlowDataset([path, other], 4, 2, "val").validation_indices(2) == [0, 4, 5, 9]
    convert_depth(raw, tmp_path / "small.zarr", DepthConfig(height=16, width=24), manifest=tmp_path / "manifest.yaml",
                  action_source="next_qpos")
    with pytest.raises(ValueError, match="preprocessing differs"):
        MeanFlowDataset([path, tmp_path / "small.zarr"], 4, 2)


def test_corrupt_episode_is_explicitly_reported(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5")
    episode(raw / "b.hdf5")
    (raw / "corrupt.hdf5").write_bytes(b"truncated HDF5")
    with pytest.raises(ValueError, match="corrupt"):
        convert_depth(raw, tmp_path / "strict.zarr", DepthConfig(height=16, width=16))
    meta = convert_depth(raw, tmp_path / "skip.zarr", DepthConfig(height=16, width=16), skip_invalid=True,
                         max_skip_fraction=0.5)  # corrupt files are skipped regardless
    assert meta["complete"] and len(meta["episodes"]) == 2
    assert meta["skipped"][0]["id"] == "corrupt"


def test_backward_camera_sensitivity_single_nfe_and_release(converted, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    path, metadata, raw = converted
    data = MeanFlowDataset(path, 4, 2)
    model = MeanFlowPolicy(**model_config())
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.velocity.output.weight.detach().clone()
    loss, _ = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.cnn[0].weight.grad.norm() > 0
    assert model.velocity.time_r[1].weight.grad.norm() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.velocity.output.weight)
    model.eval()
    obs = {k: v.clone().requires_grad_(True) for k, v in batch["obs"].items()}
    model.encode(obs).square().mean().backward()
    assert obs["depth"].grad[:, :, 0].abs().sum() > 0
    assert obs["depth"].grad[:, :, 1].abs().sum() > 0
    calls = []
    handle = model.velocity.register_forward_hook(lambda *args: calls.append(1))
    noise = torch.randn(2, 4, 16)
    prediction = model.predict_action(batch["obs"], noise=noise)
    handle.remove()
    assert len(calls) == 1 and prediction["action"].shape == (2, 3, 16)
    targets = batch["action"]
    qpos = batch["obs"]["agent_pos"][:, -1]
    torch.testing.assert_close(model.unnormalize_actions(model.normalize_actions(targets, qpos), qpos), targets)
    save_meanflow_release(tmp_path / "release", model, model_config(), metadata, 1)
    loaded = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    torch.testing.assert_close(loaded.policy.predict_action(batch["obs"], noise=noise)["action"], prediction["action"])
    with h5py.File(raw / "a.hdf5", "r") as f:
        observation = {"images": {c: f[f"observations/images/{c}"][0] for c in loaded.cameras},
                       "state": f["observations/qpos"][0]}
    loaded.observe(observation)
    np.testing.assert_array_equal(loaded.history[-1]["depth"], data[0]["obs"]["depth"][-1].numpy())
    expected = loaded.infer()["actions"]
    loaded.reset()
    np.testing.assert_array_equal(loaded.infer(observation)["actions"], expected)
    with torch.no_grad():
        assert torch.isfinite(loaded.policy.compute_loss(batch)[0])


def test_dispersion_collapsed_and_singleton_are_finite():
    for b in (1, 3):
        feature = torch.zeros(b, 12, requires_grad=True)
        loss = dispersive_loss((feature,))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(feature.grad).all()


def test_padding_is_excluded_from_loss():
    # Pointwise field isolates loss masking from the U-Net's temporal coupling.
    class Field(torch.nn.Module):
        def forward(self, z, t, r, cond, return_features=False):
            velocity = z * 0
            return (velocity, (cond,)) if return_features else velocity
    p = MeanFlowPolicy(**model_config(), dispersive_weight=0)
    p.set_statistics({k: np.ones(16, "f4") if "scale" in k else np.zeros(16, "f4")
                      for k in ("state_mean", "state_scale", "action_mean", "action_scale")})
    p.velocity = Field()
    p.eval()
    batch = {"obs": {"depth": torch.ones(2, 2, 2, 2, 16, 16), "agent_pos": torch.zeros(2, 2, 16)},
             "action": torch.zeros(2, 4, 16), "action_valid": torch.tensor([[1, 0, 0, 0]] * 2)}
    torch.manual_seed(1)
    first = p.compute_loss(batch)[0]
    batch["action"][:, 1:] = 1000
    torch.manual_seed(1)
    torch.testing.assert_close(p.compute_loss(batch)[0], first)


def test_train_resume_and_evaluate_cli(converted, tmp_path):
    root = Path(__file__).parents[1]
    dataset, _, _ = converted
    cfg = yaml.safe_load((root / "conf/meanflow.yaml").read_text())
    cfg["model"] = model_config()
    cfg["train"].update(steps=1, batch_size=2, val_samples_per_episode=2, save_every=1)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "run"
    command = [sys.executable, str(root / "scripts/train_meanflow.py"), "--dataset", str(dataset),
               "--output", str(output), "--config", str(config), "--device", "cpu"]
    subprocess.run(command, check=True)
    subprocess.run(command + ["--resume", "--steps", "2"], check=True)
    state = torch.load(output / "last.pt", weights_only=True, map_location="cpu")
    assert state["step"] == 2 and np.isfinite(state["best_score"])
    subprocess.run([sys.executable, str(root / "scripts/eval_meanflow.py"), "--dataset", str(dataset),
                    "--checkpoint", str(output / "best_release"), "--device", "cpu", "--samples-per-episode", "2"],
                   check=True)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2 and metrics[-1]["val_chunks"] == 2
