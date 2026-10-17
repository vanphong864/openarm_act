"""Geometry, episode alignment and upstream FlowPolicy integration contracts."""

from dataclasses import asdict
import json

import h5py
import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("zarr")

from openarm_flow.data import FlowDataset, convert
from openarm_flow.pointcloud import Camera, PointCloudConfig, project_depth
from openarm_flow.policy import build_model, save_release, OpenArmFlowPolicy


def test_pinhole_geometry_mask_and_color():
    camera = Camera(3, 2, 2, 2, 1, 0)
    depth = np.array([[0, 500, 1200], [500, 500, 500]], dtype=np.uint16)
    cloud = PointCloudConfig(num_points=4, min_valid=1, use_color=True)
    rgb = np.full((2, 3, 3), 255, dtype=np.uint8)
    points = project_depth(depth, camera, cloud, np.random.default_rng(0), rgb)
    expected = {(0., 0., .5), (-.25, .25, .5), (0., .25, .5), (.25, .25, .5)}
    assert {tuple(p) for p in points[:, :3]} == expected
    np.testing.assert_array_equal(points[:, 3:], 1)
    with pytest.raises(ValueError, match="valid points"):
        project_depth(np.zeros((2, 3)), camera, cloud, np.random.default_rng(0), rgb)


def episode(path, offset):
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, frequency_hz=50, depth_scale=.001)
        state = np.repeat((np.arange(6, dtype="f4") + offset)[:, None], 16, axis=1)
        f.create_dataset("observations/qpos", data=state)
        f.create_dataset("observations/images/chest_depth", data=np.full((6, 2, 3), 500, dtype="u2"))


def test_conversion_sequence_and_train_only_stats(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5", 0)
    episode(raw / "b.hdf5", 100)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    output = tmp_path / "data.zarr"
    convert(raw, output, Camera(3, 2, 2, 2, 1, 0), PointCloudConfig(num_points=8, min_valid=1),
            manifest=manifest, action_source="next_qpos")
    train = FlowDataset(output, horizon=4, n_obs_steps=2)
    assert len(train) == 5
    first = train[0]
    assert first["obs"]["agent_pos"][:, 0].tolist() == [0, 0]
    assert first["action"][:, 0].tolist() == [1, 1, 2, 3]
    assert train[4]["action"][:, 0].tolist() == [4, 5, 5, 5]
    norm = train.normalizer()
    assert norm.get_input_stats()["agent_pos"]["max"][0] == 4  # val starts at 100
    assert len(FlowDataset(output, 4, 2, "val")) == 5
    with pytest.raises(FileExistsError):
        convert(raw, output, Camera(3, 2, 2, 2, 1, 0), PointCloudConfig())


def test_upstream_loss_update_and_release_roundtrip(tmp_path):
    torch.set_num_threads(2)
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parents[1] / "conf/flowpolicy.yaml").read_text())["model"]
    cfg.update(down_dims=[16, 32], horizon=4, n_action_steps=3)
    metadata = {"camera": asdict(Camera(3, 2, 2, 2, 1, 0)),
                "pointcloud": asdict(PointCloudConfig(num_points=8, min_valid=1)), "seed": 42, "fps": 50}
    model = build_model(cfg, metadata)
    from openarm_flow._vendor.model.common.normalizer import LinearNormalizer
    norm = LinearNormalizer()
    norm.fit({"point_cloud": np.random.default_rng(1).normal(size=(4, 8, 3)).astype("f4"),
              "agent_pos": np.random.default_rng(2).normal(size=(4, 16)).astype("f4"),
              "action": np.random.default_rng(3).normal(size=(4, 16)).astype("f4")})
    model.set_normalizer(norm)
    batch = {"obs": {"point_cloud": torch.randn(2, 2, 8, 3), "agent_pos": torch.randn(2, 2, 16)},
             "action": torch.randn(2, 4, 16)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    parameter = model.model.final_conv[-1].weight
    before = parameter.detach().clone()
    loss, _ = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, parameter)
    assert model.predict_action(batch["obs"])["action"].shape == (2, 3, 16)
    save_release(tmp_path / "release", model, cfg, metadata, 1)
    loaded = OpenArmFlowPolicy(tmp_path / "release", "cpu")
    torch.manual_seed(123)
    expected = model.eval().predict_action(batch["obs"])["action"].detach()
    torch.manual_seed(123)
    actual = loaded.policy.predict_action(batch["obs"])["action"].detach()
    torch.testing.assert_close(actual, expected)
    result = loaded.infer({"images": {"chest_depth": np.full((2, 3), 500, dtype="u2")},
                           "state": np.zeros(16, dtype="f4")})
    assert result["actions"].shape == (3, 16)


def test_horizon_validation():
    with pytest.raises(ValueError, match="Horizon"):
        build_model({"horizon": 4, "n_obs_steps": 2, "n_action_steps": 4, "down_dims": [16, 32]}, {})
