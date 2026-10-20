"""Object-centric features: scene estimation on a synthetic two-camera scene, and train/serve consistency."""

import h5py
import numpy as np
import pytest
import torch
import cv2

pytest.importorskip("zarr")

from test_meanflow import model_config
from openarm_flow.depth_views import DepthConfig
from openarm_flow.meanflow import MeanFlowPolicy
from openarm_flow.meanflow_data import MeanFlowDataset, convert_depth
from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy, save_meanflow_release
from openarm_flow.object_features import FEATURE_DIM, ObjectConfig, ObjectFeatures, SceneEstimator, _project
from openarm_flow.rgb_views import RGBConfig

TIN = (0.29, 0.21, np.radians(30))
PAD = (0.33, 0.02)


def look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return {"fx": 520.0, "fy": 520.0, "cx": 320.0, "cy": 240.0, "R_cam_from_world": np.stack([x, y, z]).tolist(),
            "C_world": list(map(float, eye))}


def config():
    return ObjectConfig(cameras={"chest": look_at(np.array([0.0, 0.0, 0.62]), [0.32, 0.0, 0.15]),
                                 "front": look_at(np.array([0.72, 0.0, 0.52]), [0.30, 0.0, 0.15])})


def render(cfg, cam, tin=TIN):
    img = np.full((480, 640, 3), (55, 66, 72), np.uint8)
    top = cfg.box_top_m

    def fill(points, color):
        uv, _ = _project(cfg.cameras[cam], points)
        cv2.fillConvexPoly(img, cv2.convexHull(np.round(uv).astype(np.int32)), color)

    px, py = PAD
    fill(np.array([[px + dx, py + dy, top] for dx in (-.082, .082) for dy in (-.057, .057)]), (245, 245, 240))
    x, y, a = tin
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    sx, sy, sz = cfg.tin_size_m
    corners = np.array([[dx, dy, dz] for dx in (-sx / 2, sx / 2) for dy in (-sy / 2, sy / 2) for dz in (0, sz)])
    fill(corners @ R.T + [x, y, top], (25, 140, 60))
    label = np.array([[dx, dy, sz + 1e-4] for dx in (-.025, .025) for dy in (-.012, .012)])
    fill(label @ R.T + [x, y, top], (240, 240, 235))
    return img


def test_scene_estimate_recovers_synthetic_poses():
    cfg = config()
    scene = SceneEstimator(cfg).estimate({c: render(cfg, c) for c in cfg.cameras})
    x, y, yaw, fit = scene["tin"]
    assert np.hypot(x - TIN[0], y - TIN[1]) < 0.006 and fit > 0.5
    assert min(abs(yaw - TIN[2]) % np.pi, np.pi - abs(yaw - TIN[2]) % np.pi) < np.radians(4)
    assert np.hypot(scene["pad"][0] - PAD[0], scene["pad"][1] - PAD[1]) < 0.006
    features = ObjectFeatures(cfg)(scene, np.zeros((3, 16), np.float32))
    assert features.shape == (3, FEATURE_DIM) and (features[:, 0] == 1).all()
    np.testing.assert_allclose(features[0, 10:12], [scene["pad"][0] - x, scene["pad"][1] - y], atol=1e-6)
    assert (ObjectFeatures(cfg)({"tin": None, "pad": scene["pad"]}, np.zeros((1, 16))) == 0).all()


def episode(path, cfg, offset=0):
    n = 5
    with h5py.File(path, "w") as f:
        f.attrs.update(complete=True, frequency_hz=50, depth_scale=.001)
        state = np.repeat((np.arange(n, dtype="f4") * 0.01 + offset)[:, None], 16, axis=1)
        state[:, [7, 15]] = 0.04
        f.create_dataset("observations/qpos", data=state)
        for c in cfg.cameras:
            f.create_dataset(f"observations/images/{c}_depth", data=np.full((n, 480, 640), 600, dtype="u2"))
            f.create_dataset(f"observations/images/{c}_rgb", data=np.stack([render(cfg, c)] * n))


def test_object_features_train_serve_consistency(tmp_path):
    cfg = config()
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5", cfg)
    episode(raw / "b.hdf5", cfg, 0.1)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    meta = convert_depth(raw, tmp_path / "d.zarr", DepthConfig(height=16, width=16), manifest=manifest,
                         action_source="next_qpos", rgb=RGBConfig(height=16, width=16), objects=cfg)
    assert meta["objects"]["features"][0] == "valid" and meta["episodes"][0]["scene"]["tin"] is not None
    data = MeanFlowDataset(tmp_path / "d.zarr", 4, 2)
    assert data[0]["obs"]["objects"].shape == (2, FEATURE_DIM)
    model_cfg = {**model_config(), "use_rgb": True, "object_features": True, "object_dropout": .5,
                 "object_noise_m": .01}
    model = MeanFlowPolicy(**model_cfg)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    loss, _ = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.objects[0].weight.grad.norm() > 0
    save_meanflow_release(tmp_path / "release", model, model_cfg, meta, 1)
    policy = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    with h5py.File(raw / "a.hdf5", "r") as f:
        obs = {"images": {k: f[f"observations/images/{k}"][0] for k in
                          (*policy.cameras, *policy.rgb_config.cameras)}, "state": f["observations/qpos"][0]}
    policy.observe(obs)
    np.testing.assert_allclose(policy.history[-1]["objects"], data[0]["obs"]["objects"][-1].numpy(), atol=1e-5)
    assert policy.infer()["actions"].shape == (3, 16)
    with pytest.raises(ValueError, match="objects"):
        model.encode({k: v for k, v in batch["obs"].items() if k != "objects"})
