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
from openarm_flow.meanflow_data import HoldingDetector, MeanFlowDataset, convert_depth, grasp_phase_flags, holding_flags
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


def test_scene_estimate_ignores_a_larger_green_stain():
    # A flat tin-green patch bigger than the tin (as the box stains look in some simulated views) used to win as
    # the "largest green blob" and the fit then searched around the stain.
    cfg = config()
    images = {}
    for cam in cfg.cameras:
        img = render(cfg, cam)
        stain = np.array([[0.29 + dx, -0.15 + dy, cfg.box_top_m] for dx in (-.07, .07) for dy in (-.05, .05)])
        uv, _ = _project(cfg.cameras[cam], stain)
        cv2.fillConvexPoly(img, cv2.convexHull(np.round(uv).astype(np.int32)), (25, 140, 60))
        images[cam] = img
    x, y, _, fit = SceneEstimator(cfg).estimate(images)["tin"]
    assert np.hypot(x - TIN[0], y - TIN[1]) < 0.006 and fit > 0.5


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


def test_convert_retries_stages_and_aborts_on_unreadable(tmp_path, monkeypatch):
    import openarm_flow.meanflow_data as md
    cfg = config()
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5", cfg)
    episode(raw / "b.hdf5", cfg, 0.1)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    calls = {"n": 0}
    real_copy = md.shutil.copyfile

    def flaky_copy(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(107, "Transport endpoint is not connected")
        return real_copy(src, dst)

    monkeypatch.setattr(md.shutil, "copyfile", flaky_copy)
    monkeypatch.setattr(md.time, "sleep", lambda s: None)
    stage = tmp_path / "stage"
    meta = convert_depth(raw, tmp_path / "ok.zarr", DepthConfig(height=16, width=16), manifest=manifest,
                         action_source="next_qpos", stage_dir=stage)
    assert len(meta["episodes"]) == 2 and calls["n"] >= 3 and not list(stage.iterdir())   # retried; staged copies removed
    episode(raw / "c.hdf5", cfg, 0.2)
    episode(raw / "d.hdf5", cfg, 0.3)
    monkeypatch.setattr(md.shutil, "copyfile", lambda s, d: (_ for _ in ()).throw(OSError(107, "Transport endpoint")))
    with pytest.raises(ValueError, match="network/mount errors"):
        convert_depth(raw, tmp_path / "bad.zarr", DepthConfig(height=16, width=16), manifest=manifest,
                      action_source="next_qpos", stage_dir=stage, skip_invalid=True)


def test_ground_truth_scene_object_targets_and_gripper_command(tmp_path):
    cfg = config()
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5", cfg)
    episode(raw / "b.hdf5", cfg, 0.1)
    truth = (TIN[0] + 0.02, TIN[1] - 0.01)   # the "simulator" knows better than the image estimate
    with h5py.File(raw / "a.hdf5", "a") as f:
        q = f["observations/qpos"]
        q[3:, 7] = q[3:, 15] = 0.01            # gripper commanded closed from frame 2 on (next_qpos labels)
        pose = np.zeros((5, 7), np.float32)
        pose[:, :2], pose[:, 2], pose[:, 3] = truth, cfg.box_top_m + 0.008, 1
        pose[4, 2] += 0.05                     # lifted at the end
        f.create_dataset("observations/object_pose/mentos_tin", data=pose)
        for c in cfg.cameras:
            f.create_dataset(f"observations/object_bbox/{c}", data=np.zeros((5, 5), "i4"))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    meta = convert_depth(raw, tmp_path / "d.zarr", DepthConfig(height=16, width=16), manifest=manifest,
                         action_source="next_qpos", rgb=RGBConfig(height=16, width=16), objects=cfg)
    plain = MeanFlowDataset(tmp_path / "d.zarr", 4, 2)
    options = dict(gripper_command=True, object_targets=True, ground_truth_objects=True)
    data = MeanFlowDataset(tmp_path / "d.zarr", 4, 2, **options)
    val = MeanFlowDataset(tmp_path / "d.zarr", 4, 2, "val", **options)
    assert np.hypot(*(plain[0]["obs"]["objects"][0, 1:3].numpy() - TIN[:2])) < 0.006   # estimated
    np.testing.assert_allclose(data[0]["obs"]["objects"][0, 1:3], truth, atol=1e-6)    # ground truth
    np.testing.assert_allclose(data.statistics()["object_mean"][1:3], truth, atol=1e-6)
    item = data[3]                              # observations at frames 2 and 3
    assert item["obs"]["gripper_command"].tolist() == [[0, 0], [1, 1]]
    assert data[0]["obs"]["gripper_command"].tolist() == [[0, 0], [0, 0]]
    assert item["object_target_valid"] and item["object_target"][2] == pytest.approx(cfg.box_top_m + 0.008)
    tcp = ObjectFeatures(cfg).tcp(item["obs"]["agent_pos"][-1:].numpy())[0]
    np.testing.assert_allclose(item["object_target"][3:].reshape(2, 3), pose[3, :3] - tcp, atol=1e-5)
    assert not val[0]["object_target_valid"] and (val[0]["object_target"] == 0).all()

    model_cfg = {**model_config(), "use_rgb": True, "object_features": True, "gripper_state_input": False,
                 "gripper_command_input": True, "object_aux_weight": 0.1}
    with pytest.raises(ValueError, match="gripper_state_input"):
        MeanFlowPolicy(**{**model_cfg, "gripper_state_input": True})
    model = MeanFlowPolicy(**model_cfg)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=4)))
    loss, parts = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and parts["object_aux_loss"] > 0 and model.object_head[0].weight.grad.norm() > 0
    model.eval()
    assert model.predict_object(batch["obs"]).shape == (4, 9)
    flipped = {**batch["obs"], "gripper_command": 1 - batch["obs"]["gripper_command"]}
    assert not torch.allclose(model.encode(batch["obs"]), model.encode(flipped))
    with pytest.raises(ValueError, match="gripper_command"):
        model.encode({k: v for k, v in batch["obs"].items() if k != "gripper_command"})

    save_meanflow_release(tmp_path / "release", model, model_cfg, meta, 1, options)
    policy = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    assert policy.metadata["data_options"] == options
    with h5py.File(raw / "a.hdf5", "r") as f:
        obs = {"images": {k: f[f"observations/images/{k}"][3] for k in (*policy.cameras, *policy.rgb_config.cameras)},
               "state": f["observations/qpos"][3]}
    with pytest.raises(ValueError, match="command"):
        policy.observe(obs)
    policy.observe({**obs, "command": np.full(16, 0.01, np.float32)})
    assert policy.history[-1]["gripper_command"].tolist() == [1, 1]
    assert policy.infer()["actions"].shape == (3, 16)


def test_holding_detector_is_off_until_the_fingers_settle_on_something():
    fps = 50
    open_, closing, held = [0.043] * 20, list(np.linspace(0.043, 0.024, 10)), [0.024] * 30
    released = list(np.linspace(0.024, 0.043, 10)) + [0.043] * 5
    left = np.array(open_ + closing + held + released)
    right = np.full(len(left), 0.002)                      # closed on air: never "holding"
    states = np.zeros((len(left), 16))
    states[:, 7], states[:, 15] = left, right
    flags = holding_flags(states, fps)
    assert not flags[:, 1].any()
    assert not flags[:30, 0].any()                         # open and closing: the close decision never sees it
    on = np.flatnonzero(flags[:, 0])
    assert 30 + round(0.3 * fps) - 4 <= on[0] <= 30 + round(0.3 * fps) - 1   # once it has stayed put for 0.3 s
    assert flags[on[0]:63, 0].all() and not flags[63:, 0].any()   # off once the opening leaves the range (> 30 mm)
    detector = HoldingDetector(fps)                        # incremental use (deployment) gives the same flags
    np.testing.assert_array_equal(np.stack([detector(s[[7, 15]]) for s in states]), flags)
    actions = states.copy()
    actions[30:60, 7] = 0.0                                # closed command, then released at 60
    _, phase = grasp_phase_flags(actions, fps)
    _, both = grasp_phase_flags(actions, fps, release=True)
    assert not phase[61:, 0].any() and both[10:85, 0].all()


def test_holding_input_dataset_model_and_policy(tmp_path):
    cfg = config()
    raw = tmp_path / "raw"
    raw.mkdir()
    episode(raw / "a.hdf5", cfg)
    episode(raw / "b.hdf5", cfg, 0.1)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    meta = convert_depth(raw, tmp_path / "d.zarr", DepthConfig(height=16, width=16), manifest=manifest,
                         action_source="next_qpos", rgb=RGBConfig(height=16, width=16), objects=cfg)
    options = dict(holding=True, release_phase=True)
    data = MeanFlowDataset(tmp_path / "d.zarr", 4, 2, **options)
    assert data[3]["obs"]["holding"].shape == (2, 2)
    model_cfg = {**model_config(), "use_rgb": True, "object_features": True, "gripper_state_input": False,
                 "holding_input": True}
    with pytest.raises(ValueError, match="holding_input"):
        MeanFlowPolicy(**{**model_cfg, "gripper_command_input": True})
    model = MeanFlowPolicy(**model_cfg)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=4)))
    loss, _ = model.compute_loss(batch)
    assert torch.isfinite(loss)
    model.eval()
    flipped = {**batch["obs"], "holding": 1 - batch["obs"]["holding"]}
    assert not torch.allclose(model.encode(batch["obs"]), model.encode(flipped))
    with pytest.raises(ValueError, match="holding"):
        model.encode({k: v for k, v in batch["obs"].items() if k != "holding"})
    save_meanflow_release(tmp_path / "release", model, model_cfg, meta, 1, options)
    policy = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    with h5py.File(raw / "a.hdf5", "r") as f:
        obs = {"images": {k: f[f"observations/images/{k}"][3] for k in (*policy.cameras, *policy.rgb_config.cameras)},
               "state": f["observations/qpos"][3].copy()}
    obs["state"][7] = 0.024
    for _ in range(round(0.3 * 50)):
        policy.observe(obs)
    assert policy.history[-1]["holding"].tolist() == [1, 0]
    assert policy.infer()["actions"].shape == (3, 16)
    policy.reset()
    policy.observe(obs)
    assert policy.history[-1]["holding"].tolist() == [0, 0]
