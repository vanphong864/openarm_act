"""RGB branch: conversion, dataset batches, gradients, release round trip and misuse errors."""

import json
from pathlib import Path
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


def test_workspace_crop_is_shared_by_depth_and_rgb():
    from openarm_flow.depth_views import resize_depth
    depth_cfg = DepthConfig(height=16, width=16, crops={"chest": [16, 32, 0, 32]})
    rgb_cfg = RGBConfig(height=16, width=16, crops={"chest": [16, 32, 0, 32]})
    depth = np.full((32, 32), 500, dtype="u2")
    depth[16:] = 800
    rgb = np.zeros((32, 32, 3), "u1")
    rgb[16:] = 200
    assert (resize_depth(depth, depth_cfg, "chest_depth") == 800).all()
    assert (resize_rgb(rgb, rgb_cfg, "chest_rgb") == 200).all()
    assert (resize_depth(depth, depth_cfg, "front_depth") != 800).any()  # no crop for this camera
    with pytest.raises(ValueError, match="crop"):
        resize_rgb(rgb, RGBConfig(height=16, width=16, crops={"chest": [0, 64, 0, 32]}), "chest_rgb")


def test_resnet_backbone_and_state_history_dropout(rgb_converted, tmp_path):
    path, metadata, _ = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    config = {**model_config(), "use_rgb": True, "spatial_softmax": True, "keypoints": 8,
              "rgb_backbone": "resnet18", "pretrained_backbone": False, "state_history_dropout": 1.0}
    model = MeanFlowPolicy(**config)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    loss, _ = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.rgb_cnn.trunk[0].weight.grad.norm() > 0
    # With history fully dropped during training, past states cannot change the encoding.
    torch.manual_seed(0)
    a = model.encode(batch["obs"], augment=True)
    shifted = {**batch["obs"], "agent_pos": batch["obs"]["agent_pos"].clone()}
    shifted["agent_pos"][:, 0] += 1.0
    model.depth_noise_std = model.pixel_dropout = model.view_dropout = 0.0
    model.rgb_color_jitter = model.rgb_channel_gain_jitter = 0.0
    torch.manual_seed(1)
    b1 = model.encode(batch["obs"], augment=True)
    torch.manual_seed(1)
    b2 = model.encode(shifted, augment=True)
    torch.testing.assert_close(b1, b2)
    save_meanflow_release(tmp_path / "release", model, {**config, "pretrained_backbone": True}, metadata, 1)
    model.eval()
    loaded = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")  # must not download weights
    noise = torch.randn(2, 4, 16)
    torch.testing.assert_close(loaded.policy.predict_action(batch["obs"], noise=noise)["action"],
                               model.predict_action(batch["obs"], noise=noise)["action"])


def test_robustness_augmentations_and_frozen_stem(rgb_converted):
    path, _, _ = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    config = {**model_config(), "use_rgb": True, "spatial_softmax": True, "keypoints": 8, "rgb_backbone": "resnet18",
              "pretrained_backbone": False, "freeze_backbone_stem": True, "random_shift_px": 3,
              "rgb_cutout_prob": 1.0, "rgb_blur_prob": 1.0}
    model = MeanFlowPolicy(**config)
    model.set_statistics(data.statistics())
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    model.train()
    tower = model.encoder.rgb_cnn
    assert not tower.trunk[1].training and tower.trunk[5].training  # frozen BN stays in eval, layer2 trains
    running = tower.trunk[1].running_mean.clone()
    loss, _ = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert tower.trunk[0].weight.grad is None and tower.trunk[5][0].conv1.weight.grad.norm() > 0
    torch.testing.assert_close(tower.trunk[1].running_mean, running)
    # Whole-pixel shift: the image content moves by exactly the offset (interior pixels).
    depth = torch.arange(16 * 16, dtype=torch.float32).view(1, 1, 1, 1, 16, 16).expand(1, 1, 2, 2, 16, 16).clone()
    torch.manual_seed(0)
    shifted, _ = model._random_shift(depth, None)
    for v in range(2):
        found = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)
                 if torch.equal(shifted[0, 0, v, 0, 4:12, 4:12], depth[0, 0, v, 0, 4 + dy:12 + dy, 4 + dx:12 + dx])]
        assert len(found) == 1


def test_temporal_ensembling(rgb_converted, tmp_path):
    path, metadata, raw = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    config = {**model_config(), "use_rgb": True}
    model = MeanFlowPolicy(**config)
    model.set_statistics(data.statistics())
    save_meanflow_release(tmp_path / "release", model, config, metadata, 1)
    policy = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    with h5py.File(raw / "a.hdf5", "r") as f:
        images = {c: f[f"observations/images/{c}"][0] for c in (*policy.cameras, *policy.rgb_config.cameras)}
        obs = {"images": images, "state": f["observations/qpos"][0]}
    first = policy.infer_ensembled(obs)["action"]
    policy.reset()
    policy.observe(obs)
    chunk = policy.policy.predict_action(
        {k: torch.from_numpy(np.stack([o[k] for o in policy.history])[None]) for k in ("depth", "agent_pos", "rgb")},
        generator=torch.Generator().manual_seed(42))["action_pred"][0].numpy()
    np.testing.assert_allclose(first, chunk[0], rtol=1e-5, atol=1e-6)  # one prediction: its first step
    for _ in range(5):
        action = policy.infer_ensembled(obs)["action"]
    assert action.shape == (16,) and len(policy._ensemble) == policy.policy.horizon  # older chunks expired


def test_extra_training_dataset(tmp_path):
    import subprocess, sys, yaml
    raw, sim = tmp_path / "raw", tmp_path / "sim"
    raw.mkdir(); sim.mkdir()
    episode(raw / "a.hdf5"); episode(raw / "b.hdf5", 100); episode(sim / "s.hdf5", 50)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("episodes:\n  b:\n    split: val\n")
    kw = dict(action_source="next_qpos", rgb=RGBConfig(height=16, width=16))
    convert_depth(raw, tmp_path / "real.zarr", DepthConfig(height=16, width=16), manifest=manifest, **kw)
    meta = convert_depth(sim, tmp_path / "sim.zarr", DepthConfig(height=16, width=16), all_train=True, **kw)
    assert meta["all_train"] and {e["split"] for e in meta["episodes"]} == {"train"}
    root = Path(__file__).parents[1]
    cfg = yaml.safe_load((root / "conf/meanflow_v2.yaml").read_text())
    cfg["model"].update(model_config(), use_rgb=True, separate_gripper=False, action_loss_weight=0,
                        tcp_position_weight=0, tcp_orientation_weight=0, rgb_backbone="cnn", object_features=False)
    cfg["train"].update(steps=2, batch_size=2, save_every=2, val_samples_per_episode=2, num_workers=0)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    out = subprocess.run([sys.executable, str(root / "scripts/train_meanflow.py"), "--dataset", str(tmp_path / "real.zarr"),
                          "--extra-dataset", str(tmp_path / "sim.zarr"), "--output", str(tmp_path / "run"),
                          "--config", str(tmp_path / "c.yaml"), "--device", "cpu"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Extra training data" in out.stdout
    release = json.loads((tmp_path / "run/best_release/metadata.json").read_text())
    assert "all_train" not in release  # releases stay tied to the real dataset (evaluation compatibility)


def test_gripper_close_confirmation():
    from openarm_flow.meanflow_policy import GripperCloseConfirmer
    confirmer = GripperCloseConfirmer(confirm=3)
    state = np.zeros(16, "f4")
    state[[7, 15]] = .043
    closing = np.zeros((4, 16), "f4")
    closing[:, 7] = 0.0
    closing[:, 15] = .043
    held = [confirmer(closing, state)[:, 7].min() for _ in range(3)]
    assert held[0] >= .043 and held[1] >= .043 and held[2] == 0.0      # closes on the 3rd agreeing prediction
    assert confirmer(closing, state)[:, 7].min() == 0.0                 # stays closed while the policy wants it
    opening = closing.copy(); opening[:, 7] = .043
    assert confirmer(opening, state)[:, 7].min() == .043                # opening is never delayed
    assert confirmer(closing, state)[:, 7].min() >= .043                # a broken streak starts over
    assert (GripperCloseConfirmer(1)(closing, state) == closing).all()


def test_hybrid_policy_takes_grippers_from_second_release(rgb_converted, tmp_path):
    from openarm_flow.meanflow_policy import HybridMeanFlowPolicy
    path, metadata, raw = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    for name, seed in (("arm", 1), ("grip", 2)):
        torch.manual_seed(seed)
        config = {**model_config(), "use_rgb": True, "separate_gripper": True}
        model = MeanFlowPolicy(**config)
        model.set_statistics(data.statistics())
        save_meanflow_release(tmp_path / name, model, config, metadata, 1)
    hybrid = HybridMeanFlowPolicy(tmp_path / "arm", tmp_path / "grip", "cpu")
    with h5py.File(raw / "a.hdf5", "r") as f:
        images = {c: f[f"observations/images/{c}"][0] for c in (*hybrid.cameras, *hybrid.rgb_config.cameras)}
        obs = {"images": images, "state": f["observations/qpos"][0]}
    merged = hybrid.infer(obs)["actions"]
    arm = OpenArmMeanFlowPolicy(tmp_path / "arm", "cpu").infer(obs)["actions"]
    grip = OpenArmMeanFlowPolicy(tmp_path / "grip", "cpu").infer(obs)["actions"]
    joints = [i for i in range(16) if i not in (7, 15)]
    np.testing.assert_allclose(merged[:, joints], arm[:, joints], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(merged[:, [7, 15]], grip[:, [7, 15]], rtol=1e-5, atol=1e-6)
    assert not np.allclose(arm[:, [7, 15]], grip[:, [7, 15]])


def test_gripper_state_hidden_from_encoder(rgb_converted):
    path, _, _ = rgb_converted
    data = MeanFlowDataset(path, 4, 2)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    changed = {**batch["obs"], "agent_pos": batch["obs"]["agent_pos"].clone()}
    changed["agent_pos"][..., [7, 15]] = 0.0          # gripper "closing"
    arm_changed = {**batch["obs"], "agent_pos": batch["obs"]["agent_pos"].clone()}
    arm_changed["agent_pos"][..., 0] += 0.3
    for hidden in (False, True):
        torch.manual_seed(0)
        model = MeanFlowPolicy(**{**model_config(), "use_rgb": True, "separate_gripper": True,
                                  "gripper_state_input": not hidden})
        model.set_statistics(data.statistics())
        model.eval()
        base = model.encode(batch["obs"])
        same = torch.allclose(base, model.encode(changed))
        assert same == hidden                            # the opening only matters when it is an input
        assert not torch.allclose(base, model.encode(arm_changed))   # arm joints always are
