"""FK geometry/gradients, phase labels and upgraded MeanFlow training contracts."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from test_meanflow import converted, model_config
from openarm_flow.kinematics import OpenArmKinematics
from openarm_flow.meanflow import MeanFlowPolicy
from openarm_flow.meanflow_data import MeanFlowDataset, grasp_phase_flags
from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy, save_meanflow_release


def v2_config():
    return {**model_config(), "separate_gripper": True, "action_loss_weight": .5,
            "tcp_position_weight": .1, "tcp_orientation_weight": .02, "gripper_loss_weight": 1.,
            "phase_weight": 3.}


def test_fk_matches_mujoco_random_joint_poses():
    root = Path(__file__).parents[1]
    if not (root / "simulation/openarm_sim").is_dir():
        pytest.skip("Robot simulator is not shipped/required in the Colab training bundle")
    mujoco = pytest.importorskip("mujoco")
    sys.path.insert(0, str(root / "simulation"))
    from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE
    from openarm_sim.runner import Sim
    from openarm_sim.scene import load_scene, load_config
    model, _ = load_scene(DEFAULT_SCENE)
    sim = Sim(model, load_config(DEFAULT_CONFIG))
    sim.reset()
    fk = OpenArmKinematics().double()
    rng = np.random.default_rng(5)
    for q in rng.uniform(-.8, .8, (10, 16)):
        sim.set_qpos16(q)
        mujoco.mj_kinematics(model, sim.d)
        p, r = fk(torch.tensor(q))
        for i, side in enumerate(("left", "right")):
            position, rotation = sim.arms[side].tcp_pose(sim.d)
            np.testing.assert_allclose(p[i].numpy(), position, atol=1e-6)
            np.testing.assert_allclose(r[i].numpy(), rotation, atol=1e-6)


def test_fk_gradient_matches_finite_differences():
    fk = OpenArmKinematics().double()
    q = torch.linspace(-.2, .3, 16, dtype=torch.float64)[None].requires_grad_()
    def forward(value):
        p, r = fk(value)
        return torch.cat([p.flatten(), r.flatten()])
    assert torch.autograd.gradcheck(forward, (q,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_phase_window_is_episode_local_and_excludes_initially_closed_gripper():
    actions = np.zeros((160, 16), dtype="f4")
    actions[:70, 15] = .043
    approach, phase = grasp_phase_flags(actions, 50)
    assert not approach[:, 0].any() and not phase[:, 0].any()
    assert np.flatnonzero(approach[:, 1]).tolist() == list(range(20, 70))
    assert np.flatnonzero(phase[:, 1]).tolist() == list(range(20, 95))


def test_phase_weight_does_not_cancel_for_an_all_approach_sample(converted):
    path, _, _ = converted
    dataset = MeanFlowDataset(path, 4, 2)
    model = MeanFlowPolicy(**model_config(), phase_weight=3., dispersive_weight=0)
    model.set_statistics(dataset.statistics())
    model.eval()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    batch["action_phase"] = torch.zeros(2, 4, 2, dtype=torch.bool)
    torch.manual_seed(4)
    baseline = model.compute_loss(batch)[0]
    g0 = torch.autograd.grad(baseline, model.velocity.output.weight)[0]
    batch["action_phase"].fill_(True)
    torch.manual_seed(4)
    weighted = model.compute_loss(batch)[0]
    g1 = torch.autograd.grad(weighted, model.velocity.output.weight)[0]
    torch.testing.assert_close(weighted, 3 * baseline)
    torch.testing.assert_close(g1, 3 * g0)


def test_v2_backward_endpoint_fk_and_noise_independent_gripper(converted, tmp_path):
    torch.set_num_threads(2)
    data_path, metadata, _ = converted
    dataset = MeanFlowDataset(data_path, 4, 2)
    model = MeanFlowPolicy(**v2_config())
    model.set_statistics(dataset.statistics())
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    before = model.velocity.output.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, parts = model.compute_loss(batch)
    assert {"meanflow_loss", "action_loss", "gripper_loss", "tcp_position_loss", "tcp_orientation_loss"} <= set(parts)
    loss.backward()
    for parameter in (model.encoder.cnn[0].weight, model.velocity.output.weight, model.gripper_head[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(before, model.velocity.output.weight)
    model.eval()
    calls = []
    hook = model.velocity.register_forward_hook(lambda *args: calls.append(1))
    a = model.predict_action(batch["obs"], noise=torch.zeros(2, 4, 14))["action"]
    hook.remove()
    assert len(calls) == 1
    b = model.predict_action(batch["obs"], noise=torch.ones(2, 4, 14))["action"]
    torch.testing.assert_close(a[..., [7, 15]], b[..., [7, 15]], rtol=0, atol=0)
    assert not torch.equal(a[..., :7], b[..., :7])
    assert (a >= model.kinematics.lower).all() and (a <= model.kinematics.upper).all()
    save_meanflow_release(tmp_path / "release", model, v2_config(), metadata, 1)
    assert json.loads((tmp_path / "release/metadata.json").read_text())["policy_type"] == "meanflow_depth_v2"
    loaded = OpenArmMeanFlowPolicy(tmp_path / "release", "cpu")
    actual = loaded.policy.predict_action(batch["obs"], noise=torch.zeros(2, 4, 14))["action"]
    torch.testing.assert_close(actual, a)
    with torch.no_grad():
        assert torch.isfinite(loaded.policy.compute_loss(batch)[0])


def test_padding_excluded_from_all_auxiliary_losses():
    class Field(torch.nn.Module):
        def forward(self, z, t, r, cond, return_features=False):
            result = z * 0
            return (result, (cond,)) if return_features else result
    model = MeanFlowPolicy(**v2_config(), dispersive_weight=0)
    model.velocity = Field()
    model.set_statistics({k: np.ones(16, "f4") if "scale" in k else np.zeros(16, "f4")
                          for k in ("state_mean", "state_scale", "action_mean", "action_scale")})
    model.eval()
    batch = {"obs": {"depth": torch.ones(2, 2, 2, 2, 16, 16), "agent_pos": torch.zeros(2, 2, 16)},
             "action": torch.zeros(2, 4, 16), "action_valid": torch.tensor([[1, 0, 0, 0]] * 2).bool(),
             "action_phase": torch.ones(2, 4, 2, dtype=torch.bool)}
    torch.manual_seed(3)
    expected, expected_parts = model.compute_loss(batch)
    batch["action"][:, 1:] = 1000
    torch.manual_seed(3)
    actual, actual_parts = model.compute_loss(batch)
    torch.testing.assert_close(actual, expected)
    assert actual_parts == expected_parts


def test_v2_cli_resume_and_approach_checkpoint_selection(converted, tmp_path):
    import zarr
    root = Path(__file__).parents[1]
    path, _, _ = converted
    store = zarr.open_group(str(path), mode="a")
    actions = store["data/action"][:]
    for start in (0, 5):
        actions[start:start + 2, [7, 15]] = .043
        actions[start + 2:start + 5, [7, 15]] = 0
    store["data/action"][:] = actions
    cfg = yaml.safe_load((root / "conf/meanflow_v2.yaml").read_text())
    cfg["model"] = v2_config()
    cfg["train"].update(steps=1, batch_size=2, val_samples_per_episode=3, save_every=1)
    config = tmp_path / "v2.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "run"
    cmd = [sys.executable, str(root / "scripts/train_meanflow.py"), "--dataset", str(path),
           "--output", str(output), "--config", str(config), "--device", "cpu"]
    subprocess.run(cmd, check=True)
    subprocess.run(cmd + ["--resume", "--steps", "2"], check=True)
    state = torch.load(output / "last.pt", weights_only=True, map_location="cpu")
    assert state["step"] == 2 and state["selection_metric"] == "val_selection_score"
    metrics = json.loads((output / "best_metrics.json").read_text())
    assert metrics["val_selection_uses_approach"] and metrics["val_approach_samples"] > 0
    assert np.isfinite(metrics["val_approach_tcp_rmse_mm"])
    subprocess.run([sys.executable, str(root / "scripts/eval_meanflow.py"), "--dataset", str(path),
                    "--checkpoint", str(output / "best_release"), "--device", "cpu", "--samples-per-episode", "3"], check=True)
