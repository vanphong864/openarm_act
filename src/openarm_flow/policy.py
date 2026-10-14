"""Self-contained FlowPolicy releases: XYZ preprocessing, history and absolute targets."""

from collections import deque
import copy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from openarm_flow._vendor.policy.flowpolicy import FlowPolicy
from openarm_flow.pointcloud import Camera, PointCloudConfig, project_depth


def build_model(config, metadata):
    cfg = copy.deepcopy(config)
    horizon, obs, actions = cfg["horizon"], cfg["n_obs_steps"], cfg["n_action_steps"]
    divisor = 2 ** (len(cfg["down_dims"]) - 1)
    if min(horizon, obs, actions) < 1 or horizon % divisor or obs + actions - 1 > horizon:
        raise ValueError("Horizon must fit observation/action alignment and U-Net downsampling")
    if not cfg.get("obs_as_global_cond", True):
        raise ValueError("OpenArm adapter supports global observation conditioning")
    points = metadata["pointcloud"]
    cfg["use_pc_color"] = points["use_color"]
    channels = 6 if points["use_color"] else 3
    cfg["shape_meta"] = {"obs": {"point_cloud": {"shape": [points["num_points"], channels]},
                                  "agent_pos": {"shape": [16]}}, "action": {"shape": [16]}}
    cfg["shape_meta"] = OmegaConf.create(cfg["shape_meta"])
    cfg["pointcloud_encoder_cfg"] = OmegaConf.create(cfg["pointcloud_encoder_cfg"])
    if cfg["pointcloud_encoder_cfg"].out_channels != cfg["encoder_output_dim"]:
        raise ValueError("PointNet output width must match encoder_output_dim")
    return FlowPolicy(**cfg)


def save_release(directory, model, config, metadata, step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The run owns this directory; atomically replace the weight file at save points.
    temporary = directory / "model.pt.tmp"
    torch.save(model.state_dict(), temporary)
    temporary.replace(directory / "model.pt")
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    (directory / "metadata.json").write_text(json.dumps({**metadata, "policy_type": "flowpolicy", "step": step}, indent=2))


class OpenArmFlowPolicy:
    def __init__(self, checkpoint, device="auto"):
        directory = Path(checkpoint)
        self.config = json.loads((directory / "config.json").read_text())
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata.get("policy_type") != "flowpolicy":
            raise ValueError("Not an OpenArm FlowPolicy release")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.policy = build_model(self.config, self.metadata)
        self.policy.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        self.policy.to(self.device).eval().requires_grad_(False)
        self.camera = Camera(**self.metadata["camera"])
        self.cloud = PointCloudConfig(**self.metadata["pointcloud"])
        self.state_steps = self.config["n_obs_steps"]
        self.chunk_size = self.config["n_action_steps"]
        self.cameras = ["chest_depth"] + (["chest"] if self.cloud.use_color else [])
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.state_steps)
        self.rng = np.random.default_rng(self.metadata["seed"])

    def observe(self, obs):
        """Call at dataset fps, including while executing queued chunk actions."""
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected finite current qpos [16]")
        points = project_depth(obs["images"]["chest_depth"], self.camera, self.cloud, self.rng,
                               obs["images"].get("chest"))
        item = {"point_cloud": points, "agent_pos": state.copy()}
        if not self.history:
            self.history.extend([item] * self.state_steps)
        else:
            self.history.append(item)

    @torch.inference_mode()
    def infer(self, obs=None):
        if obs is not None:
            self.observe(obs)
        if not self.history:
            raise ValueError("Call observe() before infer()")
        batch = {key: torch.from_numpy(np.stack([o[key] for o in self.history])[None]).to(self.device)
                 for key in ("point_cloud", "agent_pos")}
        action = self.policy.predict_action(batch)["action"][0].cpu().numpy()
        if action.shape != (self.chunk_size, 16) or not np.isfinite(action).all():
            raise ValueError(f"Invalid predicted action: {action.shape}")
        return {"actions": action}
