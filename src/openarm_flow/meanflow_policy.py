"""Self-contained dual-camera MeanFlow release and history adapter."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from openarm_flow.depth_views import DepthConfig, encode_depth, resize_depth
from openarm_flow.meanflow import MeanFlowPolicy
from openarm_flow.rgb_views import RGBConfig, resize_rgb


def save_meanflow_release(directory, model, config, metadata, step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt.tmp")
    (directory / "model.pt.tmp").replace(directory / "model.pt")
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    kind = "meanflow_depth_v2" if model.separate_gripper else "meanflow_depth"
    (directory / "metadata.json").write_text(json.dumps({**metadata, "policy_type": kind, "step": step}, indent=2))


class OpenArmMeanFlowPolicy:
    def __init__(self, checkpoint, device="auto", seed=42):
        directory = Path(checkpoint)
        self.config = json.loads((directory / "config.json").read_text())
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata.get("policy_type") not in ("meanflow_depth", "meanflow_depth_v2"):
            raise ValueError("Expected dual-depth MeanFlow release")
        if (self.metadata["policy_type"] == "meanflow_depth_v2") != bool(self.config.get("separate_gripper", False)):
            raise ValueError("Release architecture version disagrees with config")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.policy = MeanFlowPolicy(**self.config)
        self.policy.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        self.policy.to(self.device).eval().requires_grad_(False)
        self.depth_config = DepthConfig(**self.metadata["depth"])
        self.cameras = list(self.depth_config.cameras)
        if self.policy.use_rgb != bool(self.metadata.get("rgb")):
            raise ValueError("Model RGB setting disagrees with release metadata")
        self.rgb_config = RGBConfig(**self.metadata["rgb"]) if self.policy.use_rgb else None
        self.state_steps, self.chunk_size = self.policy.n_obs_steps, self.policy.n_action_steps
        self.seed = seed
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.state_steps)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def observe(self, obs):
        """Call every 50 Hz tick, including while executing the queued action chunk."""
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected finite qpos [16]")
        frames = []
        for camera in self.cameras:
            raw = np.asarray(obs["images"][camera])
            if list(raw.shape) != self.metadata["source_shapes"][camera]:
                raise ValueError(f"{camera}: resolution differs from recording")
            frames.append(resize_depth(raw, self.depth_config))
        item = {"depth": encode_depth(np.stack(frames), self.depth_config), "agent_pos": state.copy()}
        if self.rgb_config is not None:
            colors = []
            for camera in self.rgb_config.cameras:
                raw = np.asarray(obs["images"][camera])
                if list(raw.shape[:2]) != self.metadata["source_shapes"][camera]:
                    raise ValueError(f"{camera}: resolution differs from recording")
                colors.append(resize_rgb(raw, self.rgb_config))
            item["rgb"] = np.stack(colors)
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
                 for key in ("depth", "agent_pos", *(("rgb",) if self.rgb_config else ()))}
        action = self.policy.predict_action(batch, generator=self.generator)["action"][0].cpu().numpy()
        if action.shape != (self.chunk_size, 16) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/invalid action chunk")
        return {"actions": action}
