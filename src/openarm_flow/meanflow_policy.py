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


class GripperCloseConfirmer:
    """Closing a gripper too early is harmful (it grabs air), closing a few ticks late is not.

    A gripper may only close after `confirm` consecutive predictions that all want it closed; until then its
    commands are held at (at least) its current opening. Opening is never delayed.
    """
    COLUMNS = (7, 15)

    def __init__(self, confirm=1, threshold=.03):
        if confirm < 1:
            raise ValueError("confirm must be at least 1")
        self.confirm, self.threshold = confirm, threshold
        self.reset()

    def reset(self):
        self.votes = {c: 0 for c in self.COLUMNS}
        self.latched = {c: False for c in self.COLUMNS}

    def __call__(self, chunk, state):
        """chunk [T,16] -> chunk with unconfirmed closes removed; `state` is the current qpos [16]."""
        if self.confirm == 1:
            return chunk
        chunk = chunk.copy()
        for c in self.COLUMNS:
            if chunk[:, c].min() < self.threshold:
                self.votes[c] += 1
                self.latched[c] = self.latched[c] or self.votes[c] >= self.confirm
            else:
                self.votes[c], self.latched[c] = 0, False
            if not self.latched[c]:
                chunk[:, c] = np.maximum(chunk[:, c], max(float(state[c]), self.threshold))
        return chunk


class OpenArmMeanFlowPolicy:
    def __init__(self, checkpoint, device="auto", seed=42, close_confirm=1):
        directory = Path(checkpoint)
        self.config = json.loads((directory / "config.json").read_text())
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata.get("policy_type") not in ("meanflow_depth", "meanflow_depth_v2"):
            raise ValueError("Expected dual-depth MeanFlow release")
        if (self.metadata["policy_type"] == "meanflow_depth_v2") != bool(self.config.get("separate_gripper", False)):
            raise ValueError("Release architecture version disagrees with config")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        # Weights come from model.pt: never download pretrained backbone weights when loading a release.
        self.policy = MeanFlowPolicy(**{**self.config, "pretrained_backbone": False})
        self.policy.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        self.policy.to(self.device).eval().requires_grad_(False)
        self.depth_config = DepthConfig(**self.metadata["depth"])
        self.cameras = list(self.depth_config.cameras)
        if self.policy.use_rgb != bool(self.metadata.get("rgb")):
            raise ValueError("Model RGB setting disagrees with release metadata")
        self.rgb_config = RGBConfig(**self.metadata["rgb"]) if self.policy.use_rgb else None
        self.estimator = self.object_features = None
        if self.policy.object_features:
            if not self.metadata.get("objects"):
                raise ValueError("Model uses object features but the release metadata has no object config")
            from openarm_flow.object_features import ObjectConfig, ObjectFeatures, SceneEstimator
            object_config = ObjectConfig.from_metadata(self.metadata["objects"]["config"])
            self.estimator, self.object_features = SceneEstimator(object_config), ObjectFeatures(object_config)
            self.object_cameras = list(object_config.cameras)
        self.state_steps, self.chunk_size = self.policy.n_obs_steps, self.policy.n_action_steps
        self.seed = seed
        self.confirmer = GripperCloseConfirmer(close_confirm)
        self.reset()

    def _keys(self):
        return ("depth", "agent_pos", *(("rgb",) if self.rgb_config else ()),
                *(("objects",) if self.object_features is not None else ()))

    def reset(self):
        self.history = deque(maxlen=self.state_steps)
        self._ensemble, self._tick = [], 0
        self.scene = None   # object-centric scene, estimated from the first observation of an episode
        if hasattr(self, "confirmer"):
            self.confirmer.reset()
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
            frames.append(resize_depth(raw, self.depth_config, camera))
        item = {"depth": encode_depth(np.stack(frames), self.depth_config), "agent_pos": state.copy()}
        if self.object_features is not None:
            if self.scene is None:
                self.scene = self.estimator.estimate({c: np.asarray(obs["images"][f"{c}_rgb"])
                                                      for c in self.object_cameras})
            item["objects"] = self.object_features(self.scene, state[None])[0]
        if self.rgb_config is not None:
            colors = []
            for camera in self.rgb_config.cameras:
                raw = np.asarray(obs["images"][camera])
                if list(raw.shape[:2]) != self.metadata["source_shapes"][camera]:
                    raise ValueError(f"{camera}: resolution differs from recording")
                colors.append(resize_rgb(raw, self.rgb_config, camera))
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
                 for key in self._keys()}
        action = self.policy.predict_action(batch, generator=self.generator)["action"][0].cpu().numpy()
        if action.shape != (self.chunk_size, 16) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/invalid action chunk")
        return {"actions": self.confirmer(action, self.history[-1]["agent_pos"])}

    @torch.inference_mode()
    def infer_ensembled(self, obs=None, *, decay=0.1, horizon=None):
        """Temporal ensembling (ACT): call once per control tick. Predicts a full chunk every tick and returns the
        weighted mean of all past chunks' predictions for this tick, weights exp(-decay * i) with i = 0 the
        oldest prediction. Smooths the closed loop and averages out per-sample noise."""
        if obs is not None:
            self.observe(obs)
        if not self.history:
            raise ValueError("Call observe() before infer_ensembled()")
        batch = {key: torch.from_numpy(np.stack([o[key] for o in self.history])[None]).to(self.device)
                 for key in self._keys()}
        chunk = self.policy.predict_action(batch, generator=self.generator)["action_pred"][0].cpu().numpy()
        horizon = min(horizon or len(chunk), len(chunk))
        self._ensemble.append((self._tick, chunk[:horizon]))
        self._ensemble = [(t0, c) for t0, c in self._ensemble if self._tick - t0 < len(c)]
        preds = np.stack([c[self._tick - t0] for t0, c in self._ensemble])  # oldest first
        weights = np.exp(-decay * np.arange(len(preds)))
        action = (weights[:, None] * preds).sum(0) / weights.sum()
        self._tick += 1
        action = self.confirmer(action[None], self.history[-1]["agent_pos"])[0]
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/invalid ensembled action")
        return {"action": action}


class HybridMeanFlowPolicy:
    """Arm joints from one release, the two grippers from another (same observation/control contract).

    The arm policy decides where to go, the gripper policy decides when to close; each preprocesses the raw camera
    images itself (its own crops, resolution and history), so releases of different families can be combined.
    """

    def __init__(self, arm_checkpoint, gripper_checkpoint, device="auto", seed=42, **kwargs):
        self.arm = OpenArmMeanFlowPolicy(arm_checkpoint, device, seed, **kwargs)
        self.gripper = OpenArmMeanFlowPolicy(gripper_checkpoint, device, seed, **kwargs)
        a, g = self.arm, self.gripper
        if (a.chunk_size, a.metadata["fps"]) != (g.chunk_size, g.metadata["fps"]):
            raise ValueError("Releases must share the action chunk size and control rate")
        if a.state_steps != g.state_steps:
            pass  # each keeps its own history; only the chunk length has to agree
        self.metadata, self.config, self.policy = a.metadata, a.config, a.policy
        self.device, self.chunk_size, self.state_steps = a.device, a.chunk_size, a.state_steps
        self.cameras = list(dict.fromkeys([*a.cameras, *g.cameras]))
        self.rgb_config = a.rgb_config
        self._gripper_columns = list(GripperCloseConfirmer.COLUMNS)

    @property
    def history(self):
        return self.arm.history

    def reset(self):
        self.arm.reset()
        self.gripper.reset()

    def observe(self, obs):
        self.arm.observe(obs)
        self.gripper.observe(obs)

    def _merge(self, arm_part, gripper_part):
        merged = np.array(arm_part, copy=True)
        merged[..., self._gripper_columns] = np.asarray(gripper_part)[..., self._gripper_columns]
        return merged

    def infer(self, obs=None):
        if obs is not None:
            self.observe(obs)
        return {"actions": self._merge(self.arm.infer()["actions"], self.gripper.infer()["actions"])}

    def infer_ensembled(self, obs=None, **kwargs):
        if obs is not None:
            self.observe(obs)
        return {"action": self._merge(self.arm.infer_ensembled(**kwargs)["action"],
                                      self.gripper.infer_ensembled(**kwargs)["action"])}
