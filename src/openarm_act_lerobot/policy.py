"""A trained ACT, taking OpenArm observations as openarm_act hands them over.

    infer({"images": {"chest": uint8 [H, W, 3], "chest_depth": uint16 [H, W] mm},
           "state": float32 [16] qpos})
        -> {"actions": float32 [chunk, 16]}  absolute targets, the dataset's
                                             joint order and units

The observation is what openarm_act_lerobot_client sends; the server and the
offline evaluation both call this. The images go through images.py, as the
converter's did, then the processors saved with the checkpoint (normalising
state and actions with the run's stats, delta actions when the run used
them), then the model, which resizes and normalises the images itself.
Nothing about the pipeline is rebuilt from defaults. ACT predicts with its
latent at zero, so the same observation always gives the same chunk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from openarm_act_lerobot import constants as C
from openarm_act_lerobot.images import depth_to_metres
from openarm_act_lerobot.images import to_model_input
from openarm_act_lerobot.run import checkpoint_step
from openarm_act_lerobot.run import load_run_config
from openarm_act_lerobot.train_config import policy_metadata
from openarm_act_lerobot.train_config import resolve_device

log = logging.getLogger(__name__)


class OpenArmPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class

        import openarm_act_lerobot.configuration_openarm_act  # noqa: F401  -- registers "openarm_act"
        from openarm_act_lerobot.processors import link_delta_steps  # also registers the steps for loading

        self.run_cfg, self.directory = load_run_config(checkpoint)
        config = PreTrainedConfig.from_pretrained(self.directory)
        config.device = resolve_device(device)
        config.pretrained_path = self.directory

        # The weights, the backbone's included, all come from the checkpoint: no Hub.
        self.policy = get_policy_class(config.type).from_pretrained(self.directory, config=config)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(self.directory),
            preprocessor_overrides={"device_processor": {"device": config.device}},
        )
        link_delta_steps(self.preprocessor, self.postprocessor)
        self.device = config.device
        self.cameras = [str(c) for c in self.run_cfg.data.cameras]
        self.chunk_size = int(config.chunk_size)
        self.metadata = policy_metadata(self.run_cfg)
        self.metadata["checkpoint"] = str(self.directory)
        self.metadata["model_image_size"] = list(config.image_size)
        self.metadata["step"] = checkpoint_step(self.directory)
        if self.metadata["step"] is None and (self.directory / "metadata.json").exists():  # a release
            self.metadata["step"] = json.loads((self.directory / "metadata.json").read_text()).get("step")
        log.info("loaded %s on %s: cameras %s, chunk %d, the backbone sees %s", self.directory, self.device,
                 self.cameras, self.chunk_size, list(config.image_size))

    def batch(self, obs: dict) -> dict:
        """The observation as LeRobot's dataset yields a frame (no batch dim)."""
        import torch

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (C.NUM_JOINTS,):
            raise ValueError(f"state has {state.shape[0]} values, expected {C.NUM_JOINTS}")
        images = obs.get("images") or {}
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise KeyError(f"no image from camera(s) {missing}; got {sorted(images)}")
        batch = {C.LEROBOT_STATE: torch.from_numpy(state)}
        raw_depth = str(self.run_cfg.data.get("depth_format") or "gray8") == "raw"
        for camera in self.cameras:
            image = images[camera]
            if camera == C.CAMERA_DEPTH and raw_depth:  # metres, as LeRobot's depth feature yields them
                array = depth_to_metres(image)
            else:
                array = to_model_input(image)
            batch[C.LEROBOT_IMAGE_PREFIX + camera] = torch.from_numpy(array)
        return batch

    def infer(self, obs: dict) -> dict:
        import torch

        with torch.inference_mode():
            batch = self.preprocessor(self.batch(obs))
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(actions)
        actions = actions.to(torch.float32).cpu().numpy()[0]
        if actions.shape != (self.chunk_size, C.NUM_JOINTS):
            raise AssertionError(f"the policy returned actions of shape {actions.shape}")
        return {"actions": actions}

    def reset(self) -> None:
        self.policy.reset()
