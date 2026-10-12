"""Offline evaluation: a trained policy on held-out episodes.

For every `every`-th frame of each episode in the split, the policy predicts
a chunk from that frame's observation, exactly as openarm_act would hand it
over (raw RGB, raw depth, qpos), and the chunk is compared with the
episode's own labels. The "hold" baseline predicts staying where the arm is;
a policy that does not beat it has learned nothing about where to go.

For picking checkpoints and catching broken pipelines. It does not replace
running on the robot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths
from openarm_act_lerobot.convert import plan
from openarm_act_lerobot.convert import used_cameras
from openarm_act_lerobot.episode import EpisodeReader
from openarm_act_lerobot.run import PRETRAINED_DIR

log = logging.getLogger(__name__)


class ErrorTally:
    """Squared and absolute errors, per joint and per step of the chunk."""

    def __init__(self, horizon: int):
        self.sq = np.zeros((horizon, C.NUM_JOINTS))
        self.abs = np.zeros((horizon, C.NUM_JOINTS))
        self.count = 0

    def add(self, predicted: np.ndarray, truth: np.ndarray) -> None:
        error = predicted - truth
        self.sq += error**2
        self.abs += np.abs(error)
        self.count += 1

    def summary(self) -> dict:
        if not self.count:
            return {}
        rmse_joint = np.sqrt(self.sq.sum(0) / (self.count * len(self.sq)))
        arm = list(C.ARM_JOINTS)
        grippers = list(C.GRIPPERS)
        return {
            "chunks": self.count,
            "arm_rmse_rad": float(np.sqrt(self.sq[:, arm].sum() / (self.count * len(self.sq) * len(arm)))),
            "gripper_mae_mm": float(1000 * self.abs[:, grippers].sum() / (self.count * len(self.sq) * len(grippers))),
            "rmse_per_joint": {name: float(v) for name, v in zip(C.JOINT_NAMES, rmse_joint, strict=True)},
            # How the error grows along the chunk: the first step, a quarter, half, the last.
            "arm_rmse_by_step": {
                str(step): float(np.sqrt(self.sq[step, arm].sum() / (self.count * len(arm))))
                for step in sorted({0, len(self.sq) // 4, len(self.sq) // 2, len(self.sq) - 1})
            },
        }


def state_history(states: np.ndarray, t: int, steps: int) -> np.ndarray:
    """states[t] (steps 1), or the last `steps` up to t, oldest first; before
    the episode's start the first state repeats, as LeRobot pads in training."""
    if steps == 1:
        return states[t]
    return states[[max(0, t - k) for k in range(steps - 1, -1, -1)]]


def evaluate(cfg: DictConfig) -> dict:
    from openarm_act_lerobot.policy import OpenArmPolicy

    policy = OpenArmPolicy(cfg.checkpoint, device=str(cfg.device))
    run_cfg, step_dir = policy.run_cfg, policy.directory
    raw = paths.resolve(cfg.paths.raw or run_cfg.paths.raw)
    cameras = used_cameras(run_cfg)
    horizon = int(run_cfg.model.chunk_size)
    every = max(1, int(cfg.every))
    # The episodes the conversion left out as this split: the manifest's
    # `split:`, and data.val_fraction's hold-out, chosen the same way.
    plan_cfg = OmegaConf.merge(run_cfg, {"paths": {"raw": str(raw)}})
    chosen = [p for p in plan(plan_cfg) if (p.skip or "").startswith(f"split {cfg.split}")]

    model, hold = ErrorTally(horizon), ErrorTally(horizon)
    per_episode = []
    for item in chosen:
        name = item.id
        with EpisodeReader(item.path) as episode:
            labels = episode.labels(
                str(run_cfg.data.action_source), int(run_cfg.data.fps), float(run_cfg.data.max_nan_fraction)
            )
            ep_model, ep_hold = ErrorTally(horizon), ErrorTally(horizon)
            for start in range(0, len(labels.indices) - horizon + 1, every):
                obs = {
                    "images": {camera: episode.image(camera, labels.indices[start]) for camera in cameras},
                    "state": state_history(labels.state, start, policy.state_steps),
                }
                predicted = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)[:horizon]
                truth = labels.actions[start : start + horizon].astype(np.float64)
                still = np.repeat(labels.state[start][None].astype(np.float64), horizon, axis=0)
                for tally, guess in ((model, predicted), (ep_model, predicted), (hold, still), (ep_hold, still)):
                    tally.add(guess, truth)
        per_episode.append({"id": name, "model": ep_model.summary(), "hold": ep_hold.summary()})
        log.info(
            "%s: arm rmse %.4f rad (hold %.4f)",
            name,
            ep_model.summary().get("arm_rmse_rad", float("nan")),
            ep_hold.summary().get("arm_rmse_rad", float("nan")),
        )
        if cfg.max_episodes and len(per_episode) >= int(cfg.max_episodes):
            break
    if not per_episode:
        raise ValueError(f"no usable episodes with split {cfg.split!r} under {raw}")

    result = {
        "checkpoint": str(step_dir),
        "split": cfg.split,
        "episodes": len(per_episode),
        "model": model.summary(),
        "hold": hold.summary(),
        "per_episode": per_episode,
    }
    # Beside pretrained_model/ in a checkpoint, inside a release.
    where = step_dir.parent if step_dir.name == PRETRAINED_DIR else step_dir
    output = Path(cfg.output) if cfg.output else where / f"offline_eval_{cfg.split}.json"
    output.write_text(json.dumps(result, indent=2))
    result["output"] = str(output)
    return result
