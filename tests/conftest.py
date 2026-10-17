"""Fake episodes in the exact layout openarm_mc's recorder writes."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from openarm_data import constants as C


def write_episode(
    path: Path,
    n: int = 120,
    commands: str | None = "full",
    task: str | None = None,
    complete: bool = True,
    seed: int = 0,
) -> Path:
    """An episode: a smooth sweep of every joint, the command leading qpos by
    three steps. `commands`: full | no_grippers | sparse_nan | mostly_nan | None."""
    rng = np.random.default_rng(seed)
    t = np.arange(n + 3) / C.RECORDER_FPS
    sweep = np.stack([0.3 * np.sin(2 * np.pi * 0.2 * t + j) for j in range(C.NUM_JOINTS)], axis=1)
    for g in C.GRIPPERS:
        sweep[:, g] = 0.02 + 0.02 * np.sin(2 * np.pi * 0.3 * t)
    command = sweep[3:].astype(np.float32)
    qpos = sweep[:n].astype(np.float32)
    if commands == "no_grippers":
        command[:, list(C.GRIPPERS)] = np.nan
    elif commands == "sparse_nan":
        command[:5] = np.nan  # before teleop took over
        command[[40, 41, 80]] = np.nan
    elif commands == "mostly_nan":
        command[::2] = np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        images = f.create_group("observations/images")
        images.create_dataset(C.H5_RGB.split("/")[-1], data=rng.integers(0, 256, (n, 240, 424, 3), dtype=np.uint8))
        images.create_dataset(C.H5_DEPTH.split("/")[-1], data=rng.integers(0, 1500, (n, 240, 424), dtype=np.uint16))
        f.create_dataset(C.H5_QPOS, data=qpos)
        f.create_dataset("observations/qvel", data=np.zeros_like(qpos))
        f.create_dataset("observations/effort", data=np.zeros_like(qpos))
        f.create_dataset(C.H5_ACTION, data=qpos)  # --use-state-as-action
        if commands is not None:
            f.create_dataset(C.H5_COMMANDS, data=command)
        f.create_dataset(C.H5_TIMESTAMP, data=(np.arange(n) * 20_000_000).astype(np.int64))
        f.attrs.update(
            {
                "sim": False,
                "frequency_hz": 50.0,
                "num_joints": 16,
                "gripper_unit": "stroke_m",
                "joint_names_json": json.dumps(list(C.JOINT_NAMES)),
                "complete": complete,
                "action_representation": "absolute_joint_position",
            }
        )
        if task is not None:
            f.attrs["task"] = task
    return path


@pytest.fixture
def episode_factory(tmp_path):
    def make(name="2026-10-08/episode_20261008_100000", **kwargs):
        return write_episode(tmp_path / "raw" / f"{name}.hdf5", **kwargs)

    return make
