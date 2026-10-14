"""Read one episode of openarm_mc's recorder and turn it into training labels.

The recorder's `action` is the measured qpos of the same step (the dashboard
records with --use-state-as-action), so on its own it says nothing about
where the arm goes next. The labels are made here instead, from one of:

    commands    joint_commands[t]: teleop's targets through the proxy, the
                ALOHA convention. Columns the source never commanded (all
                NaN, e.g. grippers) fall back to next_qpos; isolated NaN is
                interpolated, leading NaN trimmed. An episode with more
                missing rows than allowed falls back to next_qpos whole.
    next_qpos   qpos[t + stride]: where the arm actually was one step later.
    qpos        qpos[t]: what act_pipeline trains on. For comparison only.

Kept free of torch and LeRobot, so the inspector and the tests run without
either.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np

from openarm_act_lerobot import constants as C

ACTION_SOURCES = ("commands", "next_qpos", "qpos")


def find_episodes(root: Path) -> list[Path]:
    """Every finished episode under `root`, oldest first.

    `<episode>.partial.hdf5` is a recording whose compression never finished;
    it is left out (episode_compress.py turns it into the final file).
    """
    root = Path(root)
    return sorted(p for p in root.rglob("*.hdf5") if not p.name.endswith(".partial.hdf5"))


def episode_id(path: Path, root: Path) -> str:
    """`2026-10-08/episode_20261008_104216`: the path under the root, no suffix."""
    return Path(path).relative_to(root).with_suffix("").as_posix()


@dataclasses.dataclass
class Labels:
    """What one episode contributes to training, at the dataset's rate."""

    indices: np.ndarray  # [T] rows of the HDF5 the frames come from
    state: np.ndarray  # [T, 16] float32
    actions: np.ndarray  # [T, 16] float32, absolute targets
    source: str  # the action source actually used
    notes: list[str]


class EpisodeReader:
    """One HDF5 episode, open for reading. Use as a context manager."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        # Some recorder-produced HDF5 files contain an attribute datatype that
        # makes HDF5's bulk attribute iterator fail (while normal datasets and
        # individual attributes remain readable).  Only these contract fields
        # are needed by the reader; avoid iterating the whole attribute table.
        self.attrs = {}
        for name in ("task", "frequency_hz", "complete", "joint_names_json",
                     "gripper_unit", "depth_scale"):
            try:
                self.attrs[name] = self.file.attrs[name]
            except (KeyError, RuntimeError, OSError, ValueError):
                # The HF-exported episodes have a malformed/missing
                # ``complete`` attribute, but their datasets are readable and
                # the converter validates shapes/frequency below.  Treat that
                # marker as complete; other optional fields keep their normal
                # defaults.
                if name == "complete":
                    self.attrs[name] = True

    def __enter__(self) -> EpisodeReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.file.close()

    def __len__(self) -> int:
        return int(self.file[C.H5_QPOS].shape[0])

    @property
    def task(self) -> str | None:
        """attrs["task"], once the recorder writes one."""
        task = self.attrs.get("task")
        if task is None:
            return None
        if isinstance(task, bytes):
            task = task.decode()
        return str(task).strip() or None

    @property
    def frequency_hz(self) -> float:
        return float(self.attrs.get("frequency_hz", C.RECORDER_FPS))

    @property
    def image_size(self) -> tuple[int, int]:
        """(height, width) of the camera frames; openarm_mc has recorded both
        640x480 and, since, 424x240."""
        return tuple(int(v) for v in self.file[C.H5_RGB].shape[1:3])

    def problems(
        self, min_frames: int = 1, need_depth: bool = False, image_size: tuple[int, int] | None = None
    ) -> list[str]:
        """Why this episode cannot be trained on; empty when it can. With
        `image_size`, frames of another size are a problem too: one dataset,
        one camera resolution."""
        found = []
        for key in (C.H5_RGB, C.H5_QPOS) + ((C.H5_DEPTH,) if need_depth else ()):
            if key not in self.file:
                found.append(f"no {key}")
        if found:
            return found
        if not bool(self.attrs.get("complete", False)):
            found.append("not complete (compression unfinished?)")
        names = self.attrs.get("joint_names_json")
        if names is not None and tuple(json.loads(names)) != C.JOINT_NAMES:
            found.append(f"joint order {json.loads(names)} is not the dataset's")
        if self.file[C.H5_QPOS].shape[1:] != (C.NUM_JOINTS,):
            found.append(f"qpos has shape {self.file[C.H5_QPOS].shape}")
        unit = self.attrs.get("gripper_unit", "stroke_m")
        if unit != "stroke_m":
            found.append(f"gripper_unit is {unit!r}, expected 'stroke_m'")
        rgb_shape = self.file[C.H5_RGB].shape[1:]
        if len(rgb_shape) != 3 or rgb_shape[2] != 3:
            found.append(f"chest_rgb frames are {rgb_shape}, expected (height, width, 3)")
        elif image_size is not None and tuple(rgb_shape[:2]) != tuple(image_size):
            found.append(f"chest_rgb frames are {rgb_shape[1]}x{rgb_shape[0]}, the dataset's are "
                         f"{image_size[1]}x{image_size[0]}")
        if self.file[C.H5_RGB].shape[0] != len(self):
            found.append("chest_rgb and qpos differ in length")
        if need_depth and self.file[C.H5_DEPTH].shape != (len(self), *rgb_shape[:2]):
            found.append(f"chest_depth has shape {self.file[C.H5_DEPTH].shape}, chest_rgb {self.file[C.H5_RGB].shape}")
        if abs(self.frequency_hz - C.RECORDER_FPS) > 1e-6:
            found.append(f"recorded at {self.frequency_hz} Hz, expected {C.RECORDER_FPS}")
        if len(self) < min_frames:
            found.append(f"{len(self)} frames, fewer than {min_frames}")
        elif not np.all(np.isfinite(self.file[C.H5_QPOS][:])):
            found.append("qpos holds NaN or infinity")
        return found

    def qpos(self) -> np.ndarray:
        return np.asarray(self.file[C.H5_QPOS][:], dtype=np.float32)

    def commands(self) -> np.ndarray | None:
        if C.H5_COMMANDS not in self.file:
            return None
        return np.asarray(self.file[C.H5_COMMANDS][:], dtype=np.float32)

    def rgb(self, index: int) -> np.ndarray:
        return np.asarray(self.file[C.H5_RGB][int(index)], dtype=np.uint8)

    def depth(self, index: int) -> np.ndarray:
        """uint16 mm, 0 = no reading."""
        return np.asarray(self.file[C.H5_DEPTH][int(index)], dtype=np.uint16)

    def image(self, camera: str, index: int) -> np.ndarray:
        """A camera by its dataset name: chest (RGB) or chest_depth (raw mm)."""
        if camera == C.CAMERA_RGB:
            return self.rgb(index)
        if camera == C.CAMERA_DEPTH:
            return self.depth(index)
        raise ValueError(f"unknown camera {camera!r}; the recorder has {C.CAMERA_RGB}, {C.CAMERA_DEPTH}")

    def labels(self, action_source: str = "commands", fps: int = C.RECORDER_FPS, max_nan_fraction: float = 0.02) -> Labels:
        """State and action labels at `fps`, which must divide the recorder's."""
        if action_source not in ACTION_SOURCES:
            raise ValueError(f"action_source must be one of {ACTION_SOURCES}, not {action_source!r}")
        if fps <= 0 or C.RECORDER_FPS % fps:
            raise ValueError(f"fps must divide {C.RECORDER_FPS}, not {fps}")
        stride = C.RECORDER_FPS // fps
        qpos = self.qpos()
        n = len(qpos)
        notes: list[str] = []

        source = action_source
        start, end = 0, n
        if source == "commands":
            filled = _fill_commands(self.commands(), qpos, stride, max_nan_fraction, notes)
            if filled is None:
                source = "next_qpos"
            else:
                targets, start, end = filled
        if source == "next_qpos":
            targets, end = _next_qpos(qpos, stride), n - stride
        elif source == "qpos":
            targets = qpos

        indices = np.arange(start, end, stride)
        if len(indices) == 0:
            raise ValueError(f"{self.path}: no frames left after labelling")
        actions = targets[indices].astype(np.float32)
        if not np.all(np.isfinite(actions)):
            raise AssertionError(f"{self.path}: labels still hold NaN")
        return Labels(indices, qpos[indices].astype(np.float32), actions, source, notes)


def _next_qpos(qpos: np.ndarray, stride: int) -> np.ndarray:
    """qpos[t + stride] at t; the last `stride` rows have none (NaN)."""
    out = np.full_like(qpos, np.nan)
    out[:-stride] = qpos[stride:]
    return out


def _fill_commands(commands, qpos, stride, max_nan_fraction, notes):
    """-> (targets, start, end) from joint_commands, or None to fall back."""
    if commands is None:
        notes.append("no joint_commands in the file: next_qpos")
        return None
    commands = commands.copy()
    missing = ~np.isfinite(commands)
    never = np.all(missing, axis=0)
    if np.all(never):
        notes.append("joint_commands all NaN: next_qpos")
        return None
    end = len(qpos)
    if np.any(never):
        names = [C.JOINT_NAMES[i] for i in np.flatnonzero(never)]
        notes.append(f"never commanded, next_qpos for: {', '.join(names)}")
        commands[:, never] = _next_qpos(qpos, stride)[:, never]
        end = len(qpos) - stride
        missing[:, never] = False

    rows = np.any(missing, axis=1)
    first = int(np.argmax(~rows))
    if rows[first:].mean() > max_nan_fraction:
        notes.append(f"{rows[first:].mean():.1%} of command rows missing (> {max_nan_fraction:.1%}): next_qpos")
        return None
    if first:
        notes.append(f"trimmed {first} leading rows with no command")
    if rows[first:].any():
        notes.append(f"interpolated {int(rows[first:].sum())} rows with missing commands")
        steps = np.arange(len(commands))
        for column in np.flatnonzero(missing[first:].any(axis=0)):
            valid = ~missing[:, column]
            commands[:, column] = np.interp(steps, steps[valid], commands[valid, column])
    return commands, first, end


def command_lag(qpos: np.ndarray, commands: np.ndarray, max_lag: int = 25) -> dict[str, float]:
    """Steps by which qpos follows the command, per arm joint (cross-correlated
    velocities); NaN for a joint that barely moves or has no command."""
    lags = {}
    for i in C.ARM_JOINTS:
        q, c = np.diff(qpos[:, i]), np.diff(commands[:, i])
        ok = np.isfinite(c)
        if ok.sum() < 4 * max_lag or np.std(q[ok]) < 1e-4:
            lags[C.JOINT_NAMES[i]] = float("nan")
            continue
        c = np.where(ok, c, 0.0)
        # Correlation, not a raw dot product: the overlap shrinks with the lag,
        # which would otherwise favour lag 0.
        scores = [np.corrcoef(c[: len(c) - lag], q[lag:])[0, 1] for lag in range(max_lag + 1)]
        lags[C.JOINT_NAMES[i]] = float(np.nanargmax(scores))
    return lags
