"""Streaming HDF5 conversion and episode-safe sequences for FlowPolicy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from openarm_act_lerobot.constants import H5_DEPTH, H5_RGB, JOINT_NAMES
from openarm_act_lerobot.episode import EpisodeReader, episode_id, find_episodes
from openarm_act_lerobot.manifest import Manifest
from openarm_flow.pointcloud import geometry_metadata, project_depth


def convert(raw, output, camera, cloud, *, manifest=None, fps=50, action_source="commands",
            val_fraction=0.15, seed=42, max_episodes=None, max_frames=None):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}. Choose a new output to avoid mixing datasets")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    manifest = Manifest.load(manifest)
    episodes = []
    skipped = []
    for path in find_episodes(raw):
        eid = episode_id(path, raw)
        with EpisodeReader(path) as reader:
            meta = manifest.resolve(eid, reader.task)
            if meta.exclude:
                skipped.append({"id": eid, "reason": "manifest exclude"})
                continue
            # Depth-only data is supported; do not require an RGB stream.
            if H5_DEPTH not in reader.file:
                raise ValueError(f"{eid}: no depth")
            if reader.file[H5_DEPTH].shape != (len(reader), camera.height, camera.width):
                raise ValueError(f"{eid}: depth dimensions do not match calibration")
            if not reader.attrs.get("complete", False):
                raise ValueError(f"{eid}: incomplete episode")
            if reader.frequency_hz != 50 or reader.qpos().shape != (len(reader), 16):
                raise ValueError(f"{eid}: expected 50 Hz and 16 joints")
            names = reader.attrs.get("joint_names_json")
            if names is not None and tuple(json.loads(names)) != JOINT_NAMES:
                raise ValueError(f"{eid}: joint order mismatch")
            if reader.attrs.get("gripper_unit", "stroke_m") != "stroke_m":
                raise ValueError(f"{eid}: grippers must be in metres")
            if not np.isfinite(reader.qpos()).all():
                raise ValueError(f"{eid}: nonfinite qpos")
            if not np.isclose(reader.attrs.get("depth_scale", camera.depth_scale), camera.depth_scale):
                raise ValueError(f"{eid}: depth_scale mismatch")
            if cloud.use_color and H5_RGB not in reader.file:
                raise ValueError(f"{eid}: XYZRGB requested but RGB is absent")
            labels = reader.labels(action_source, fps)
            count = min(len(labels.indices), max_frames) if max_frames else len(labels.indices)
            episodes.append({"id": eid, "path": path, "split": meta.split, "count": count,
                             "action_source": labels.source, "notes": labels.notes})
        if max_episodes and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError("No eligible episodes")
    # Honor explicit validation episodes. Otherwise deterministic episode-level split.
    if val_fraction and not any(e["split"] == "val" for e in episodes):
        if len(episodes) < 2:
            raise ValueError("Need at least two episodes for train/validation")
        ranked = sorted(episodes, key=lambda e: hashlib.sha256(f"{seed}:{e['id']}".encode()).hexdigest())
        for e in ranked[:min(len(ranked) - 1, max(1, round(len(ranked) * val_fraction)))]:
            e["split"] = "val"
    if not any(e["split"] == "train" for e in episodes):
        raise ValueError("No training episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(output), mode="w")
    total = sum(e["count"] for e in episodes)
    channels = 6 if cloud.use_color else 3
    points = store.create_dataset("data/point_cloud", shape=(total, cloud.num_points, channels),
                                  chunks=(32, cloud.num_points, channels), dtype="f4")
    states = store.create_dataset("data/state", shape=(total, 16), chunks=(256, 16), dtype="f4")
    actions = store.create_dataset("data/action", shape=(total, 16), chunks=(256, 16), dtype="f4")
    source_indices = store.create_dataset("data/source_index", shape=(total,), chunks=(256,), dtype="i8")
    metadata = {"format": "openarm_flow_v1", **geometry_metadata(camera, cloud), "fps": fps,
                "joint_names": list(JOINT_NAMES), "action_representation": "absolute_joint_position",
                "action_source_requested": action_source, "seed": seed, "complete": False,
                "raw_root": str(raw.resolve()), "skipped": skipped,
                "debug_subset": bool(max_episodes or max_frames)}
    store.attrs.update(metadata)
    offset, ends = 0, []
    for e in episodes:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{e['id']}".encode()).digest()[:8], "little"))
        with EpisodeReader(e["path"]) as reader:
            labels = reader.labels(action_source, fps)
            n = e["count"]
            for first in range(0, n, 32):
                indices = labels.indices[first:min(first + 32, n)]
                block = []
                for index in indices:
                    try:
                        block.append(project_depth(reader.depth(index), camera, cloud, rng,
                                                   reader.rgb(index) if cloud.use_color else None))
                    except ValueError as exc:
                        raise ValueError(f"{e['id']} frame {index}: {exc}") from exc
                points[offset + first:offset + first + len(block)] = np.stack(block)
            states[offset:offset + n] = labels.state[:n]
            actions[offset:offset + n] = labels.actions[:n]
            source_indices[offset:offset + n] = labels.indices[:n]
        offset += n
        ends.append(offset)
        print(f"{e['id']}: {n} frames, {e['split']}, action={e['action_source']}", flush=True)
    store.create_dataset("meta/episode_ends", data=np.asarray(ends, dtype=np.int64))
    metadata["episodes"] = [{k: v for k, v in e.items() if k != "path"} for e in episodes]
    metadata["complete"] = True
    store.attrs.update(metadata)
    return metadata


class FlowDataset(torch.utils.data.Dataset):
    def __init__(self, path, horizon, n_obs_steps, split="train"):
        self.path = str(path)
        self.root = zarr.open_group(self.path, mode="r")
        self.metadata = dict(self.root.attrs)
        if not self.metadata.get("complete"):
            raise ValueError("Incomplete conversion; rebuild dataset in a new directory")
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        ends = self.root["meta/episode_ends"][:]
        self.ranges = [(int(start), int(end)) for start, end, e in
                       zip(np.r_[0, ends[:-1]], ends, self.metadata["episodes"], strict=True)
                       if e["split"] == split]
        self.cumulative = np.cumsum([b - a for a, b in self.ranges])

    def __len__(self):
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.cumulative, index, side="right"))
        start, end = self.ranges[episode]
        previous = self.cumulative[episode - 1] if episode else 0
        now = start + int(index - previous)
        first = now - self.n_obs_steps + 1
        obs_indices = np.clip(np.arange(first, now + 1), start, end - 1)
        act_indices = np.clip(np.arange(first, first + self.horizon), start, end - 1)
        return {"obs": {"point_cloud": torch.from_numpy(self.root["data/point_cloud"].oindex[obs_indices]),
                        "agent_pos": torch.from_numpy(self.root["data/state"].oindex[obs_indices])},
                "action": torch.from_numpy(self.root["data/action"].oindex[act_indices])}

    def normalizer(self):
        """Fit only training episodes, streaming all points without loading the dataset into RAM."""
        from openarm_flow._vendor.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
        result = LinearNormalizer()
        for name, key in [("point_cloud", "point_cloud"), ("agent_pos", "state"), ("action", "action")]:
            array = self.root[f"data/{key}"]
            lo, hi, sums, squares, count = None, None, 0, 0, 0
            for begin, end in self.ranges:
                for first in range(begin, end, 128):
                    x = array[first:min(first + 128, end)].reshape(-1, array.shape[-1]).astype(np.float64)
                    lo = x.min(0) if lo is None else np.minimum(lo, x.min(0))
                    hi = x.max(0) if hi is None else np.maximum(hi, x.max(0))
                    sums, squares, count = sums + x.sum(0), squares + (x*x).sum(0), count + len(x)
            if not count:
                raise ValueError("No training samples for normalization")
            span = hi - lo
            scale = 2 / np.where(span < 1e-4, 2, span)
            offset = np.where(span < 1e-4, -lo, -1 - scale * lo)
            stats = {"min": lo, "max": hi, "mean": sums / count,
                     "std": np.sqrt(np.maximum(0, (squares - sums*sums/count) / max(1, count - 1)))}
            result[name] = SingleFieldLinearNormalizer.create_manual(
                scale.astype("f4"), offset.astype("f4"), {k: v.astype("f4") for k, v in stats.items()})
        result.requires_grad_(False)
        return result
