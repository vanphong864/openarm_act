"""Dual-depth conversion, episode splits, masked chunks and train-only statistics."""

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import zarr

from openarm_data.constants import JOINT_NAMES
from openarm_data.episode import EpisodeReader, episode_id, find_episodes
from openarm_data.manifest import Manifest
from openarm_flow.depth_views import DepthConfig, encode_depth, resize_depth
from openarm_flow.rgb_views import RGBConfig, resize_rgb


def grasp_phase_flags(actions, fps):
    """Supervision-only masks: 1s before and .5s after an open->closed command.

    Episode-local, separately for each gripper. Starting an episode already
    closed is not evidence of a new closing event. Never provided to the encoder.
    """
    n = len(actions)
    approach, phase = np.zeros((n, 2), bool), np.zeros((n, 2), bool)
    for side, column in enumerate((7, 15)):
        closed = actions[:, column] < .03
        events = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
        for t in events:
            start, stop = max(0, t - round(fps)), min(n, t + round(.5 * fps))
            approach[start:t, side] = True
            phase[start:stop, side] = True
    return approach, phase


def _retry(action, what, tries=4):
    """Network drives (Colab's Google Drive) drop the mount now and then: retry reads with growing waits."""
    for attempt in range(tries):
        try:
            return action()
        except OSError as exc:
            transient = exc.errno in (5, 103, 104, 107, 110) or "Transport endpoint" in str(exc)
            if not transient or attempt == tries - 1:   # a truncated/corrupt file will not heal by waiting
                raise
            wait = 5 * 3 ** attempt
            print(f"RETRY {what}: {exc} (attempt {attempt + 1}/{tries}, waiting {wait}s)", flush=True)
            time.sleep(wait)


def convert_depth(raw, output, cfg=DepthConfig(), *, manifest=None, action_source="commands",
                  val_fraction=0.15, seed=42, max_episodes=None, max_frames=None,
                  skip_invalid=False, episode_names=None, rgb: RGBConfig | None = None, all_train=False,
                  objects=None, stage_dir=None, max_skip_fraction=0.1):
    """objects: an object_features.ObjectConfig to add per-frame object-centric features (scene estimated once
    from the first recorded frame of each episode, as at deployment)."""
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if not all_train and not 0 < val_fraction < 1:
        raise ValueError("A held-out episode split is required: 0 < val_fraction < 1")
    if any(v is not None and v < 1 for v in (max_episodes, max_frames)):
        raise ValueError("Debug limits must be positive")
    manifest = Manifest.load(manifest)
    episodes, skipped, source_shapes = [], [], None
    paths = find_episodes(raw)
    if episode_names:
        wanted = set(episode_names)
        paths = [p for p in paths if p.name in wanted]
        if {p.name for p in paths} != wanted:
            raise ValueError("Some requested episode filenames were not found")
    for path in paths:
        eid = episode_id(path, raw)
        try:
            _retry(lambda: EpisodeReader(path).close(), f"open {eid}")
            with EpisodeReader(path) as reader:
                meta = manifest.resolve(eid, reader.task)
                if meta.exclude:
                    skipped.append({"id": eid, "reason": "manifest exclude"})
                    continue
                f = reader.file
                if not reader.attrs.get("complete") or reader.frequency_hz != 50:
                    raise ValueError("Expected complete 50 Hz recording")
                if reader.qpos().shape != (len(reader), 16) or not np.isfinite(reader.qpos()).all():
                    raise ValueError("Expected finite qpos [T,16]")
                names = reader.attrs.get("joint_names_json")
                if names is not None and tuple(json.loads(names)) != JOINT_NAMES:
                    raise ValueError("Joint order mismatch")
                if reader.attrs.get("gripper_unit", "stroke_m") != "stroke_m":
                    raise ValueError("Grippers must use metres")
                if not np.isclose(reader.attrs.get("depth_scale", cfg.depth_scale), cfg.depth_scale):
                    raise ValueError("Depth scale mismatch")
                shapes = {}
                for camera in cfg.cameras:
                    d = f[f"observations/images/{camera}"]
                    if d.ndim != 3 or d.shape[0] != len(reader) or d.dtype != np.uint16:
                        raise ValueError(f"{camera}: expected uint16 [T,H,W]")
                    shapes[camera] = list(d.shape[1:])
                    resize_depth(d[0], cfg, camera)
                if rgb is not None:
                    for camera in rgb.cameras:
                        d = f[f"observations/images/{camera}"]
                        if d.ndim != 4 or d.shape[0] != len(reader) or d.shape[-1] != 3 or d.dtype != np.uint8:
                            raise ValueError(f"{camera}: expected uint8 [T,H,W,3]")
                        shapes[camera] = list(d.shape[1:3])
                        resize_rgb(d[0], rgb, camera)
                if source_shapes is not None and shapes != source_shapes:
                    raise ValueError("Source camera resolution changed between episodes")
                labels = reader.labels(action_source, 50)
                n = min(len(labels.indices), max_frames) if max_frames else len(labels.indices)
                episodes.append({"id": eid, "path": path, "split": meta.split, "count": n,
                                 "action_source": labels.source, "notes": labels.notes})
                source_shapes = shapes
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            if not skip_invalid:
                raise ValueError(f"{eid}: {exc}. Use --skip-invalid to record and skip this episode") from exc
            skipped.append({"id": eid, "reason": str(exc)})
            print(f"SKIP {eid}: {exc}", flush=True)
        if max_episodes and len(episodes) >= max_episodes:
            break
    if all_train:  # extra training data (e.g. simulated demos): never validated on
        for e in episodes:
            e["split"] = "train"
    elif len(episodes) < 2:
        raise ValueError("Need at least two readable episodes for train/validation")
    if not episodes:
        raise ValueError("No readable episodes")
    if not all_train and not any(e["split"] == "val" for e in episodes):
        ranked = sorted(episodes, key=lambda e: hashlib.sha256(f"{seed}:{e['id']}".encode()).hexdigest())
        for e in ranked[:min(len(ranked) - 1, max(1, round(len(ranked) * val_fraction)))]:
            e["split"] = "val"
    if not any(e["split"] == "train" for e in episodes):
        raise ValueError("No training episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(output), mode="w")
    metadata = {"format": "openarm_meanflow_depth_v1", "depth": cfg.metadata(), "fps": 50,
                "source_shapes": source_shapes, "joint_names": list(JOINT_NAMES),
                "action_representation": "absolute_joint_position", "action_source_requested": action_source,
                "raw_root": str(raw.resolve()), "seed": seed, "complete": False, "skipped": skipped,
                "debug_subset": bool(max_episodes or max_frames or episode_names)}
    if all_train:
        metadata["all_train"] = True
    if rgb is not None:
        metadata["rgb"] = rgb.metadata()
    estimator = features = None
    if objects is not None:
        from openarm_flow.object_features import FEATURE_NAMES, ObjectFeatures, SceneEstimator
        estimator, features = SceneEstimator(objects), ObjectFeatures(objects)
        metadata["objects"] = {"config": objects.metadata(), "features": list(FEATURE_NAMES),
                               "scene": "estimated once from raw frame 0 of each episode"}
    store.attrs.update(metadata)
    total = sum(e["count"] for e in episodes)
    depth = store.create_dataset("data/depth", shape=(total, 2, cfg.height, cfg.width),
                                 chunks=(8, 2, cfg.height, cfg.width), dtype="u2")
    rgb_data = None
    if rgb is not None:
        rgb_data = store.create_dataset("data/rgb", shape=(total, 2, 3, rgb.height, rgb.width),
                                        chunks=(8, 2, 3, rgb.height, rgb.width), dtype="u1")
    states = store.create_dataset("data/state", shape=(total, 16), chunks=(256, 16), dtype="f4")
    object_data = None
    if features is not None:
        from openarm_flow.object_features import FEATURE_DIM
        object_data = store.create_dataset("data/objects", shape=(total, FEATURE_DIM), chunks=(256, FEATURE_DIM),
                                           dtype="f4")
    actions = store.create_dataset("data/action", shape=(total, 16), chunks=(256, 16), dtype="f4")
    indices = store.create_dataset("data/source_index", shape=(total,), chunks=(256,), dtype="i8")
    offset, ends, accepted = 0, [], []
    for e in episodes:
        local = e["path"]
        try:
            if stage_dir is not None:      # one episode at a time on local disk: raw data can be far larger than it
                Path(stage_dir).mkdir(parents=True, exist_ok=True)
                local = Path(stage_dir) / Path(e["path"]).name
                _retry(lambda: shutil.copyfile(e["path"], local), f"copy {e['id']}")
            with EpisodeReader(local) as reader:
                labels = reader.labels(action_source, 50)
                n = e["count"]
                for first in range(0, n, 8):
                    block = [np.stack([resize_depth(reader.file[f"observations/images/{c}"][int(i)], cfg, c)
                                      for c in cfg.cameras]) for i in labels.indices[first:min(first + 8, n)]]
                    depth[offset + first:offset + first + len(block)] = np.stack(block)
                    if rgb_data is not None:
                        rgb_data[offset + first:offset + first + len(block)] = np.stack(
                            [np.stack([resize_rgb(reader.file[f"observations/images/{c}"][int(i)], rgb, c)
                                       for c in rgb.cameras]) for i in labels.indices[first:min(first + 8, n)]])
                states[offset:offset + n] = labels.state[:n]
                if object_data is not None:
                    scene = estimator.estimate({c: reader.file[f"observations/images/{c}_rgb"][0]
                                                for c in objects.cameras})
                    object_data[offset:offset + n] = features(scene, labels.state[:n])
                    e["scene"] = scene
                actions[offset:offset + n] = labels.actions[:n]
                indices[offset:offset + n] = labels.indices[:n]
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            if not skip_invalid:
                raise ValueError(f"{e['id']}: conversion failed: {exc}") from exc
            skipped.append({"id": e["id"], "reason": f"frame read: {exc}"})
            print(f"SKIP {e['id']}: frame read: {exc}", flush=True)
            continue
        finally:
            if stage_dir is not None and Path(local) != Path(e["path"]):
                Path(local).unlink(missing_ok=True)
        offset += n
        ends.append(offset)
        accepted.append({k: v for k, v in e.items() if k != "path"})
        print(f"{e['id']}: {n} frames, {e['split']}, action={e['action_source']}", flush=True)
    unreadable = [s for s in skipped if s["reason"] != "manifest exclude"]
    if len(unreadable) > max_skip_fraction * len(paths):
        raise ValueError(f"{len(unreadable)} of {len(paths)} episodes could not be read (first: {unreadable[0]['reason'][:160]}). "
                         "On Colab remount Drive (drive.mount(..., force_remount=True)) and convert again.")
    if not all_train and {e["split"] for e in accepted} != {"train", "val"}:
        raise ValueError("Unreadable frames removed an entire split; rebuild with another split/seed")
    for array in (depth, states, actions, indices, *(a for a in (rgb_data, object_data) if a is not None)):
        array.resize((offset, *array.shape[1:]))
    store.create_dataset("meta/episode_ends", data=np.asarray(ends, dtype=np.int64))
    metadata.update(episodes=accepted, skipped=skipped, complete=True)
    store.attrs.update(metadata)
    return metadata


class MeanFlowDataset(torch.utils.data.Dataset):
    def __init__(self, path, horizon, n_obs_steps, split="train"):
        self.root = zarr.open_group(str(path), mode="r")
        self.metadata = dict(self.root.attrs)
        if self.metadata.get("format") != "openarm_meanflow_depth_v1" or not self.metadata.get("complete"):
            raise ValueError("Expected complete dual-depth MeanFlow dataset")
        if split not in ("train", "val") or min(horizon, n_obs_steps) < 1:
            raise ValueError("Invalid split/horizon/history")
        self.cfg = DepthConfig(**self.metadata["depth"])
        self.has_rgb = bool(self.metadata.get("rgb"))
        self.has_objects = bool(self.metadata.get("objects"))
        self.horizon, self.n_obs_steps, self.split = horizon, n_obs_steps, split
        ends = self.root["meta/episode_ends"][:]
        self.ranges = [(int(a), int(b)) for a, b, e in zip(np.r_[0, ends[:-1]], ends,
                        self.metadata["episodes"], strict=True) if e["split"] == split]
        self.cumulative = np.cumsum([b - a for a, b in self.ranges])
        self.phase_flags = [grasp_phase_flags(self.root["data/action"][a:b], self.metadata["fps"])
                            for a, b in self.ranges]

    def __len__(self):
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = int(np.searchsorted(self.cumulative, index, side="right"))
        start, end = self.ranges[ep]
        now = start + int(index - (self.cumulative[ep - 1] if ep else 0))
        obs_ids = np.clip(np.arange(now - self.n_obs_steps + 1, now + 1), start, end - 1)
        # Chunk starts at the current observation, with NO past-action prefix.
        ids = np.arange(now, now + self.horizon)
        actions = self.root["data/action"].oindex[np.minimum(ids, end - 1)]
        local_ids = np.minimum(ids, end - 1) - start
        approach, phase = self.phase_flags[ep]
        valid = ids < end
        obs = {"depth": torch.from_numpy(encode_depth(self.root["data/depth"].oindex[obs_ids], self.cfg)),
               "agent_pos": torch.from_numpy(self.root["data/state"].oindex[obs_ids])}
        if self.has_rgb:
            obs["rgb"] = torch.from_numpy(self.root["data/rgb"].oindex[obs_ids])  # uint8 [T,2,3,H,W]
        if self.has_objects:
            obs["objects"] = torch.from_numpy(self.root["data/objects"].oindex[obs_ids])  # [T,F]
        return {"obs": obs,
                "action": torch.from_numpy(actions), "action_valid": torch.from_numpy(valid),
                "action_approach": torch.from_numpy(approach[local_ids] & valid[:, None]),
                "action_phase": torch.from_numpy(phase[local_ids] & valid[:, None])}

    def statistics(self, delta_actions=True):
        """State z-score and future residual scales, using ONLY unpadded train targets."""
        if self.split != "train" or not len(self):
            raise ValueError("Statistics must be fit on the nonempty training split")
        sums = {k: np.zeros(16, np.float64) for k in ("state", "action")}
        squares = {k: np.zeros(16, np.float64) for k in sums}
        counts = dict.fromkeys(sums, 0)
        for a, b in self.ranges:
            state = self.root["data/state"][a:b].astype(np.float64)
            action = self.root["data/action"][a:b].astype(np.float64)
            sums["state"] += state.sum(0)
            squares["state"] += (state**2).sum(0)
            counts["state"] += len(state)
            for k in range(min(self.horizon, len(state))):
                target = action[k:].copy()
                if delta_actions:
                    target -= state[:len(target)]
                    target[:, [7, 15]] = action[k:, [7, 15]]
                sums["action"] += target.sum(0)
                squares["action"] += (target**2).sum(0)
                counts["action"] += len(target)
        result = {}
        for key in sums:
            mean = sums[key] / counts[key]
            std = np.sqrt(np.maximum(0, squares[key] / counts[key] - mean**2))
            floor = np.full(16, 0.01 if key == "action" else 0.05)
            floor[[7, 15]] = 0.002 if key == "action" else 0.005
            result[key + "_mean"] = mean.astype("f4")
            result[key + "_scale"] = np.maximum(std, floor).astype("f4")
        if self.has_objects:
            values = np.concatenate([self.root["data/objects"][a:b] for a, b in self.ranges]).astype(np.float64)
            mean, std = values.mean(0), values.std(0)
            mean[0], std[0] = 0.0, 1.0            # "valid" flag stays 0/1 so a missing scene is visible
            result["object_mean"] = mean.astype("f4")
            result["object_scale"] = np.maximum(std, 0.01).astype("f4")
        return result

    def validation_indices(self, per_episode=64):
        """Cover all held-out episodes and their whole timelines deterministically."""
        if per_episode < 1:
            raise ValueError("per_episode must be positive")
        result, offset = [], 0
        for a, b in self.ranges:
            result.extend((offset + np.unique(np.linspace(0, b - a - 1, min(per_episode, b - a)).astype(int))).tolist())
            offset += b - a
        return result
