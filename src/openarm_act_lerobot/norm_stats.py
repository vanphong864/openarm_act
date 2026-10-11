"""Normalisation stats for state and actions, as the model sees them.

ACT normalises state and actions with mean/std (MEAN_STD). LeRobot's own
meta/stats.json holds them per frame, for absolute actions. With
data.delta_actions the network instead sees each chunk's arm joints relative
to the state the chunk starts from, action[t + k] - state[t], whose spread
grows with k: those stats have to come from whole chunks. This computes
both kinds the same way, from the dataset's parquet files only (no video
decoding), so the step takes seconds:

    state    every frame's qpos
    actions  every (t, k) of every chunk, k < action horizon, clipped at the
             episode's end the way training masks padded steps out

Written to assets/h<H>_<absolute|delta>/<repo_id>/stats.json, shared by
every run on the same data, horizon and delta setting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths

log = logging.getLogger(__name__)
STATS_FILE = "stats.json"


def norm_stats_root(cfg: DictConfig) -> Path:
    """Shared by every model and fine-tuning mode on this data pipeline."""
    kind = "delta" if cfg.data.delta_actions else "absolute"
    return paths.resolve(cfg.paths.assets) / f"h{cfg.model.chunk_size}_{kind}"


def norm_stats_path(cfg: DictConfig) -> Path:
    return norm_stats_root(cfg) / str(cfg.data.repo_id) / STATS_FILE


def dataset_root(cfg: DictConfig) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / str(cfg.data.repo_id)


def read_episodes(root: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(state [T, 16], action [T, 16])] per episode, in frame order."""
    import pyarrow.parquet as pq

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}; convert the dataset first")
    columns = ["episode_index", "frame_index", C.LEROBOT_STATE, C.LEROBOT_ACTION]
    tables = [pq.read_table(f, columns=columns).to_pydict() for f in files]
    episode = np.concatenate([np.asarray(t["episode_index"]) for t in tables])
    frame = np.concatenate([np.asarray(t["frame_index"]) for t in tables])
    state = np.concatenate([np.asarray(t[C.LEROBOT_STATE], dtype=np.float64) for t in tables])
    action = np.concatenate([np.asarray(t[C.LEROBOT_ACTION], dtype=np.float64) for t in tables])
    order = np.lexsort((frame, episode))
    episode, state, action = episode[order], state[order], action[order]
    return [(state[episode == e], action[episode == e]) for e in np.unique(episode)]


def chunk_actions(state: np.ndarray, action: np.ndarray, horizon: int, delta: bool) -> np.ndarray:
    """Every in-episode (t, k) target the network is trained on -> [N, 16]."""
    mask = np.asarray(C.DELTA_MASK, dtype=np.float64)
    out = []
    for k in range(min(horizon, len(action))):
        target = action[k:]
        if delta:
            target = target - state[: len(target)] * mask
        out.append(target)
    return np.concatenate(out)


def describe(values: np.ndarray) -> dict:
    """LeRobot's stats layout; the network only uses mean and std."""
    return {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [len(values)],
    }


def compute_norm_stats(cfg: DictConfig) -> Path:
    episodes = read_episodes(dataset_root(cfg))
    horizon, delta = int(cfg.model.chunk_size), bool(cfg.data.delta_actions)
    state = np.concatenate([s for s, _ in episodes])
    actions = np.concatenate([chunk_actions(s, a, horizon, delta) for s, a in episodes])
    stats = {C.LEROBOT_STATE: describe(state), C.LEROBOT_ACTION: describe(actions)}
    for key, values in stats.items():
        tiny = [C.JOINT_NAMES[i] for i, s in enumerate(values["std"]) if s < 1e-6]
        if tiny:
            log.warning("%s: no spread in %s; normalising divides by ~0 there", key, ", ".join(tiny))
    out = norm_stats_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    log.info("%d episodes, %d frames, %d action targets -> %s", len(episodes), len(state), len(actions), out)
    return out


def load_norm_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(Path(path).read_text())
    return {key: {k: np.asarray(v, dtype=np.float32) for k, v in values.items()} for key, values in raw.items()}
