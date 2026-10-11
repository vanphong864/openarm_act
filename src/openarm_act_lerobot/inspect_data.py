"""A report on the raw episodes, before anything is converted.

Answers the questions the training setup depends on: how much data there is
per task, which episodes would be skipped and why, and whether
joint_commands can serve as labels (how much of it is missing, how far the
arm lags behind it, how far the grippers sit from what they were told).
"""

from __future__ import annotations

import collections

import numpy as np
from omegaconf import DictConfig

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths
from openarm_act_lerobot.convert import plan
from openarm_act_lerobot.episode import EpisodeReader
from openarm_act_lerobot.episode import command_lag


def inspect(cfg: DictConfig) -> dict:
    planned = plan(cfg)
    rows = []
    for item in planned:
        with EpisodeReader(item.path) as episode:
            row = {"id": item.id, "frames": len(episode), "task": item.task, "skip": item.skip}
            row["seconds"] = len(episode) / episode.frequency_hz
            # Readable episodes: everything but missing data or a bad layout.
            if item.image_size is not None:
                row["image_size"] = list(item.image_size)
                qpos, commands = episode.qpos(), episode.commands()
                if commands is not None:
                    missing = ~np.isfinite(commands)
                    row["command_nan"] = {n: float(m) for n, m in zip(C.JOINT_NAMES, missing.mean(0), strict=True)}
                    row["command_lag_steps"] = command_lag(qpos, commands)
                    grippers = list(C.GRIPPERS)
                    diff = commands[:, grippers] - qpos[:, grippers]
                    row["gripper_cmd_minus_qpos_mm"] = (
                        float(1000 * np.nanmedian(diff)) if np.isfinite(diff).any() else None
                    )
                labels = episode.labels(
                    str(cfg.data.action_source), int(cfg.data.fps), float(cfg.data.max_nan_fraction)
                )
                row["action_source"] = labels.source
                row["notes"] = labels.notes
        rows.append(row)

    used = [r for r in rows if r["skip"] is None]
    by_task = collections.Counter(r["task"] for r in used)
    seconds = sum(r["seconds"] for r in used)
    lags = [v for r in used for v in r.get("command_lag_steps", {}).values() if np.isfinite(v)]
    return {
        "raw": str(paths.resolve(cfg.paths.raw)),
        "episodes": len(rows),
        "usable": len(used),
        "hours": seconds / 3600,
        "by_task": dict(by_task),
        "held_out": sum(1 for r in rows if (r["skip"] or "").startswith("split")),
        "skipped": {r["id"]: r["skip"] for r in rows if r["skip"] and not r["skip"].startswith("split")},
        "action_sources": dict(collections.Counter(r.get("action_source") for r in used)),
        "median_command_lag_steps": float(np.median(lags)) if lags else None,
        "rows": rows,
    }
