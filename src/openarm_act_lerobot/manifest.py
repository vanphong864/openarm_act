"""Per-episode task, split and exclusions, from a YAML manifest.

The recorder does not write the task yet, so the manifest says it:

    defaults:                 # every episode starts from these
      task: "put the red block in the bowl"
    rules:                    # in order, later rules win; glob on the episode id
      - match: "2026-10-08/*"
        task: "hand the cup over"
      - match: "2026-10-09/episode_20261009_10*"
        exclude: true
        note: "camera stalled"
      - match: "2026-10-10/*"
        split: val            # held out for offline_eval, not converted
    episodes:                 # one episode, by id; wins over everything
      2026-10-09/episode_20261009_111502:
        success: false

The episode id is the path under the dataset root without `.hdf5`. A task
in the file itself (attrs["task"]) wins over defaults and rules, and loses
to an `episodes:` entry that names one -- the manifest is where a wrong one
gets corrected. Keys other than task / exclude / split are kept as metadata.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "val")
_KNOWN = ("task", "exclude", "split")


@dataclasses.dataclass
class EpisodeMeta:
    task: str | None = None
    exclude: bool = False
    split: str = "train"
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


class Manifest:
    def __init__(self, data: dict | None = None):
        data = data or {}
        unknown = set(data) - {"defaults", "rules", "episodes"}
        if unknown:
            raise ValueError(f"manifest: unknown sections {sorted(unknown)}")
        self.defaults = dict(data.get("defaults") or {})
        self.rules = list(data.get("rules") or [])
        self.episodes = {str(k): dict(v or {}) for k, v in (data.get("episodes") or {}).items()}
        for rule in self.rules:
            if "match" not in rule:
                raise ValueError(f"manifest: rule without 'match': {rule}")

    @classmethod
    def load(cls, path: str | Path | None) -> Manifest:
        if not path:
            return cls()
        return cls(yaml.safe_load(Path(path).read_text()))

    def resolve(self, episode_id: str, file_task: str | None = None) -> EpisodeMeta:
        values = dict(self.defaults)
        for rule in self.rules:
            if fnmatch.fnmatchcase(episode_id, str(rule["match"])):
                values.update({k: v for k, v in rule.items() if k != "match"})
        if file_task:
            values["task"] = file_task
        values.update(self.episodes.get(episode_id, {}))

        split = str(values.get("split", "train"))
        if split not in SPLITS:
            raise ValueError(f"manifest: {episode_id}: split must be one of {SPLITS}, not {split!r}")
        task = values.get("task")
        task = None if task is None else (str(task).strip() or None)
        return EpisodeMeta(
            task=task,
            exclude=bool(values.get("exclude", False)),
            split=split,
            extra={k: v for k, v in values.items() if k not in _KNOWN},
        )
