"""openarm_mc HDF5 episodes -> the LeRobot dataset ACT trains on.

Writes HF_LEROBOT_HOME/<data.repo_id>/ (LeRobot dataset format v3) with,
per frame:

    observation.images.<camera>  uint8 [H, W, 3]  each camera data.cameras names
                                 (chest_depth encoded as ACT scales depth)
    observation.state            float32 [16]  qpos
    action                       float32 [16]  the labels (episode.py)
    task                         from the file or the manifest, else
                                 data.fallback_task (ACT reads no prompt)

and openarm_conversion.json beside it: which episodes went in, with which
action source, and why the others did not.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_act_lerobot import constants as C
from openarm_act_lerobot import paths
from openarm_act_lerobot.episode import EpisodeReader
from openarm_act_lerobot.episode import episode_id
from openarm_act_lerobot.episode import find_episodes
from openarm_act_lerobot.images import parse_image
from openarm_act_lerobot.manifest import Manifest

log = logging.getLogger(__name__)
REPORT = "openarm_conversion.json"


@dataclasses.dataclass
class Planned:
    path: Path
    id: str
    task: str | None = None
    skip: str | None = None  # why it is left out, or None
    image_size: tuple[int, int] | None = None  # (height, width)


def used_cameras(cfg: DictConfig) -> list[str]:
    """The recorder's cameras the dataset holds, in the order ACT sees them."""
    cameras = [str(c) for c in cfg.data.cameras]
    for camera in cameras:
        if camera not in (C.CAMERA_RGB, C.CAMERA_DEPTH):
            raise ValueError(f"data.cameras: unknown camera {camera!r}; the recorder has chest, chest_depth")
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError(f"data.cameras must name each camera once, got {cameras}")
    return cameras


def plan(cfg: DictConfig) -> list[Planned]:
    """Every episode under paths.raw, with its task or the reason it is skipped."""
    raw = paths.resolve(cfg.paths.raw)
    if not raw.is_dir():
        raise FileNotFoundError(f"paths.raw: {raw} is not a directory")
    manifest_path = paths.resolve(cfg.data.manifest) if cfg.data.manifest else None
    manifest = Manifest.load(manifest_path if manifest_path and manifest_path.exists() else None)
    need_depth = C.CAMERA_DEPTH in used_cameras(cfg)

    planned = []
    for path in find_episodes(raw):
        item = Planned(path, episode_id(path, raw))
        with EpisodeReader(path) as episode:
            problems = episode.problems(int(cfg.data.min_frames), need_depth=need_depth)
            meta = manifest.resolve(item.id, episode.task)
            if not problems:
                item.image_size = episode.image_size
        item.task = meta.task
        if not item.task and not cfg.data.get("require_task", True):
            item.task = str(cfg.data.fallback_task)
        if problems:
            item.skip = "; ".join(problems)
        elif meta.exclude:
            item.skip = "excluded by the manifest" + (f" ({meta.extra['note']})" if "note" in meta.extra else "")
        elif meta.split != "train":
            item.skip = f"split {meta.split}"
        elif not item.task:
            item.skip = "NO TASK"
        planned.append(item)

    # One dataset, one frame size: data.image_size, else the most common one.
    size = dataset_image_size(cfg, planned)
    for item in planned:
        if item.skip is None and item.image_size != size:
            item.skip = f"frames are {item.image_size[1]}x{item.image_size[0]}, the dataset's {size[1]}x{size[0]}"

    # A short run: the first N usable episodes only (by id, oldest first).
    limit = cfg.data.get("max_episodes")
    if limit:
        for item in [p for p in planned if p.skip is None][int(limit):]:
            item.skip = f"beyond data.max_episodes ({int(limit)})"

    # As act_pipeline's val_split: a share of the trainable episodes held out.
    fraction = float(cfg.data.get("val_fraction") or 0.0)
    held = held_out([p.id for p in planned if p.skip is None], fraction, int(cfg.data.get("split_seed") or 0))
    for item in planned:
        if item.id in held:
            item.skip = f"split val (val_fraction {fraction:g})"
    return planned


def held_out(ids: list[str], fraction: float, seed: int = 0) -> set[str]:
    """round(fraction * n) of the episode ids, at least one, chosen by a hash
    of seed and id: the same ones every time, and an episode recorded later
    does not reshuffle the others."""
    if fraction <= 0 or len(ids) < 2:
        return set()
    count = min(len(ids) - 1, max(1, round(fraction * len(ids))))
    ranked = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
    return set(ranked[:count])


def dataset_image_size(cfg: DictConfig, planned: list[Planned]) -> tuple[int, int] | None:
    """(height, width) every converted frame has."""
    if cfg.data.get("image_size"):
        return tuple(int(v) for v in cfg.data.image_size)
    sizes = collections.Counter(p.image_size for p in planned if p.skip is None)
    return sizes.most_common(1)[0][0] if sizes else None


def features(cfg: DictConfig, image_size: tuple[int, int]) -> dict:
    mode = "video" if cfg.data.video else "image"
    names = list(C.JOINT_NAMES)
    out = {
        C.LEROBOT_STATE: {"dtype": "float32", "shape": (C.NUM_JOINTS,), "names": names},
        C.LEROBOT_ACTION: {"dtype": "float32", "shape": (C.NUM_JOINTS,), "names": names},
    }
    for camera in used_cameras(cfg):
        out[C.LEROBOT_IMAGE_PREFIX + camera] = {
            "dtype": mode,
            "shape": (*image_size, 3),
            "names": ["height", "width", "channels"],
        }
    return out


# The data settings that change what the converter writes.
CONVERSION_KEYS = ("action_source", "max_nan_fraction", "min_frames", "fps", "video", "cameras")


def signature(cfg: DictConfig, chosen: list[Planned]) -> dict:
    """What a converted dataset holds: settings, frame size, episodes and their tasks."""
    return json.loads(json.dumps({
        "data": {key: OmegaConf.to_container(cfg.data, resolve=True)[key] for key in CONVERSION_KEYS},
        "image_size": list(chosen[0].image_size),
        "episodes": [[p.id, p.task] for p in chosen],
    }))


def current_report(root: Path) -> dict | None:
    """The report of a finished conversion at root (written last), or None."""
    try:
        return json.loads((root / REPORT).read_text())
    except (OSError, ValueError):
        return None


def convert(cfg: DictConfig) -> dict:
    """Convert per cfg (conf/convert.yaml); -> the report written beside the dataset."""
    planned = plan(cfg)
    if not planned:
        raise FileNotFoundError(f"no episodes (*.hdf5) under {paths.resolve(cfg.paths.raw)}")
    untasked = [p.id for p in planned if p.skip == "NO TASK"]
    if untasked:
        raise ValueError(
            f"{len(untasked)} episodes have no task (set one in {cfg.data.manifest}): "
            + ", ".join(untasked[:10])
            + (" ..." if len(untasked) > 10 else "")
        )
    chosen = [p for p in planned if p.skip is None]
    for p in planned:
        log.info("%-48s %s", p.id, f"skip: {p.skip}" if p.skip else f"task: {p.task}")
    if not chosen:
        raise ValueError("every episode is skipped; nothing to convert")
    if cfg.dry_run:
        return {"dry_run": True, "episodes": [dataclasses.asdict(p) | {"path": str(p.path)} for p in planned]}

    # Imported here: lerobot reads HF_LEROBOT_HOME once, at import.
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    image_size = chosen[0].image_size
    root = HF_LEROBOT_HOME / str(cfg.data.repo_id)
    wanted = signature(cfg, chosen)
    if root.exists():
        report = current_report(root)
        if cfg.reuse_current and report and report.get("signature") == wanted:
            log.info("%s already holds exactly these %d episodes; kept", root, len(chosen))
            return report | {"reused": True}
        if not (cfg.overwrite or cfg.reuse_current):
            raise FileExistsError(f"{root} exists; pass overwrite=true to replace it")
        log.info("%s holds other episodes or settings; converting again", root)
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=str(cfg.data.repo_id),
        fps=int(cfg.data.fps),
        robot_type="openarm_v1_bimanual",
        features=features(cfg, image_size),
        use_videos=bool(cfg.data.video),
        image_writer_threads=int(cfg.image_writer_threads),
        image_writer_processes=int(cfg.image_writer_processes),
    )
    cameras = used_cameras(cfg)
    converted = []
    try:
        for item in chosen:
            with EpisodeReader(item.path) as episode:
                labels = episode.labels(
                    str(cfg.data.action_source), int(cfg.data.fps), float(cfg.data.max_nan_fraction)
                )
                for row, index in enumerate(labels.indices):
                    frame = {
                        C.LEROBOT_STATE: labels.state[row],
                        C.LEROBOT_ACTION: labels.actions[row],
                        C.LEROBOT_TASK: item.task,
                    }
                    for camera in cameras:
                        frame[C.LEROBOT_IMAGE_PREFIX + camera] = parse_image(episode.image(camera, index))
                    dataset.add_frame(frame)
            dataset.save_episode()
            converted.append({"id": item.id, "task": item.task, "frames": len(labels.indices),
                              "action_source": labels.source, "notes": labels.notes})
            log.info("%s: %d frames, actions from %s %s", item.id, len(labels.indices), labels.source, labels.notes or "")
    finally:
        # Without it the parquet files lack their footers and the dataset cannot be read.
        dataset.finalize()

    report = {
        "repo_id": str(cfg.data.repo_id),
        "root": str(root),
        "fps": int(cfg.data.fps),
        "image_size": list(image_size),
        "action_source": str(cfg.data.action_source),
        "cameras": cameras,
        "episodes": converted,
        "skipped": [{"id": p.id, "reason": p.skip} for p in planned if p.skip],
        "frames": sum(e["frames"] for e in converted),
        "signature": wanted,
    }
    (root / REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
