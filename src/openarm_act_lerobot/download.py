"""Fetch recorded episodes from a Hugging Face dataset repo into paths.raw.

Each repo lands in its own folder, <paths.raw>/<folder>/, so episode ids
read `data_openarm_8_10/episode_20261008_152119` and several repos sit side
by side; manifest rules can match a whole repo with "data_openarm_8_10/*".
Files already there and unchanged are skipped, so running it again only
fetches what is new.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
from pathlib import Path

from omegaconf import DictConfig

from openarm_act_lerobot import paths

log = logging.getLogger(__name__)
# Left free after a download: the conversion and checkpoints need room too.
MARGIN_BYTES = 5 * 10**9


def destination(cfg: DictConfig) -> Path:
    folder = cfg.folder or str(cfg.repo_id).rstrip("/").split("/")[-1]
    return paths.resolve(cfg.paths.raw) / folder


def select(files: list[str], patterns: list[str], limit: int | None) -> list[str]:
    """The repo files to fetch: matching a pattern, sorted (oldest episode
    first, by name), the first `limit` of them when one is given."""
    chosen = sorted(f for f in files if any(fnmatch.fnmatchcase(f, p) for p in patterns))
    return chosen[:limit] if limit else chosen


def download(cfg: DictConfig) -> dict:
    # Before huggingface_hub is imported: it reads this once.
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import HfApi
    from huggingface_hub import snapshot_download

    api = HfApi()
    info = api.dataset_info(str(cfg.repo_id), revision=str(cfg.revision), files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    files = select(list(sizes), list(cfg.patterns), cfg.limit)
    if not files:
        raise FileNotFoundError(f"{cfg.repo_id}@{cfg.revision}: no files match {list(cfg.patterns)}")
    out = destination(cfg)
    total = sum(sizes[f] for f in files)
    missing = missing_files(out, files, sizes)
    needed = sum(sizes[f] for f in missing)
    log.info("%s@%s (%s): %d files, %.2f GB -> %s; %d missing, %.2f GB to fetch",
             cfg.repo_id, cfg.revision, info.sha[:8], len(files), total / 1e9, out, len(missing), needed / 1e9)

    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if needed > free - MARGIN_BYTES:
        raise OSError(f"{needed / 1e9:.1f} GB to download but {free / 1e9:.1f} GB free on {out}; "
                      f"free some space, point paths.raw elsewhere, or pass limit=N")
    snapshot_download(
        repo_id=str(cfg.repo_id),
        repo_type="dataset",
        revision=info.sha,  # exactly the commit listed above
        allow_patterns=files,
        local_dir=str(out),
        max_workers=int(cfg.max_workers),
    )
    return {"repo_id": str(cfg.repo_id), "sha": info.sha, "dir": str(out), "files": files, "bytes": total,
            "fetched": missing}


def missing_files(directory: Path, files: list[str], sizes: dict[str, int]) -> list[str]:
    """The files not on disk yet, or of another size than the repo's."""
    return [f for f in files if not (directory / f).is_file() or (directory / f).stat().st_size != sizes[f]]


def check(directory: Path, files: list[str], min_frames: int = 1) -> dict[str, list[str]]:
    """Open each downloaded episode; -> {file: problems} for the ones with any."""
    from openarm_act_lerobot.episode import EpisodeReader

    found = {}
    for name in files:
        if not name.endswith(".hdf5"):
            continue
        try:
            with EpisodeReader(directory / name) as episode:
                problems = episode.problems(min_frames, need_depth=True)
        except OSError as error:  # not HDF5, or truncated
            problems = [f"cannot open: {error}"]
        if problems:
            found[name] = problems
    return found
