"""Identical train/serve preprocessing of ordered, uncalibrated depth views."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DepthConfig:
    cameras: tuple[str, ...] = ("chest_depth", "front_depth")
    height: int = 120
    width: int = 160
    depth_scale: float = 0.001
    min_m: float = 0.2
    max_m: float = 1.2
    min_coverage: float = 0.25
    min_valid_fraction: float = 0.01

    def __post_init__(self):
        if len(self.cameras) != 2 or len(set(self.cameras)) != 2:
            raise ValueError("Exactly two distinct, ordered depth cameras are required")
        if min(self.height, self.width) < 16 or not 0 < self.depth_scale:
            raise ValueError("Invalid depth resolution/scale")
        if not 0 < self.min_m < self.max_m or not 0 < self.min_coverage <= 1:
            raise ValueError("Invalid depth bounds/coverage")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("Invalid minimum valid fraction")

    def metadata(self):
        return {**asdict(self), "cameras": list(self.cameras)}


def resize_depth(raw, cfg: DepthConfig):
    """Masked area average in metres, returned as integer depth units (0 = invalid).

    Never average missing/clipped pixels into a valid surface. Camera frames stay
    separate: no intrinsics, extrinsics or metric XYZ are invented.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("Expected raw uint16 depth [H,W] in the recorded depth units")
    depth = raw.astype(np.float32) * cfg.depth_scale
    valid = np.isfinite(depth) & (depth > cfg.min_m) & (depth < cfg.max_m)
    if valid.mean() < cfg.min_valid_fraction:
        raise ValueError("Too few valid depth pixels")
    values = np.where(valid, depth, 0)
    h, w = depth.shape
    if h % cfg.height == 0 and w % cfg.width == 0:
        shape = (cfg.height, h // cfg.height, cfg.width, w // cfg.width)
        numerator = values.reshape(shape).mean((1, 3))
        coverage = valid.reshape(shape).mean((1, 3))
    else:
        pair = torch.from_numpy(np.stack([values, valid.astype(np.float32)]))[None]
        numerator, coverage = F.interpolate(pair, size=(cfg.height, cfg.width), mode="area")[0].numpy()
    reduced = numerator / np.maximum(coverage, 1e-8)
    reduced[coverage < cfg.min_coverage] = 0
    if np.mean(reduced > 0) < cfg.min_valid_fraction:
        raise ValueError("Too few valid depth pixels after resize")
    return np.rint(reduced / cfg.depth_scale).astype(np.uint16)


def encode_depth(depth, cfg: DepthConfig):
    """[...,H,W] integer depth -> [...,2,H,W] normalized depth + validity."""
    metres = np.asarray(depth, dtype=np.float32) * cfg.depth_scale
    valid = (metres > cfg.min_m) & (metres < cfg.max_m)
    scaled = np.where(valid, 2 * (metres - cfg.min_m) / (cfg.max_m - cfg.min_m) - 1, 0)
    return np.stack([scaled, valid.astype(np.float32)], axis=-3).astype(np.float32)
