"""Identical train/serve preprocessing of ordered, uncalibrated depth views."""

from dataclasses import asdict, dataclass, field

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
    crops: dict = field(default_factory=dict)  # base camera name -> [y0, y1, x0, x1]

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


def crop_view(raw, crops, camera):
    """Fixed per-camera region of interest [y0, y1, x0, x1] in recorded pixels (shared by depth and RGB)."""
    if camera is None or not crops:
        return raw
    box = crops.get(camera.removesuffix("_depth").removesuffix("_rgb"))
    if box is None:
        return raw
    y0, y1, x0, x1 = box
    if not (0 <= y0 < y1 <= raw.shape[0] and 0 <= x0 < x1 <= raw.shape[1]):
        raise ValueError(f"{camera}: crop {box} outside image {raw.shape[:2]}")
    return raw[y0:y1, x0:x1]


def resize_depth(raw, cfg: DepthConfig, camera: str | None = None):
    """Masked area average in metres, returned as integer depth units (0 = invalid).

    Never average missing/clipped pixels into a valid surface. Camera frames stay
    separate: no intrinsics, extrinsics or metric XYZ are invented.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("Expected raw uint16 depth [H,W] in the recorded depth units")
    raw = crop_view(raw, cfg.crops, camera)
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
