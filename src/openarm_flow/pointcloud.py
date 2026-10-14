"""One shared, calibrated depth -> camera-frame XYZ path for train and deployment."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 0.001
    rectified: bool = True
    frame: str = "camera_optical"
    provenance: str = "user calibration"

    def __post_init__(self):
        if any(v is None for v in [self.width, self.height, self.fx, self.fy, self.cx, self.cy, self.depth_scale]):
            raise ValueError("Fill all camera intrinsics; example null values are not calibration")
        if self.width < 1 or self.height < 1:
            raise ValueError("Camera width/height must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy, self.depth_scale]).all():
            raise ValueError("Camera intrinsics must be finite")
        if min(self.fx, self.fy, self.depth_scale) <= 0:
            raise ValueError("fx, fy and depth_scale must be positive")
        if not self.rectified or self.frame != "camera_optical":
            raise ValueError("Requires rectified depth in camera optical frame (x right, y down, z forward)")

    @classmethod
    def load(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text()))


@dataclass
class PointCloudConfig:
    num_points: int = 512
    depth_min: float = 0.2
    depth_max: float = 1.2
    min_valid: int = 32
    use_color: bool = False
    xyz_min: list | None = None
    xyz_max: list | None = None

    def __post_init__(self):
        if self.num_points < 1 or self.min_valid < 1 or not 0 < self.depth_min < self.depth_max:
            raise ValueError("Invalid point count or depth range")
        if (self.xyz_min is None) != (self.xyz_max is None):
            raise ValueError("Specify both xyz_min and xyz_max")
        if self.xyz_min is not None:
            if np.shape(self.xyz_min) != (3,) or np.shape(self.xyz_max) != (3,):
                raise ValueError("XYZ bounds must each have 3 values")
            if not np.all(np.asarray(self.xyz_min) < self.xyz_max):
                raise ValueError("xyz_min must be smaller than xyz_max")


def project_depth(depth, camera: Camera, cfg: PointCloudConfig, rng, rgb=None):
    """Return [N,3/6] float32. Drop invalid/clipped depth; sample uniformly.

    Input units are native sensor units (uint16 mm for the current recorder).
    No guessed intrinsics, no resizing, and no silent all-zero cloud fallback.
    """
    depth = np.asarray(depth)
    if depth.shape != (camera.height, camera.width):
        raise ValueError(f"Depth {depth.shape} differs from calibrated {(camera.height, camera.width)}")
    z = depth.astype(np.float32) * camera.depth_scale
    valid = np.isfinite(z) & (z > cfg.depth_min) & (z < cfg.depth_max)
    v, u = np.nonzero(valid)
    z = z[v, u]
    xyz = np.stack([(u - camera.cx) * z / camera.fx, (v - camera.cy) * z / camera.fy, z], axis=-1)
    if cfg.xyz_min is not None:
        keep = ((xyz >= cfg.xyz_min) & (xyz <= cfg.xyz_max)).all(axis=-1)
        xyz, u, v = xyz[keep], u[keep], v[keep]
    if len(xyz) < cfg.min_valid:
        raise ValueError(f"Only {len(xyz)} valid points after crop; need {cfg.min_valid}")
    if cfg.use_color:
        if rgb is None or np.shape(rgb) != (camera.height, camera.width, 3):
            raise ValueError("XYZRGB requires RGB aligned with depth at the calibrated resolution")
        xyz = np.concatenate([xyz, np.asarray(rgb)[v, u].astype(np.float32) / 255], axis=-1)
    indices = rng.choice(len(xyz), cfg.num_points, replace=len(xyz) < cfg.num_points)
    return xyz[indices].astype(np.float32)


def geometry_metadata(camera, cfg):
    return {"camera": asdict(camera), "pointcloud": asdict(cfg)}
