"""Train/serve preprocessing of the two fixed RGB views (same grid as the depth views)."""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from openarm_flow.depth_views import crop_view


@dataclass(frozen=True)
class RGBConfig:
    cameras: tuple[str, ...] = ("chest_rgb", "front_rgb")
    height: int = 120
    width: int = 160
    crops: dict = field(default_factory=dict)  # base camera name -> [y0, y1, x0, x1], same as depth

    def __post_init__(self):
        if len(self.cameras) != 2 or len(set(self.cameras)) != 2:
            raise ValueError("Exactly two distinct, ordered RGB cameras are required")
        if min(self.height, self.width) < 16:
            raise ValueError("Invalid RGB resolution")

    def metadata(self):
        return {**asdict(self), "cameras": list(self.cameras)}


def resize_rgb(raw, cfg: RGBConfig, camera: str | None = None):
    """Raw uint8 [H,W,3] -> area-averaged uint8 [3,H,W]."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[-1] != 3 or raw.dtype != np.uint8:
        raise ValueError("Expected raw uint8 RGB [H,W,3]")
    raw = crop_view(raw, cfg.crops, camera)
    h, w, _ = raw.shape
    if h % cfg.height == 0 and w % cfg.width == 0:
        shape = (cfg.height, h // cfg.height, cfg.width, w // cfg.width, 3)
        reduced = raw.reshape(shape).astype(np.float32).mean((1, 3)).transpose(2, 0, 1)
    else:
        tensor = torch.from_numpy(raw.astype(np.float32)).permute(2, 0, 1)[None]
        reduced = F.interpolate(tensor, size=(cfg.height, cfg.width), mode="area")[0].numpy()
    return np.rint(reduced).astype(np.uint8)
