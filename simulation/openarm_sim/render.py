"""Render offscreen (mặc định EGL; máy không có GPU NVIDIA vẫn chạy qua Mesa)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


class OffscreenRenderer:
    def __init__(self, m: mujoco.MjModel, width: int, height: int):
        self.m = m
        self.r = mujoco.Renderer(m, height=height, width=width)

    def rgb(self, d, camera, scene_option=None) -> np.ndarray:
        self.r.disable_depth_rendering()
        self.r.update_scene(d, camera=camera, scene_option=scene_option)
        return self.r.render().copy()

    def depth(self, d, camera, scene_option=None) -> np.ndarray:
        """Độ sâu (m) theo trục quang học, giống ảnh depth của RealSense."""
        self.r.enable_depth_rendering()
        self.r.update_scene(d, camera=camera, scene_option=scene_option)
        out = self.r.render().copy()
        self.r.disable_depth_rendering()
        return out

    def close(self):
        self.r.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def camera_resolution(m: mujoco.MjModel, name: str) -> tuple[int, int]:
    w, h = m.cam_resolution[m.camera(name).id]
    return int(w), int(h)


def save_depth(depth_m: np.ndarray, path_png16: str, path_vis: str, max_m: float = 1.5) -> None:
    """Lưu depth dạng PNG 16-bit (mm, như RealSense) và một ảnh màu để xem."""
    mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(path_png16)
    t = np.clip(depth_m / max_m, 0, 1)
    # bảng màu "turbo" rút gọn: gần = đỏ, xa = xanh
    stops = np.array([[0.19, 0.07, 0.23], [0.27, 0.51, 0.97], [0.16, 0.94, 0.62],
                      [0.98, 0.73, 0.22], [0.48, 0.02, 0.01]])[::-1]
    idx = t * (len(stops) - 1)
    lo = np.floor(idx).astype(int).clip(0, len(stops) - 2)
    f = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - f) + stops[lo + 1] * f
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path_vis)
