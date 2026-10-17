"""Camera ngực D435 như dữ liệu thật ghi lại: render ở độ phân giải gốc rồi thu nhỏ đúng cách `prepare_real_data.py`.

Dữ liệu thật: ảnh màu 640x480 là khung 16:9 (1280x720, FOV dọc 42°) của D435 nén ngang thành 4:3 (pixel không vuông);
depth đã căn theo ảnh màu (uint16, mm; 0 = không hợp lệ, dải 0.2–1.2 m). RGB thu nhỏ bằng lọc hộp, depth lấy mẫu gần nhất. Mô hình nhiễu depth lấy từ đo trên dữ liệu thật
(2026-10-08): nhiễu thời gian ~1 mm ở 0.6 m, ~0.2% điểm lỗi trên mặt hộp, 15–25% điểm lỗi ở dải biên trái/phải
và một dải lỗi mảnh dọc các mép vật (nơi độ sâu nhảy).
"""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image

from .render import OffscreenRenderer
from .scene import BACKDROP_GROUP, BACKGROUND_GEOM, BACKGROUND_GROUP

NATIVE = (1280, 720)       # độ phân giải gốc của camera sim (16:9)
DATASET = (640, 480)       # khung 4:3 trong HDF5 thật (chỉ dùng để áp mô hình nhiễu depth)
DEPTH_RANGE_M = (0.2, 1.2)


def d435_depth_model(z_m: np.ndarray, rng: np.random.Generator, noise_mm_at_0p6: float = 1.0,
                     edge_jump_m: float = 0.03, border_frac: float = 0.09, border_dropout: float = 0.18) -> np.ndarray:
    """Độ sâu lý tưởng (m) → độ sâu kiểu D435 (uint16 mm, 0 = không hợp lệ)."""
    z = z_m.astype(np.float32)
    z = z + rng.normal(0.0, noise_mm_at_0p6 * 1e-3, z.shape).astype(np.float32) * (z / 0.6) ** 2
    z = np.clip(z, *DEPTH_RANGE_M)       # D435 thật đã cắt về dải 0.2–1.2 m (không để 0)
    bad = np.zeros(z.shape, bool)
    # mép nhảy độ sâu: dải lỗi ~2 px quanh các điểm khác biệt lớn với láng giềng
    jump = np.zeros(z.shape, bool)
    jump[:, 1:] |= np.abs(np.diff(z, axis=1)) > edge_jump_m
    jump[1:, :] |= np.abs(np.diff(z, axis=0)) > edge_jump_m
    for _ in range(2):
        jump = jump | np.roll(jump, 1, 0) | np.roll(jump, 1, 1)
    w = z.shape[1]
    cols = np.zeros(w, bool)
    k = int(w * border_frac)
    cols[:k] = cols[-k:] = True
    border = cols[None, :] & (rng.random(z.shape) < border_dropout)
    out = np.where(bad | jump | border, 0.0, z) * 1000.0
    return np.clip(np.round(out), 0, 65535).astype(np.uint16)


class ChestCamera:
    def __init__(self, m: mujoco.MjModel, camera: str, out_size: tuple[int, int] = (192, 144),
                 rng: np.random.Generator | None = None, depth_model: bool = True,
                 use_background: bool = True, native_size: tuple[int, int] = NATIVE):
        self.camera, self.out = camera, out_size
        self.rng = rng or np.random.default_rng(0)
        self.depth_model = depth_model
        self._r = OffscreenRenderer(m, *native_size)
        self._opt = mujoco.MjvOption()
        if use_background:
            try:
                m.geom(BACKGROUND_GEOM)
                # có phông nền ảnh thật: bật nó, ẩn bàn/sàn sim (đứng trước phông)
                self._opt.geomgroup[BACKGROUND_GROUP] = 1
                self._opt.geomgroup[BACKDROP_GROUP] = 0
            except KeyError:
                pass
        else:
            self._opt.geomgroup[BACKGROUND_GROUP] = 0
        self._yd = ((np.arange(DATASET[1]) + 0.5) * native_size[1] / DATASET[1]).astype(int)
        self._xd = ((np.arange(DATASET[0]) + 0.5) * native_size[0] / DATASET[0]).astype(int)
        self._yi = ((np.arange(out_size[1]) + 0.5) * DATASET[1] / out_size[1]).astype(int)
        self._xi = ((np.arange(out_size[0]) + 0.5) * DATASET[0] / out_size[0]).astype(int)

    def observe(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8 H×W×3, depth uint16 H×W mm) ở kích thước ra."""
        rgb = self._r.rgb(d, self.camera, self._opt)
        z = self._r.depth(d, self.camera, self._opt)[self._yd][:, self._xd]          # lấy mẫu về khung 640x480 như dữ liệu thật
        depth = d435_depth_model(z, self.rng) if self.depth_model else \
            np.clip(np.round(z * 1000.0), 0, 65535).astype(np.uint16)
        rgb = np.asarray(Image.fromarray(rgb).resize(self.out, Image.BOX))
        return rgb, depth[self._yi][:, self._xi]

    def close(self) -> None:
        self._r.close()


class CalibratedCamera:
    """Render the release's pinhole rays without the legacy 16:9 resize.

    Depth-only releases use the depth optical pose. XYZRGB releases assume
    depth aligned to RGB and use the color optical pose with RGB intrinsics.
    """

    def __init__(self, m, camera, calibration):
        self.camera = camera
        c = calibration
        width, height = c.width, c.height
        # Render a slightly wider centered view, then sample each requested ray.
        focal = min(c.fx, c.fy) * 0.95
        m.cam_fovy[m.camera(camera).id] = np.degrees(2 * np.arctan(height / (2 * focal)))
        self._r = OffscreenRenderer(m, width, height)
        u = np.arange(width)
        v = np.arange(height)
        self._x = np.rint(width / 2 - 0.5 + (u - c.cx) * focal / c.fx).astype(int)
        self._y = np.rint(height / 2 - 0.5 + (v - c.cy) * focal / c.fy).astype(int)
        if self._x.min() < 0 or self._x.max() >= width or self._y.min() < 0 or self._y.max() >= height:
            self._r.close()
            raise ValueError("Calibration rays fall outside the render viewport")
        self.scale = c.depth_scale

    def observe(self, d):
        rgb = self._r.rgb(d, self.camera)[self._y][:, self._x]
        z = self._r.depth(d, self.camera)[self._y][:, self._x]
        # Preserve distances: far pixels must not become a fake 1.2m surface.
        valid = np.isfinite(z) & (z > 0) & (z / self.scale <= 65535)
        depth = np.where(valid, np.rint(z / self.scale), 0).clip(0, 65535).astype(np.uint16)
        return rgb, depth

    def close(self):
        self._r.close()
