"""Camera sim "ghép nền thật" cho camera cố định đã hiệu chuẩn.

Sim chỉ render robot + vật (nhóm DYNAMIC); nền tĩnh (hộp kê, ô giấy, bàn, phông phòng) lấy từ ảnh nền thật dựng
bằng scripts/make_real_backgrounds.py. Ghép theo z-buffer: pixel lấy từ sim khi sim có vật thể động gần hơn nền
(hoặc nền mất depth). Depth của vật động thêm nhiễu kiểu D435 (nhiễu tỉ lệ z², mất điểm ở mép nhảy độ sâu) để giống
dữ liệu ghi. Mọi thứ ở độ phân giải ghi 640×480 với đúng K đã hiệu chuẩn.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from .measured_scene import DYNAMIC_GROUP
from .render import OffscreenRenderer


class CompositeCamera:
    def __init__(self, m: mujoco.MjModel, name: str, background_dir: str | Path, size=(640, 480),
                 rng: np.random.Generator | None = None, noise: bool = True, noise_mm_at_0p6: float = 1.0,
                 edge_jump_m: float = 0.03, tag: str = ""):
        self.name, (self.W, self.H) = name, size
        self.dir = Path(background_dir)
        self.set_background(tag)
        self.r = None
        self._init_render(m, rng, noise, noise_mm_at_0p6, edge_jump_m)

    def set_background(self, tag: str = "", directory: str | Path | None = None) -> None:
        """Đổi ảnh nền (vd nền riêng của một episode: tag = "_episode_..."; hộp/giấy xê dịch giữa các phiên)."""
        bg = Path(directory) if directory else self.dir
        self.bg_rgb = cv2.imread(str(bg / f"background_{self.name}{tag}_rgb.png"))[..., ::-1].copy()
        self.bg_depth = cv2.imread(str(bg / f"background_{self.name}{tag}_depth.png"), cv2.IMREAD_UNCHANGED)
        if self.bg_rgb.shape[:2] != (self.H, self.W) or self.bg_depth.shape != (self.H, self.W):
            raise ValueError(f"Ảnh nền {self.name}{tag} không đúng kích thước {(self.W, self.H)}")
        # Pixel luôn bị robot thật che (0 mẫu) bị để đen; lấp RGB từ lân cận. Depth giữ 0 như cảm biến mất điểm.
        samples = bg / f"background_{self.name}{tag}_samples.png"
        if samples.exists():
            holes = (cv2.imread(str(samples), cv2.IMREAD_UNCHANGED) == 0).astype(np.uint8)
            if holes.any():
                self.bg_rgb = cv2.inpaint(self.bg_rgb, holes, 5, cv2.INPAINT_TELEA)

    def _init_render(self, m, rng, noise, noise_mm_at_0p6, edge_jump_m):
        self.r = OffscreenRenderer(m, self.W, self.H)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[DYNAMIC_GROUP] = 1
        self.opt.geomgroup[2] = 1          # mesh hiển thị của robot
        # Cảnh tĩnh mô hình (hộp kê, ô giấy, bàn): dùng làm z-buffer ở pixel mà nền thật mất depth (giấy trắng).
        self.opt_static = mujoco.MjvOption()
        self.opt_static.geomgroup[:] = 0
        from .measured_scene import STATIC_GROUP
        self.opt_static.geomgroup[STATIC_GROUP] = 1
        self.rng = rng or np.random.default_rng(0)
        self.noise, self.sigma, self.edge = noise, noise_mm_at_0p6 * 1e-3, edge_jump_m

    def observe(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8 H×W×3, depth uint16 mm H×W) như HDF5 thật."""
        rgb = self.r.rgb(d, self.name, self.opt)
        z = self.r.depth(d, self.name, self.opt).astype(np.float64)
        self.r.r.enable_segmentation_rendering()
        self.r.r.update_scene(d, camera=self.name, scene_option=self.opt)
        seg = self.r.r.render()[..., 0]
        self.r.r.disable_segmentation_rendering()
        dyn = seg >= 0
        bgz = self.bg_depth.astype(np.float64) * 1e-3
        z_static = self.r.depth(d, self.name, self.opt_static).astype(np.float64)
        occluder = np.where(bgz > 0, bgz, z_static)       # nền thật; chỗ mất depth thì dùng cảnh tĩnh mô hình
        front = dyn & (z < occluder - 0.003)
        if self.noise:
            z = z + self.rng.normal(0, 1, z.shape) * self.sigma * (z / 0.6) ** 2
            jump = np.zeros(z.shape, bool)
            zf = np.where(front, z, bgz)
            jump[:, 1:] |= np.abs(np.diff(zf, axis=1)) > self.edge
            jump[1:, :] |= np.abs(np.diff(zf, axis=0)) > self.edge
            jump = cv2.dilate(jump.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        else:
            jump = np.zeros(z.shape, bool)
        depth_m = np.where(front, z, bgz)
        valid = (depth_m > 0.2) & (depth_m < 1.2) & ~(jump & front)
        depth = np.where(valid, np.rint(depth_m * 1000), 0).astype(np.uint16)
        out_rgb = np.where(front[..., None], rgb, self.bg_rgb)
        return out_rgb, depth

    def close(self) -> None:
        self.r.close()
