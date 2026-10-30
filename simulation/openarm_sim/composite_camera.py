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
                 edge_jump_m: float = 0.03, tag: str = "", dynamic_depth_scale: float = 1.0,
                 color_gains: dict | None = None, blur_sigma: float = 0.0, depth_dropout: dict | None = None):
        self.name, (self.W, self.H) = name, size
        # The real D435 reads the ROBOT ~19 mm farther than its URDF render (scripts/sim_realism.py). Only the
        # reported depth of robot pixels is scaled: the occlusion test uses the geometric depth, otherwise a thin
        # object (the 16 mm tin) is pushed behind the real table background and disappears from the image.
        self.dynamic_depth_scale = dynamic_depth_scale
        self.robot_geoms = None
        # Appearance of the rendered layer (scripts/compare_sim_images.py measures each): per-channel gains of the
        # robot and of the task objects ({"robot": [r, g, b], "objects": [r, g, b], "robot_contrast": c}; the real
        # cameras see the robot with a colour cast and less contrast than the fixed render lighting gives), blur of
        # the rendered pixels and their outline (the real RGB is softer than the aliased render), and depth missing
        # in patches on the robot and the tin ({"robot": [p at black, p at white], "objects": p}, on top of range
        # and edge losses; the D435 loses the black and the shiny parts much more than a clean render).
        color_gains = dict(color_gains or {})
        self.robot_contrast = float(color_gains.pop("robot_contrast", 1.0))
        self.gains = {k: np.asarray(v, np.float32) for k, v in color_gains.items()}
        self.blur_sigma, self.depth_dropout = blur_sigma, depth_dropout or {}
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

    def match_color(self, reference_rgb: np.ndarray) -> np.ndarray:
        """Khớp histogram từng kênh của nền với ảnh thật tham chiếu (vd khung đầu episode): camera thật tự cân bằng
        trắng/phơi sáng theo thời gian, còn nền là trung vị cả episode. Ánh xạ theo phân vị nên vùng bão hoà (giấy
        trắng) vẫn trắng. Trả về hệ số trung vị mỗi kênh (để ghi log)."""
        ref = np.asarray(reference_rgb).reshape(-1, 3)
        bg = self.bg_rgb.reshape(-1, 3)
        quantiles = np.linspace(0, 1, 257)
        out = np.empty_like(bg)
        for ch in range(3):
            src_q = np.quantile(bg[:, ch], quantiles)
            ref_q = np.quantile(ref[:, ch], quantiles)
            lut = np.interp(np.arange(256), src_q + np.arange(257) * 1e-6, ref_q)  # src_q phải tăng chặt
            out[:, ch] = np.clip(np.rint(lut[bg[:, ch]]), 0, 255)
        gain = np.median(ref, 0) / np.maximum(np.median(bg, 0), 1.0)
        self.bg_rgb = out.reshape(self.bg_rgb.shape)
        return gain

    @classmethod
    def from_config(cls, m: mujoco.MjModel, name: str, background_dir: str | Path, composite_cfg: dict | None,
                    **kwargs) -> "CompositeCamera":
        """Camera `name` with the depth scale and appearance of the scene config's `composite` section."""
        composite_cfg = composite_cfg or {}
        for key in ("dynamic_depth_scale", "blur_sigma"):
            if key in composite_cfg:
                kwargs.setdefault(key, composite_cfg[key])
        for key in ("color_gains", "depth_dropout"):   # per camera
            if name in composite_cfg.get(key, {}):
                kwargs.setdefault(key, composite_cfg[key][name])
        return cls(m, name, background_dir, **kwargs)

    def _init_render(self, m, rng, noise, noise_mm_at_0p6, edge_jump_m):
        self._model = m
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

    def _robot_geoms(self) -> np.ndarray:
        if self.robot_geoms is None:
            m = self._model
            free = {b for b in range(m.nbody) if m.body_jntnum[b] and
                    m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}
            self.robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] not in free])
        return self.robot_geoms

    def _dropout(self, robot: np.ndarray, objects: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Pixels whose depth goes missing: a smooth random field (blobs of ~10 px, like the D435's holes)
        thresholded at the per-pixel probability."""
        field = cv2.GaussianBlur(self.rng.normal(0, 1, robot.shape).astype(np.float32), (0, 0), 4.0)
        u = 1 / (1 + np.exp(-1.702 * field / max(float(field.std()), 1e-6)))   # ~uniform on [0, 1]
        p = np.zeros(robot.shape, np.float32)
        if "robot" in self.depth_dropout:
            dark, bright = self.depth_dropout["robot"]
            p = np.where(robot, dark + (bright - dark) * rgb.mean(-1) / 255, p)
        p = np.where(objects, self.depth_dropout.get("objects", 0.0), p)
        return u < p

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
        robot = np.isin(seg, self._robot_geoms())   # geoms of free bodies (task objects) are not robot
        if self.dynamic_depth_scale != 1.0:   # task objects keep their geometric depth
            z = np.where(robot, z * self.dynamic_depth_scale, z)
        if self.robot_contrast != 1.0 and robot.any():   # about the robot's mean colour in this frame
            pivot = rgb[robot].mean(0)
            rgb = np.where(robot[..., None], pivot + self.robot_contrast * (rgb - pivot), rgb)
        if self.gains:
            ones = np.ones(3, np.float32)
            gain = np.where(robot[..., None], self.gains.get("robot", ones), self.gains.get("objects", ones))
            rgb = rgb * gain
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
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
        if self.noise and self.depth_dropout and front.any():
            valid &= ~self._dropout(front & robot, front & ~robot, rgb)
        depth = np.where(valid, np.rint(depth_m * 1000), 0).astype(np.uint16)
        out_rgb = np.where(front[..., None], rgb, self.bg_rgb)
        if self.blur_sigma > 0 and front.any():   # rendered pixels and their outline only; the background is real
            region = cv2.dilate(front.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            out_rgb = np.where(region[..., None], cv2.GaussianBlur(out_rgb, (0, 0), self.blur_sigma), out_rgb)
        # geom id của vật động đang hiện (trước nền) ở mỗi pixel, -1 = nền thật; dùng để so vùng sim/thật.
        self.last_segmentation = np.where(front, seg, -1)
        return out_rgb, depth

    def close(self) -> None:
        self.r.close()
