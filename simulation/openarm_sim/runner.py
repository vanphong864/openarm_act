"""Vòng mô phỏng dùng chung cho sinh demo, replay và môi trường RL."""

from __future__ import annotations

import mujoco
import numpy as np

from .arm import GRIP_OPEN, SIDES, Arm
from .policy import NOMINAL_Q, choose_orientation
from .task import set_object_poses


class Sim:
    def __init__(self, m: mujoco.MjModel, cfg: dict):
        self.m, self.cfg = m, cfg
        self.d = mujoco.MjData(m)
        self.arms = {s: Arm(m, s, cfg["task"].get("ik_margin")) for s in SIDES}
        self.fps = cfg["task"]["fps"]
        self.substeps = max(1, round(1.0 / (self.fps * m.opt.timestep)))
        self.object_names = list((cfg.get("objects") or {}).keys())
        self.home = {s: self._home_q(s) for s in SIDES}
        # Scene đo đạc: offset zero khớp (encoder thật → khớp mô hình), hiệu chuẩn từ camera
        # (scripts/calibrate_joint_offsets.py). Giao diện ngoài (reset/step/qpos16) giữ đơn vị encoder thật.
        self.offset = np.zeros(16)
        jo = (cfg.get("_measured") or {}).get("joint_offsets")
        if jo:
            for i, s in enumerate(SIDES):
                self.offset[8 * i:8 * i + 7] = jo["arms"][s]["offsets_rad"]
        # Scene đo đạc: lệnh đi qua bộ giới hạn tốc độ như phía robot thật (openarm_sim/actuation.py).
        self.limiter = None
        act = (cfg.get("_measured") or {}).get("actuators")
        if act:
            from .actuation import RateLimitedTargets, vmax16
            self.limiter = RateLimitedTargets(vmax16(act["params"]), np.concatenate([self.home[s] for s in SIDES]))

    def _home_q(self, side: str) -> np.ndarray:
        hq = self.cfg["task"].get("home_qpos")
        if hq is not None:
            # tư thế khớp cho sẵn (16 giá trị [trái 8 | phải 8]), vd tư thế buông tay của dữ liệu thật
            i = SIDES.index(side)
            return np.asarray(hq[8 * i:8 * i + 8], float)
        x, y, z = self.cfg["task"]["home_tcp"]
        pos = [x, abs(y) if side == "left" else -abs(y), z]
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("zero").id)
        # hướng kẹp thẳng đứng, yaw sao cho mọi khớp xa giới hạn nhất
        _, _, q = choose_orientation(self.arms[side], self.d, pos, pos, list(np.linspace(-np.pi / 2, np.pi / 2, 13)),   # hướng home tuỳ ý: quét mịn để chọn lề tốt nhất
                                     
                                     seeds=(NOMINAL_Q,))
        q, err = self.arms[side].ik(self.d, pos, None, q_init=q, iters=50)
        if err > 2e-3:
            raise ValueError(f"home_tcp {pos} ngoài tầm với tay {side} (sai số IK {err:.3f} m)")
        return np.append(q, GRIP_OPEN)

    def reset(self, object_poses: np.ndarray | None = None, qpos16: np.ndarray | None = None) -> None:
        """Về tư thế home (hoặc qpos16 cho trước, vd khung đầu của demo thật), đặt vật, lệnh = trạng thái."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("zero").id)
        q = np.concatenate([self.home[s] for s in SIDES]) if qpos16 is None else np.asarray(qpos16, float)
        self._set_model_q(q + self.offset)
        if object_poses is not None:
            set_object_poses(m, d, self.object_names, object_poses)
        for a in self.arms.values():
            a.set_ctrl(d, a.qpos(d))
        if self.limiter is not None:
            self.limiter.target = self.qpos16() + self.offset
            self.limiter.command = self.limiter.target.copy()
        mujoco.mj_forward(m, d)

    def settle(self, seconds: float = 0.3) -> None:
        """Để vật ổn định trên mặt hộp trước khi bắt đầu (tay giữ nguyên)."""
        for _ in range(int(seconds / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)

    def qpos16(self) -> np.ndarray:
        """qpos theo đơn vị encoder thật (đã trừ offset hiệu chuẩn)."""
        return np.concatenate([self.arms[s].qpos(self.d) for s in SIDES]) - self.offset

    def _set_model_q(self, q: np.ndarray) -> None:
        for i, s in enumerate(SIDES):
            self.arms[s].set_qpos(self.d, q[8 * i:8 * i + 8])

    def qvel16(self) -> np.ndarray:
        return np.concatenate([self.arms[s].qvel(self.d) for s in SIDES])

    def set_qpos16(self, q: np.ndarray) -> None:
        """Đặt qpos theo đơn vị encoder thật."""
        self._set_model_q(np.asarray(q, float) + self.offset)

    def step(self, action16: np.ndarray) -> None:
        """Một bước điều khiển (1/fps giây): đặt lệnh vị trí rồi chạy vật lý."""
        action16 = np.asarray(action16, float) + self.offset
        if self.limiter is not None:
            self.limiter.set_command(action16)
            for _ in range(self.substeps):
                target = self.limiter.step(self.m.opt.timestep)
                for i, s in enumerate(SIDES):
                    self.arms[s].set_ctrl(self.d, target[8 * i:8 * i + 8])
                mujoco.mj_step(self.m, self.d)
            return
        for i, s in enumerate(SIDES):
            self.arms[s].set_ctrl(self.d, action16[8 * i:8 * i + 8])
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)

    def arm_slice(self, side: str) -> slice:
        i = SIDES.index(side)
        return slice(8 * i, 8 * i + 8)
