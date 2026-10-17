"""Mô hình bộ chấp hành của OpenArm thật, nhận dạng từ dữ liệu (joint_commands, qpos, qvel, effort).

Chuỗi lệnh → mô-men như robot thật ghi được (2026-10-13):
  1. Bộ giới hạn tốc độ phía robot: target tiến về joint_command với tốc độ tối đa vmax[j] (rad/s); khi lệnh còn
     NaN (trước lúc teleop bắt đầu) target giữ nguyên.
  2. Servo PD không bù trọng lực: tau = kp (target − q) − kd q̇, kẹp trong giới hạn mô-men của motor.
Gripper: target độ mở tiến về lệnh với tốc độ vmax_grip (m/s), servo vị trí giới hạn lực (chạm vật thì dừng).
"""

from __future__ import annotations

import mujoco
import numpy as np

SIDES = ("left", "right")


class RateLimitedTargets:
    """Giữ target 16 chiều [trái 7 + kẹp | phải 7 + kẹp] và cập nhật mỗi bước vật lý."""

    def __init__(self, vmax16: np.ndarray, q0: np.ndarray):
        self.vmax = np.asarray(vmax16, float)
        self.target = np.asarray(q0, float).copy()
        self.command = self.target.copy()

    def set_command(self, cmd16: np.ndarray) -> None:
        cmd = np.asarray(cmd16, float)
        self.command = np.where(np.isfinite(cmd), cmd, self.command)

    def step(self, dt: float) -> np.ndarray:
        lim = self.vmax * dt
        self.target = self.target + np.clip(self.command - self.target, -lim, lim)
        return self.target


def apply_actuators(spec_or_model, params: dict, *, model: mujoco.MjModel | None = None) -> None:
    """Ghi kp, kd, giới hạn lực, giảm chấn/ma sát/armature khớp vào MjModel đã compile (sửa tại chỗ).

    params: {"left"|"right": {"kp":[7], "kd":[7], "damping":[7], "frictionloss":[7], "armature":[7],
             "force_limit":[7]}, "gripper": {"kp", "kv", "force_limit"}, "gravcomp": 0|1}
    """
    m = model if model is not None else spec_or_model
    for side in SIDES:
        p = params[side]
        for i in range(7):
            a = m.actuator(f"{side}_joint{i + 1}_ctrl").id
            j = m.joint(f"openarm_{side}_joint{i + 1}")
            kp, kd = p["kp"][i], p["kd"][i]
            m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_gainprm[a, :3] = [kp, 0, 0]
            m.actuator_biasprm[a, :3] = [0, -kp, -kd]
            m.actuator_ctrllimited[a] = 1
            m.actuator_ctrlrange[a] = j.range
            if "force_limit" in p:
                m.actuator_forcelimited[a] = 1
                m.actuator_forcerange[a] = [-p["force_limit"][i], p["force_limit"][i]]
            d = j.dofadr[0]
            for key, arr in (("damping", m.dof_damping), ("frictionloss", m.dof_frictionloss),
                             ("armature", m.dof_armature)):
                if key in p:
                    arr[d] = p[key][i]
        g = params.get("gripper")
        if g:
            for k in (1, 2):
                a = m.actuator(f"{side}_finger{k}_ctrl").id
                m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
                m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
                m.actuator_gainprm[a, :3] = [g["kp"], 0, 0]
                m.actuator_biasprm[a, :3] = [0, -g["kp"], -g["kv"]]
                m.actuator_forcelimited[a] = 1
                m.actuator_forcerange[a] = [-g["force_limit"], g["force_limit"]]
    # Robot thật không bù trọng lực (effort ≈ mô-men trọng lực theo URDF, hệ số ≈ 1.0).
    for b in range(m.nbody):
        if m.body(b).name.startswith("openarm_"):
            m.body_gravcomp[b] = 0.0


def vmax16(params: dict) -> np.ndarray:
    out = []
    for side in SIDES:
        out += list(params[side]["vmax"]) + [params.get("gripper", {}).get("vmax", 0.05)]
    return np.array(out, float)
