"""Một tay OpenArm: chỉ số khớp/actuator, đọc trạng thái, gửi lệnh vị trí, IK cho TCP.

Bố cục vector 8 chiều của một tay (dùng chung cho qpos/action trong HDF5):
    [joint1 .. joint7 (rad), gripper (m, độ mở mỗi ngón: 0 = kẹp hết, 0.044 = mở hết)]
"""

from __future__ import annotations

import mujoco
import numpy as np

SIDES = ("left", "right")
GRIP_OPEN = 0.044
GRIP_CLOSED = 0.0
# Hướng TCP khi kẹp từ trên xuống ở yaw = 0 (giống tư thế q = 0: z_tcp chỉ xuống, x_tcp chỉ ra trước).
R_DOWN = np.diag([1.0, -1.0, -1.0])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def grasp_rot(yaw: float, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Hướng TCP kẹp từ trên xuống, xoay quanh trục z thế giới một góc yaw.
    roll ≠ 0 nghiêng hướng kẹp sang ngang (roll > 0: mũi kẹp chĩa về +y).
    pitch > 0 nghiêng mũi kẹp về phía TRƯỚC (+x), tức hướng tiếp cận từ phía robot: đây là cách người điều khiển thật
    kẹp (nghiêng ~25–50°) và giúp tay với xa hơn nhiều so với kẹp thẳng đứng."""
    return rot_y(-pitch) @ rot_z(yaw) @ rot_x(roll) @ R_DOWN


class Arm:
    # IK giữ khớp cách giới hạn ít nhất tỉ lệ này của tầm quay, để demo không dựa vào việc "chạm trần".
    # Lưu ý: với setup thật, người điều khiển kẹp ô giữa hộp bằng khớp 2 cách giới hạn chỉ ~1.7% tầm quay; đặt lề
    # lớn hơn thì ô đó thành "ngoài tầm". Chỉnh qua `task.ik_margin` trong config.
    ik_margin = 0.03

    def __init__(self, m: mujoco.MjModel, side: str, ik_margin: float | None = None):
        if ik_margin is not None:
            self.ik_margin = ik_margin      # ghi đè mặc định của lớp (config: task.ik_margin)
        if side not in SIDES:
            raise ValueError(side)
        self.m, self.side = m, side
        joints = [m.joint(f"openarm_{side}_joint{i}") for i in range(1, 8)]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.range = np.array([j.range for j in joints])
        self.finger_qadr = np.array([m.joint(f"openarm_{side}_finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.finger_dadr = np.array([m.joint(f"openarm_{side}_finger_joint{i}").dofadr[0] for i in (1, 2)])
        self.act = np.array([m.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, 8)])
        self.finger_act = np.array([m.actuator(f"{side}_finger{i}_ctrl").id for i in (1, 2)])
        self.tcp = m.body(f"openarm_{side}_hand_tcp").id
        self.elbow = m.body(f"openarm_{side}_link4").id
        self.act_forcerange = m.actuator_forcerange[self.act].copy()
        self.finger_geoms = {m.geom(f"openarm_{side}_{f}_finger_collision").id for f in ("left", "right")}
        self._scratch = mujoco.MjData(m)

    # ------------------------------------------------------------ trạng thái / lệnh
    def qpos(self, d: mujoco.MjData) -> np.ndarray:
        return np.append(d.qpos[self.qadr], d.qpos[self.finger_qadr[0]])

    def qvel(self, d: mujoco.MjData) -> np.ndarray:
        return np.append(d.qvel[self.dadr], d.qvel[self.finger_dadr[0]])

    def set_ctrl(self, d: mujoco.MjData, target: np.ndarray) -> None:
        """target 8 chiều: 7 góc khớp + độ mở gripper (đã giới hạn theo range)."""
        d.ctrl[self.act] = np.clip(target[:7], self.range[:, 0], self.range[:, 1])
        d.ctrl[self.finger_act] = np.clip(target[7], GRIP_CLOSED, GRIP_OPEN)

    def set_qpos(self, d: mujoco.MjData, q: np.ndarray) -> None:
        d.qpos[self.qadr] = q[:7]
        d.qpos[self.finger_qadr] = q[7]

    def tcp_pose(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return d.xpos[self.tcp].copy(), d.xmat[self.tcp].reshape(3, 3).copy()

    def jacobian(self, d: mujoco.MjData) -> np.ndarray:
        """Jacobian 6×7 của TCP theo 7 khớp tay (3 hàng vị trí m/rad, 3 hàng hướng rad/rad), theo trạng thái hiện tại."""
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, d, jp, jr, self.tcp)
        return np.vstack([jp[:, self.dadr], jr[:, self.dadr]])

    def joint_torque_ratio(self, d: mujoco.MjData) -> np.ndarray:
        """|mô-men động cơ| / giới hạn của từng khớp (1 = chạm trần)."""
        return np.abs(d.actuator_force[self.act]) / self.act_forcerange[:, 1]

    def sag_compensated(self, d: mujoco.MjData, q_des: np.ndarray, iters: int = 3) -> np.ndarray:
        """Lệnh khớp (7 chiều) để servo mềm đứng đúng tại q_des dưới trọng lực: q_cmd = q_des + mô-men giữ / kp.

        Robot thật không bù trọng lực nên bị võng khi lệnh = vị trí mong muốn; người điều khiển bù bằng mắt. Policy kịch
        bản làm việc đó bằng cách tính mô-men giữ tĩnh (qfrc_bias − qfrc_passive, đã trừ phần bù trọng lực nếu có)
        tại q_des. Bỏ qua tải của vật kẹp và lực tiếp xúc (nhỏ).
        """
        s = self._scratch
        kp = self.m.actuator_gainprm[self.act, 0]
        s.qpos[:] = d.qpos
        s.qvel[:] = 0
        s.qpos[self.qadr] = q_des
        mujoco.mj_forward(self.m, s)
        hold = s.qfrc_bias[self.dadr] - s.qfrc_passive[self.dadr]
        q_cmd = q_des + hold / kp
        for _ in range(iters - 1):      # mô-men giữ đổi nhẹ theo tư thế thực tế (võng ≠ 0)
            s.qpos[self.qadr] = q_des
            hold = np.clip(hold, -self.act_forcerange[:, 1], self.act_forcerange[:, 1])
            q_cmd = q_des + hold / kp
        return np.clip(q_cmd, self.range[:, 0], self.range[:, 1])

    # ------------------------------------------------------------ IK
    def ik(self, d: mujoco.MjData, pos, rot=None, q_init=None, iters: int = 300, tol: float = 1e-3,
           rot_weight: float = 0.3, damping: float = 1e-3, stall_iters: int = 15, center_gain: float = 0.15) -> tuple[np.ndarray, float]:
        """IK bình phương tối thiểu có giảm chấn cho TCP. Trả (q 7 khớp, sai số vị trí m) — sai số đo tại đúng q trả về.

        rot = ma trận 3x3 mong muốn (None = chỉ vị trí). Sai số hướng (rad) của q trả về lưu ở `self.last_rot_err`.
        Không làm thay đổi `d`.
        """
        s = self._scratch
        s.qpos[:] = d.qpos
        if q_init is not None:
            s.qpos[self.qadr] = q_init
        pos = np.asarray(pos, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        span = self.range[:, 1] - self.range[:, 0]
        lo, hi = self.range[:, 0] + self.ik_margin * span, self.range[:, 1] - self.ik_margin * span
        mid = self.range.mean(axis=1)
        # Kẹp seed vào vùng cho phép và ghi lại vào scratch, để sai số luôn đo đúng tư thế sẽ trả về.
        q = np.clip(s.qpos[self.qadr], lo, hi)
        s.qpos[self.qadr] = q

        def measure():
            """Sai số tại tư thế hiện có trong scratch: (vector sai số có trọng số, sai số vị trí, sai số hướng)."""
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            x, R = s.xpos[self.tcp], s.xmat[self.tcp].reshape(3, 3)
            e = pos - x
            if rot is None:
                return e, float(np.linalg.norm(e)), 0.0
            # sai số hướng = vee(phần phản xứng của rot·Rᵀ) (= ½ Σ R_i × rot_i, nhưng rẻ hơn np.cross)
            M = rot @ R.T
            er = 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
            return np.concatenate([e, rot_weight * er]), float(np.linalg.norm(e)), float(np.linalg.norm(er))

        best, since_best = np.inf, 0
        for _ in range(iters):
            e, err_p, err_r = measure()
            if err_p < tol and err_r < 1e-2:
                break
            # dừng sớm khi sai số không giảm nữa (điểm ngoài tầm với / kẹt cực tiểu cục bộ)
            en = float(np.linalg.norm(e))
            if en < best - 1e-5:
                best, since_best = en, 0
            else:
                since_best += 1
                if since_best >= stall_iters:
                    break
            mujoco.mj_jacBody(self.m, s, jp, jr, self.tcp)
            J = jp[:, self.dadr] if rot is None else np.vstack([jp[:, self.dadr], rot_weight * jr[:, self.dadr]])
            Jp_inv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(J.shape[0]))
            dq = Jp_inv @ e
            if center_gain:
                # bậc dư (7 khớp > 6 chiều tác vụ): kéo tư thế về giữa tầm quay trong không gian null của J,
                # nên không làm lệch vị trí/hướng TCP nhưng tránh sát giới hạn khớp
                dq += (np.eye(7) - Jp_inv @ J) @ (center_gain * (mid - q))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, lo, hi)
            s.qpos[self.qadr] = q
        else:
            # hết số vòng lặp: q vừa được cập nhật sau lần đo cuối → đo lại cho khớp với q trả về
            _, err_p, err_r = measure()
        self.last_rot_err = err_r
        return q, err_p


def bimanual_vector(arms: dict[str, Arm], d: mujoco.MjData, fn: str = "qpos") -> np.ndarray:
    """Vector 16 chiều [trái 8, phải 8] như trong file HDF5."""
    return np.concatenate([getattr(arms[s], fn)(d) for s in SIDES])
