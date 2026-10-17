"""Bài toán gắp một vật đặt vào ô đích trắng bằng một tay: trạng thái, reward, điều kiện thành công/thất bại.

Reward dày (dense) theo bậc, mỗi bước nằm trong [-penalty, 1]:
    reach  0.1 · (1 − tanh(10·|TCP − vật|))         tay tiến lại gần vật
    grasp  0.2 · [hai ngón cùng chạm vật]
    lift   0.2 · grasp · clip(độ nâng / 5 cm, 0, 1)
    place  0.3 · grasp · lift · (1 − tanh(10·|vật − ô đích|_xy))   mang vật tới trên ô đích
    success = 1.0 (thay cho tổng trên) khi vật nằm yên trong ô đích và đã nhả kẹp
    − penalty khi tay va chạm bàn / camera / thân robot
Tổng các thành phần khi chưa thành công tối đa 0.8 < 1, nên reward luôn tăng theo tiến độ.
Chế độ "sparse": 1 khi thành công, 0 còn lại (không phạt).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import Arm
from .scene import BOX_GEOM, CAM_COLLISION_GEOM, PAD_GEOM, TABLE_GEOM, box_top


@dataclass
class TaskState:
    tcp: np.ndarray
    obj: np.ndarray
    obj_vel: float
    target: np.ndarray
    grasped: bool
    lift: float              # độ cao vật so với lúc nằm trên hộp (m)
    in_target: bool          # tâm vật nằm trong ô đích theo xy, vật chạm mặt hộp/ô đích
    released: bool           # không ngón nào chạm vật
    success: bool
    dropped: bool            # vật rơi khỏi mặt hộp
    collision: bool          # tay chạm bàn / camera / thân robot
    reach_d: float
    place_d: float


class PickPlaceTask:
    def __init__(self, m: mujoco.MjModel, cfg: dict, reward_mode: str = "dense", collision_penalty: float = 0.1):
        t = cfg["task"]
        self.m, self.cfg = m, cfg
        self.arm = Arm(m, t["arm"], t.get("ik_margin"))
        self.obj_name = t["object"]
        self.obj_body = m.body(self.obj_name).id
        self.obj_geom = m.geom(f"{self.obj_name}_geom").id
        self.obj_jnt = m.body_jntadr[self.obj_body]
        self.obj_qadr = m.jnt_qposadr[self.obj_jnt]
        self.obj_dadr = m.jnt_dofadr[self.obj_jnt]
        self.target_geom = m.geom(PAD_GEOM).id
        self.target_half = m.geom_size[self.target_geom][:2].copy()
        self.pad_thickness = 2 * float(m.geom_size[self.target_geom][2])
        self.reward_mode = reward_mode
        self.collision_penalty = collision_penalty
        self.obj_half = m.geom_size[self.obj_geom].copy()   # nửa cạnh (x, y, z) của vật dạng hộp
        self.obj_half_h = _half_height(m, self.obj_geom)
        self.bad_geoms = {m.geom(n).id for n in (TABLE_GEOM, CAM_COLLISION_GEOM)}
        base = m.body(cfg["robot"]["base_body"]).id
        self.bad_geoms |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == base and m.geom_contype[g]}
        self.arm_geoms = _subtree_geoms(m, m.body(f"openarm_{t['arm']}_link1").id)
        self.box_geom = m.geom(BOX_GEOM).id

    # ------------------------------------------------------------ trạng thái
    def rest_z(self, d) -> float:
        return box_top(self.m, d)["z"] + self.obj_half_h

    def target_pos(self, d) -> np.ndarray:
        """Tâm vật khi nằm trên ô đích (tâm ô, z = mặt ô + nửa chiều cao vật)."""
        p = d.geom_xpos[self.target_geom].copy()
        p[2] = self.rest_z(d) + self.pad_thickness
        return p

    def state(self, d: mujoco.MjData) -> TaskState:
        m = self.m
        tcp = d.xpos[self.arm.tcp].copy()
        obj = d.xpos[self.obj_body].copy()
        fingers_on_obj, obj_on_box, collision = set(), False, False
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a == self.obj_geom and b in self.arm.finger_geoms:
                    fingers_on_obj.add(b)
                if a == self.obj_geom and b in (self.box_geom, self.target_geom):   # đứng trên hộp hoặc ô đích
                    obj_on_box = True
                if a in self.arm_geoms and b in self.bad_geoms:
                    collision = True
        grasped = len(fingers_on_obj) == 2
        rest = self.rest_z(d)
        tgt = self.target_pos(d)
        dxy = np.abs(obj[:2] - tgt[:2])
        in_target = bool((dxy <= self.target_half).all() and obj_on_box)
        released = not fingers_on_obj
        vel = float(np.linalg.norm(d.qvel[self.obj_dadr:self.obj_dadr + 3]))
        dropped = obj[2] < box_top(m, d)["z"] - 0.02
        return TaskState(
            tcp=tcp, obj=obj, obj_vel=vel, target=tgt, grasped=grasped, lift=float(obj[2] - rest),
            in_target=in_target, released=released, success=in_target and released and vel < 0.05,
            dropped=bool(dropped), collision=collision,
            reach_d=float(np.linalg.norm(tcp - obj)), place_d=float(np.linalg.norm(obj[:2] - tgt[:2])))

    # ------------------------------------------------------------ reward
    def reward(self, s: TaskState) -> tuple[float, dict]:
        if self.reward_mode == "sparse":
            r = float(s.success)
            return r, {"success": r}
        terms = {
            "reach": 0.1 * (1 - np.tanh(10 * s.reach_d)),
            "grasp": 0.2 * s.grasped,
        }
        lift_frac = float(np.clip(s.lift / 0.05, 0, 1)) * s.grasped
        terms["lift"] = 0.2 * lift_frac
        terms["place"] = 0.3 * lift_frac * (1 - np.tanh(10 * s.place_d))
        r = 1.0 if s.success else float(sum(terms.values()))
        terms["collision"] = -self.collision_penalty * s.collision
        r += terms["collision"]
        terms["success"] = float(s.success)
        return r, terms

    # ------------------------------------------------------------ đặt vật
    def set_object_pose(self, d: mujoco.MjData, xy, yaw: float = 0.0) -> None:
        q = self.obj_qadr
        d.qpos[q:q + 3] = [xy[0], xy[1], self.rest_z(d) + 0.0005]
        d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[self.obj_dadr:self.obj_dadr + 6] = 0


def _half_height(m: mujoco.MjModel, g: int) -> float:
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return float(s[2])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(s[1])
    return float(s[0])


def _subtree_geoms(m: mujoco.MjModel, root: int) -> set[int]:
    out = set()
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        b = m.geom_bodyid[g]
        while b > 0:
            if b == root:
                out.add(g)
                break
            b = m.body_parentid[b]
    return out


def object_poses(m: mujoco.MjModel, d: mujoco.MjData, names: list[str]) -> np.ndarray:
    """(n, 7) pos + quat của các vật có khớp tự do."""
    out = []
    for n in names:
        q = m.jnt_qposadr[m.body_jntadr[m.body(n).id]]
        out.append(d.qpos[q:q + 7].copy())
    return np.array(out)


def set_object_poses(m: mujoco.MjModel, d: mujoco.MjData, names: list[str], poses: np.ndarray) -> None:
    for n, p in zip(names, poses):
        b = m.body(n).id
        q, v = m.jnt_qposadr[m.body_jntadr[b]], m.jnt_dofadr[m.body_jntadr[b]]
        d.qpos[q:q + 7] = p
        d.qvel[v:v + 6] = 0
