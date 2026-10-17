"""Điểm mẫu trên bề mặt mesh hiển thị của robot, đặt theo qpos 16 chiều của dataset.

Dùng để hiệu chuẩn camera từ chính robot: robot là vật duy nhất trong cảnh có hình học và vị trí biết chính xác
(URDF + encoder) ở mọi khung, nên không cần số đo tay của cảnh.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM16 = [*range(7), *range(8, 15)]


class RobotSurface:
    def __init__(self, model_path: str, spacing_m: float = 0.004, seed: int = 0, include_body: bool = True,
                 exclude_bodies: tuple[str, ...] = ()):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        m, rng = self.m, np.random.default_rng(seed)
        self.qadr = [m.joint(f"openarm_{s}_joint{i}").qposadr[0] for s in ("left", "right") for i in range(1, 8)]
        self.fadr = {s: [m.joint(f"openarm_{s}_finger_joint{i}").qposadr[0] for i in (1, 2)]
                     for s in ("left", "right")}
        geoms, pts, nrm, owner = [], [], [], []
        for g in range(m.ngeom):
            name = m.geom(g).name
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or "_visual" not in name:
                continue
            if not include_body and name.startswith("openarm_body"):
                continue
            if m.body(m.geom_bodyid[g]).name in exclude_bodies:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            cr = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(cr, axis=1)
            n = max(8, int(area.sum() / spacing_m**2))
            tri = rng.choice(len(f), n, p=area / area.sum())
            r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
            p = (1 - r1) * a[tri] + r1 * (1 - r2) * b[tri] + r1 * r2 * c[tri]
            nn = cr[tri] / np.maximum(np.linalg.norm(cr[tri], axis=1, keepdims=True), 1e-12)
            geoms.append(g)
            pts.append(p)
            nrm.append(nn)
            owner.append(np.full(n, len(geoms) - 1))
        self.geoms = np.array(geoms)
        self.local_pts, self.local_nrm = np.concatenate(pts), np.concatenate(nrm)
        self.owner = np.concatenate(owner)
        self.tcp = {s: m.body(f"openarm_{s}_hand_tcp").id for s in ("left", "right")}

    def tcp_pos(self, q16: np.ndarray, side: str) -> np.ndarray:
        self.set_qpos(q16)
        return self.d.xpos[self.tcp[side]].copy()

    def set_qpos(self, q16: np.ndarray) -> None:
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.qadr] = np.asarray(q16)[ARM16]
        for s, g in (("left", 7), ("right", 15)):
            d.qpos[self.fadr[s]] = q16[g]
        mujoco.mj_kinematics(self.m, d)

    def world(self, q16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(điểm [N,3], pháp tuyến [N,3]) trong hệ thế giới tại tư thế q16."""
        self.set_qpos(q16)
        R = self.d.geom_xmat[self.geoms].reshape(-1, 3, 3)[self.owner]
        t = self.d.geom_xpos[self.geoms][self.owner]
        p = np.einsum("nij,nj->ni", R, self.local_pts) + t
        n = np.einsum("nij,nj->ni", R, self.local_nrm)
        return p, n
