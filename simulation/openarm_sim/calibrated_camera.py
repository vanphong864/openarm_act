"""Camera MuJoCo đúng intrinsics/extrinsics đã hiệu chuẩn (quy ước OpenCV: x phải, y xuống, z trước).

Quy ước MuJoCo (đã kiểm bằng render điểm mốc): camera nhìn theo −z, y lên; `focal_pixel` = [fx, fy];
`principal_pixel` = độ lệch tâm quang học khỏi tâm cảm biến = [W/2 − (cx + 0.5), H/2 − (cy + 0.5)]
(OpenCV đặt tâm pixel ở số nguyên, OpenGL ở nửa pixel). Sai số chiếu < 0.5 px.
"""

from __future__ import annotations

import mujoco
import numpy as np


def cv_to_mujoco_quat(R_cam_from_world: np.ndarray) -> np.ndarray:
    R_wc = np.asarray(R_cam_from_world, float).T @ np.diag([1.0, -1.0, -1.0])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_wc.flatten())
    return q


def add_camera(parent: mujoco.MjsBody, name: str, calib: dict, size=(640, 480)) -> mujoco.MjsCamera:
    """Thêm camera vào `parent` (thường là worldbody: camera cố định trong hệ robot)."""
    W, H = size
    cam = parent.add_camera(name=name, pos=list(map(float, calib["C_world"])),
                            quat=cv_to_mujoco_quat(np.array(calib["R_cam_from_world"])).tolist())
    cam.resolution = [W, H]
    cam.sensor_size = [float(W), float(H)]
    cam.focal_pixel = [float(calib["fx"]), float(calib["fy"])]
    cam.principal_pixel = [W / 2 - (calib["cx"] + 0.5), H / 2 - (calib["cy"] + 0.5)]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    return cam


def project(calib: dict, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chiếu điểm thế giới → (pixel [N,2], độ sâu z [N]) theo mô hình pinhole của calib."""
    R, C = np.array(calib["R_cam_from_world"]), np.array(calib["C_world"])
    pc = (np.atleast_2d(P) - C) @ R.T
    return np.c_[calib["fx"] * pc[:, 0] / pc[:, 2] + calib["cx"], calib["fy"] * pc[:, 1] / pc[:, 2] + calib["cy"]], \
        pc[:, 2]
