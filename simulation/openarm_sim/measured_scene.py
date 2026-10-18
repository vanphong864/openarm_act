"""Scene MuJoCo dựng từ SỐ ĐO TRÊN DỮ LIỆU (không suy diễn từ cột robot như scene.py).

Nguồn (đều nằm trong simulation/realdata/, sinh bởi các script trong simulation/scripts/):
  cameras_*.json    intrinsics + extrinsics từng camera (calibrate_cameras.py → refine_cameras_render.py)
  actuators_*.json  servo PD, bộ giới hạn tốc độ, ma sát/giảm chấn khớp (identify_actuators.py)
  scene_*.json      mặt làm việc, ô giấy, kích thước/tư thế hộp kẹo (reconstruct_scene.py)
Config YAML chỉ trỏ tới các file trên và ghi những gì dữ liệu không đo được (khối lượng, ma sát vật, màu).

Tên geom/body/camera giữ như scene.py để task.py / policy.py / runner.py dùng lại được.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import yaml

from . import ROOT
from .actuation import apply_actuators
from .calibrated_camera import add_camera
from .scene import (BACKDROP_GROUP, BOX_GEOM, CAM_COLLISION_GEOM, PAD_GEOM, TABLE_GEOM, _abs, _xyaxes_quat,
                    euler_zyx_quat)

DYNAMIC_GROUP = 0      # robot + vật: camera render nhóm này rồi ghép lên nền thật
STATIC_GROUP = 1       # bàn, hộp kê, ô giấy: có va chạm; camera dùng nền thật thay cho nhóm này


def load_measured_config(path: str | os.PathLike) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = Path(path).parent
    for key in [k for k in ("cameras", "actuators", "scene", "joint_offsets") if k in cfg["measurements"]]:
        p = Path(cfg["measurements"][key])
        cfg["measurements"][key] = str(p if p.is_absolute() else (base / p).resolve())
    cfg["_measured"] = {k: json.loads(Path(v).read_text()) for k, v in cfg["measurements"].items()}
    return cfg


def _plane_quat(normal: np.ndarray, yaw: float) -> np.ndarray:
    """Hướng một hộp phẳng: trục z = pháp tuyến mặt, trục x = hướng yaw chiếu lên mặt."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x -= n * (n @ x)
    return _xyaxes_quat(x, np.cross(n, x))


def build_measured_spec(cfg: dict) -> mujoco.MjSpec:
    meas = cfg["_measured"]
    sc, cams, act = meas["scene"], meas["cameras"]["cameras"], meas["actuators"]["params"]
    model_path = _abs(cfg["robot"]["model"])
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.modelname = "openarm_v1_measured"
    spec.meshdir = str((model_path.parent / spec.meshdir).resolve())
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    vis = spec.visual
    vis.global_.offwidth, vis.global_.offheight = 1280, 960
    vis.headlight.diffuse = [0.45, 0.45, 0.45]
    vis.headlight.ambient = [0.35, 0.35, 0.35]
    vis.headlight.specular = [0, 0, 0]
    vis.map.znear = 0.005
    wb = spec.worldbody
    wb.add_light(name="top_light", pos=[0.4, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.5, 0.5, 0.5], castshadow=False)

    # ---- mặt đỡ (bàn gỗ) và hộp kê (mặt làm việc đo được)
    s = sc["surface"]
    n = np.array(s["normal"])
    top_z = s["height_at_origin_m"]
    support_z = s.get("support_plane", {}).get("height_at_origin_m", cfg["table"].get("fallback_support_z", 0.0))
    t = cfg["table"]
    (x0, x1), (y0, y1) = t["x_range"], t["y_range"]
    wb.add_geom(name=TABLE_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(x0 + x1) / 2, (y0 + y1) / 2, support_z - 0.02], size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.02],
                rgba=t["rgba"], contype=1, conaffinity=1, condim=3, friction=[1, 0.01, 0.01], group=STATIC_GROUP)
    b = cfg["box"]
    cx, cy = (b["center_xy"] if b.get("center_xy") else s["center"][:2])
    L = b.get("size_x") or s["size_long_m"]
    Wd = b.get("size_y") or s["size_short_m"]
    yaw = np.radians(b.get("yaw_deg", 0.0))
    hz = (top_z - support_z) / 2
    # Mặt trên của hộp nằm đúng trên mặt phẳng đo (kể cả độ nghiêng nhỏ): đặt tâm theo pháp tuyến.
    top_center = np.array([cx, cy, (s["offset"] - n[0] * cx - n[1] * cy) / n[2]])
    box_quat = _plane_quat(n, yaw)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, box_quat)
    center = top_center - R.reshape(3, 3)[:, 2] * hz
    box = wb.add_body(name="work_box_body", pos=center.tolist(), quat=box_quat.tolist())
    box.add_geom(name=BOX_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, Wd / 2, hz], rgba=b["rgba"],
                 contype=1, conaffinity=1, condim=3, friction=b.get("friction", [1, 0.01, 0.01]), group=STATIC_GROUP)
    box.add_site(name="work_box_top_center", pos=[0, 0, hz], size=[0.005, 0, 0], rgba=[1, 0, 0, 1], group=4)

    # ---- ô giấy trắng (theo số đo: tâm, kích thước, yaw trong hệ thế giới)
    pad = cfg["target_pad"]
    pc = np.array(pad["center"])
    rel = R.reshape(3, 3).T @ (pc - center)
    th = pad.get("thickness", 0.0005)
    pyaw = np.radians(pad["yaw_deg"]) - yaw
    box.add_geom(name=PAD_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[rel[0], rel[1], hz + th / 2],
                 quat=euler_zyx_quat(np.degrees(pyaw), 0, 0).tolist(),
                 size=[pad["size"][0] / 2, pad["size"][1] / 2, th / 2], rgba=pad["rgba"],
                 contype=1, conaffinity=1, condim=3, friction=[1, 0.01, 0.01], group=STATIC_GROUP)
    box.add_site(name="target_pad_center", pos=[rel[0], rel[1], hz + th], size=[0.004, 0, 0], rgba=[1, 0, 0, 1],
                 group=4)

    # ---- vật (thân tự do), nằm trên mặt hộp
    for name, o in cfg["objects"].items():
        sx, sy, sz = o["size"]
        p0 = np.array([*o["pos_xy"], 0.0])
        p0[2] = (s["offset"] - n[0] * p0[0] - n[1] * p0[1]) / n[2] + sz / 2 + 1e-4
        ob = wb.add_body(name=name, pos=p0.tolist(), quat=euler_zyx_quat(o.get("yaw_deg", 0.0), 0, 0).tolist())
        ob.add_freejoint(name=f"{name}_freejoint")
        ob.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, sz / 2],
                    mass=o["mass"], rgba=o["rgba"], contype=1, conaffinity=1, condim=4,
                    friction=o.get("friction", [0.6, 0.005, 0.0001]), group=DYNAMIC_GROUP)
        ob.add_site(name=f"{name}_center", size=[0.003, 0, 0], rgba=[1, 1, 0, 1], group=4)
        if o.get("top_texture"):
            # Nhãn mặt trên lấy từ ảnh thật (scripts/make_tin_texture.py): lớp mỏng chỉ để hiển thị.
            spec.add_texture(name=f"{name}_top_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                             file=str(_abs(o["top_texture"])))
            mat = spec.add_material(name=f"{name}_top_mat")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_top_tex"
            ob.add_geom(name=f"{name}_label", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, 1e-4],
                        pos=[0, 0, sz / 2 + 1e-4], material=f"{name}_top_mat", rgba=[1, 1, 1, 1],
                        contype=0, conaffinity=0, density=0, group=DYNAMIC_GROUP)

    # ---- camera đã hiệu chuẩn (+ vỏ D435 va chạm cho camera ngực, nằm sau quang tâm)
    for cam_name, c in cams.items():
        add_camera(wb, cam_name, c)
        hc = cfg.get("camera_housings", {}).get(cam_name)
        if hc:
            Rc = np.array(c["R_cam_from_world"]).T          # cột: trục camera (x phải, y xuống, z trước) trong thế giới
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, Rc.flatten())
            size = hc.get("size", [0.090, 0.025, 0.025])     # rộng (x cam) × cao (y cam) × sâu (z cam)
            pos = np.array(c["C_world"]) - Rc[:, 2] * size[2] / 2
            wb.add_geom(name=CAM_COLLISION_GEOM if cam_name == "chest" else f"{cam_name}_camera_collision",
                        type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos.tolist(), quat=q.tolist(),
                        size=[s_ / 2 for s_ in size], rgba=[0.62, 0.63, 0.66, 1], contype=1, conaffinity=1,
                        group=STATIC_GROUP)

    # ---- hiệu chuẩn động học: tịnh tiến bệ vai (link0) mỗi tay; offset zero khớp xử lý ở runner.Sim
    jo = meas.get("joint_offsets")
    if jo:
        for side in ("left", "right"):
            t = jo["arms"][side].get("base_translation_m")
            if t:
                b = spec.body(f"openarm_{side}_link0")
                b.pos = (np.array(b.pos) + np.array(t)).tolist()

    spec.add_key(name="zero", qpos=spec.compile().qpos0.tolist())
    return spec


def compile_measured(cfg: dict) -> mujoco.MjModel:
    m = build_measured_spec(cfg).compile()
    apply_actuators(m, cfg["_measured"]["actuators"]["params"])
    return m


def write_measured_scene(cfg: dict, out_path: str | os.PathLike) -> Path:
    """Ghi XML. Tham số bộ chấp hành được ghi vào XML bằng cách sửa spec rồi xuất (giữ nguyên khi load lại)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_measured_spec(cfg)
    params = cfg["_measured"]["actuators"]["params"]
    for side in ("left", "right"):
        p = params[side]
        for i in range(7):
            a = spec.actuator(f"{side}_joint{i + 1}_ctrl")
            j = spec.joint(f"openarm_{side}_joint{i + 1}")
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = p["kp"][i]
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0:3] = [0, -p["kp"][i], -p["kd"][i]]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrlrange = j.range
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-p["force_limit"][i], p["force_limit"][i]]
            j.damping[0] = p["damping"][i]
            j.frictionloss = p["frictionloss"][i]
            j.armature = p["armature"][i]
        g = params["gripper"]
        for k in (1, 2):
            a = spec.actuator(f"{side}_finger{k}_ctrl")
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = g["kp"]
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0:3] = [0, -g["kp"], -g["kv"]]
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-g["force_limit"], g["force_limit"]]
    for bdy in spec.bodies:
        if bdy.name.startswith("openarm_"):
            bdy.gravcomp = 0.0
    xml = spec.to_xml()
    meshdir_rel = os.path.relpath(spec.meshdir, out_path.parent.resolve())
    xml = xml.replace(f'meshdir="{spec.meshdir}', f'meshdir="{meshdir_rel}')
    header = ("<!-- File SINH TỰ ĐỘNG bởi scripts/build_measured_scene.py từ số đo trên dữ liệu thật. Đừng sửa tay.\n"
              "     Robot: enactic/openarm_mujoco v1 (Apache-2.0), xem third_party/openarm_mujoco/SOURCE.md -->\n")
    out_path.write_text(header + xml, encoding="utf-8")
    return out_path
