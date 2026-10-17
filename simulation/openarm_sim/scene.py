"""Sinh MJCF từ config/scene.yaml và các phép đo hình học dùng chung cho build/check/test."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from . import ROOT

CAM_BODY = "chest_camera_link"
BOX_GEOM = "work_box"
TABLE_GEOM = "table_top"
CAM_COLLISION_GEOM = "chest_camera_collision"
PAD_GEOM = "target_pad"
BACKGROUND_GEOM = "chest_background"
BACKDROP_GROUP = 1      # bàn + sàn sim: camera ngực ẩn nhóm này khi có phông nền ảnh thật
BACKGROUND_GROUP = 5    # phông nền: chỉ camera ngực bật nhóm này (viewer mặc định ẩn nhóm 3–5)
SIDES = ("left", "right")


# ---------------------------------------------------------------- config

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Đọc YAML. File có khoá `base: <file khác>` (đường dẫn tương đối so với file này) chỉ cần ghi các khoá khác
    với file gốc; dict được trộn đệ quy, list và số thì ghi đè."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(Path(path).parent / base), cfg)
    return cfg


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """Ghi đè tạm các khoá dạng "camera.pitch_deg=43" (không sửa file YAML)."""
    cfg = copy.deepcopy(cfg)
    for s in sets:
        key, _, val = s.partition("=")
        node = cfg
        *path, leaf = key.strip().split(".")
        for k in path:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"Không có khoá '{key}' trong config")
        node[leaf] = yaml.safe_load(val)
    return cfg


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- mesh slicing

def _geom_world_mesh(m: mujoco.MjModel, d: mujoco.MjData, g: int):
    mid = m.geom_dataid[g]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    v = m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return v, m.mesh_face[fa:fa + fn]


def _slice_points(v: np.ndarray, f: np.ndarray, z: float) -> np.ndarray:
    """Giao điểm của các cạnh tam giác với mặt phẳng ngang z."""
    t = v[f]
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = t[:, i], t[:, j]
        za, zb = a[:, 2], b[:, 2]
        mk = (za - z) * (zb - z) < 0
        s = (z - za[mk]) / (zb[mk] - za[mk])
        pts.append(a[mk] + s[:, None] * (b[mk] - a[mk]))
    return np.concatenate(pts) if pts else np.empty((0, 3))


def body_front_profile(m: mujoco.MjModel, d: mujoco.MjData, body: str,
                       zs) -> list[tuple[float, float]]:
    """x lớn nhất của mọi mesh thuộc `body` tại từng lát cắt z (toạ độ thế giới)."""
    bid = m.body(body).id
    meshes = [_geom_world_mesh(m, d, g) for g in range(m.ngeom)
              if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    out = []
    for z in zs:
        xs = [p[:, 0].max() for v, f in meshes if len(p := _slice_points(v, f, z))]
        out.append((float(z), float(max(xs)) if xs else float("nan")))
    return out


def measure_column_front(cfg: dict) -> tuple[float, list[tuple[float, float]]]:
    """Đo mặt trước cột chân đế từ model gốc trong dải z cấu hình."""
    r = cfg["robot"]
    m = mujoco.MjModel.from_xml_path(str(_abs(r["model"])))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    z0, z1 = r["column_probe_z"]
    prof = body_front_profile(m, d, r["base_body"], np.linspace(z0, z1, r["column_probe_steps"]))
    xs = np.array([x for _, x in prof])
    if np.isnan(xs).any():
        raise ValueError(f"Có lát cắt không chạm mesh chân đế: {prof}")
    return float(xs.max()), prof


# ---------------------------------------------------------------- rotations

def euler_zyx_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """quat (w,x,y,z) của R = Rz(yaw) Ry(pitch) Rx(roll). Pitch dương làm trục x chúc xuống."""
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rotation = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rotation.flatten())
    return q


# ---------------------------------------------------------------- build

@dataclass
class Derived:
    column_front_x: float
    column_profile: list
    camera_pos: np.ndarray            # gốc chest_camera_link (quang tâm, tâm vỏ theo y) trong base_body
    box_center: np.ndarray
    box_half: np.ndarray
    extra: dict = field(default_factory=dict)


def derive(cfg: dict) -> Derived:
    front, prof = measure_column_front(cfg)
    c, b = cfg["camera"], cfg["box"]
    cam_pos = np.array([front + c["offset_x_from_column"], c["housing_center_y"], c["lens_center_height"]])
    half = np.array([b["size_x"] / 2, b["size_y"] / 2, b["top_height"] / 2])
    center = np.array([front + b["gap_from_column"] + half[0], b["center_y"], half[2]])
    return Derived(front, prof, cam_pos, center, half)


def build_spec(cfg: dict, out_path: Path) -> tuple[mujoco.MjSpec, Derived]:
    der = derive(cfg)
    r, c, b, t = cfg["robot"], cfg["camera"], cfg["box"], cfg["table"]
    model_path = _abs(r["model"])
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.modelname = "openarm_v1_chest_cam"

    # Đường dẫn mesh tương đối so với file XML sinh ra.
    meshdir_abs = (model_path.parent / spec.meshdir).resolve()
    out_dir = out_path.parent.resolve()
    spec.meshdir = str(meshdir_abs)  # đổi sang tương đối trong write_scene, sau khi compile
    der_meshdir_rel = os.path.relpath(meshdir_abs, out_dir)
    d435_rel = os.path.relpath(_abs(c["mesh"]).resolve(), meshdir_abs)

    # ---- hình ảnh / ánh sáng
    vis = spec.visual
    vis.headlight.diffuse = [0.4, 0.4, 0.4]
    vis.headlight.ambient = [0.3, 0.3, 0.3]
    vis.headlight.specular = [0, 0, 0]
    vis.quality.shadowsize = 4096
    ow, oh = cfg["render"]["overview_size"]
    camera_configs = [c, *(cfg.get("additional_cameras") or [])]
    vis.global_.offwidth = max(ow, 1920, *(cam[key]["resolution"][0]
                                           for cam in camera_configs for key in ("color", "depth")))
    vis.global_.offheight = max(oh, 1080, *(cam[key]["resolution"][1]
                                             for cam in camera_configs for key in ("color", "depth")))
    vis.map.znear = 0.005   # tỉ lệ theo extent; đủ nhỏ để thấy vật sát camera
    spec.stat.center = [0.2, 0, 0.35]
    spec.stat.extent = 1.0

    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                     width=300, height=300)
    mat = spec.add_material(name="groundplane", texuniform=True, texrepeat=[5, 5], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.add_material(name="d435_aluminum", rgba=[0.62, 0.63, 0.66, 1], specular=0.5, shininess=0.5)
    spec.add_material(name="bracket", rgba=[0.15, 0.15, 0.15, 1])
    spec.add_mesh(name="d435_housing", file=d435_rel)

    wb = spec.worldbody
    wb.add_light(name="top_light", pos=[0.3, 0, 2.0], dir=[0, 0, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.5, 0.5, 0.5], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                pos=[0, 0, t["floor_z"]], material="groundplane", contype=0, conaffinity=0, group=BACKDROP_GROUP)

    # ---- bàn (mặt trên z = 0)
    (x0, x1), (y0, y1) = t["x_range"], t["y_range"]
    wb.add_geom(name=TABLE_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(x0 + x1) / 2, (y0 + y1) / 2, -t["thickness"] / 2],
                size=[(x1 - x0) / 2, (y1 - y0) / 2, t["thickness"] / 2],
                rgba=t["rgba"], contype=1, conaffinity=1, condim=3, friction=[1, 0.01, 0.01], group=BACKDROP_GROUP)

    # ---- hộp kê (có va chạm)
    box_body = wb.add_body(name="work_box_body", pos=der.box_center.tolist())
    box_body.add_geom(name=BOX_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=der.box_half.tolist(),
                      rgba=b["rgba"], contype=1, conaffinity=1, condim=3, friction=[1, 0.01, 0.01])
    box_body.add_site(name="work_box_top_center", pos=[0, 0, der.box_half[2]], size=[0.005, 0, 0],
                      rgba=[1, 0, 0, 1], group=4)

    # ---- ô trắng (đích đặt vật) trên mặt hộp
    pad = cfg.get("target_pad")
    if pad:
        (px, py), th = pad["offset_xy"], pad["thickness"]
        box_body.add_geom(name=PAD_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[px, py, der.box_half[2] + th / 2],
                          size=[pad["size"][0] / 2, pad["size"][1] / 2, th / 2],
                          rgba=pad["rgba"], contype=1, conaffinity=1, condim=3, friction=[1, 0.01, 0.01])
        box_body.add_site(name="target_pad_center", pos=[px, py, der.box_half[2] + th], size=[0.004, 0, 0],
                          rgba=[1, 0, 0, 1], group=4)

    # ---- vật để gắp (thân tự do), đặt nằm trên mặt hộp
    for name, o in (cfg.get("objects") or {}).items():
        sx, sy, sz = o["size"]
        top = 2 * der.box_half[2]
        ob = wb.add_body(name=name, pos=[*o["pos_xy"], top + sz / 2],
                         quat=euler_zyx_quat(o.get("yaw_deg", 0.0), 0, 0).tolist())
        ob.add_freejoint(name=f"{name}_freejoint")
        ob.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, sz / 2],
                    mass=o["mass"], rgba=o["rgba"], contype=1, conaffinity=1, condim=4,
                    friction=o.get("friction", [0.6, 0.005, 0.0001]))
        ob.add_site(name=f"{name}_center", size=[0.003, 0, 0], rgba=[1, 1, 0, 1], group=4)

    # ---- camera ngực D435i, gắn vào chân đế
    base = spec.body(r["base_body"])
    rs = c["realsense"]
    cam_body = base.add_body(name=CAM_BODY, pos=der.camera_pos.tolist(),
                             quat=euler_zyx_quat(c["yaw_deg"], c["pitch_deg"], c["roll_deg"]).tolist())
    # Khung chest_camera_link (quy ước camera_link của RealSense): x = hướng nhìn, y = trái, z = trên.
    # Gốc: mặt phẳng quang tâm (zero-depth), y = tâm vỏ. Mesh d435.stl: mặt kính z=0 hướng +z, trên = +y,
    # +x = phía có ống RGB (= trái của camera).
    glass = rs["glass_front_from_optical"]
    cam_body.add_geom(name="chest_camera_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="d435_housing",
                      pos=[glass, 0, 0], quat=_xyaxes_quat([0, 1, 0], [0, 0, 1]).tolist(),
                      material="d435_aluminum", contype=0, conaffinity=0, group=2)
    hd, hw, hh = rs["housing_size"]
    if c.get("collision", True):
        cam_body.add_geom(name=CAM_COLLISION_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[glass - hd / 2, 0, 0], size=[hd / 2, hw / 2, hh / 2],
                          contype=1, conaffinity=1, group=3, rgba=[1, 0.28, 0.1, 0.5])
    # Camera MuJoCo nhìn theo −z, ảnh "lên" = +y → x_cam = −y_link (phải của ảnh), y_cam = +z_link.
    cam_quat = _xyaxes_quat([0, -1, 0], [0, 0, 1])
    y_depth = rs["depth_y_from_housing_center"]
    for key, y in (("color", y_depth + rs["color_y_from_depth"]), ("depth", y_depth)):
        cc = c[key]
        cam = cam_body.add_camera(name=cc["name"], pos=[0, y, 0], quat=cam_quat.tolist(),
                                  fovy=cc["fovy_deg"], resolution=cc["resolution"])
        cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        cam_body.add_site(name=f"{cc['name']}_optical", pos=[0, y, 0], size=[0.003, 0, 0],
                          rgba=[0, 1, 0, 1], group=4)

    # Các D435 bổ sung: pose tính trong hệ của parent_body (hoặc world).
    for extra in cfg.get("additional_cameras") or []:
        name = extra["name"]
        parent_name = extra.get("parent_body", "world")
        parent = spec.worldbody if parent_name == "world" else spec.body(parent_name)
        body = parent.add_body(name=f"{name}_camera_link", pos=extra["pos"],
                               quat=euler_zyx_quat(extra["yaw_deg"], extra["pitch_deg"],
                                                   extra["roll_deg"]).tolist())
        offsets = extra.get("realsense", rs)
        depth_y = offsets["depth_y_from_housing_center"]
        for key, y in (("color", depth_y + offsets["color_y_from_depth"]), ("depth", depth_y)):
            cc = extra[key]
            cam = body.add_camera(name=cc["name"], pos=[0, y, 0], quat=cam_quat.tolist(),
                                  fovy=cc["fovy_deg"], resolution=cc["resolution"])
            cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            body.add_site(name=f"{cc['name']}_optical", pos=[0, y, 0], size=[0.003, 0, 0],
                          rgba=[0, 1, 0, 1], group=4)

    if c.get("mount_bracket", True):
        # Tấm đỡ đơn giản: từ mặt trước cột tới lưng camera, ngay dưới tâm lưng vỏ (chỉ hiển thị).
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, euler_zyx_quat(c["yaw_deg"], c["pitch_deg"], c["roll_deg"]))
        back = der.camera_pos + rot.reshape(3, 3) @ np.array([glass - hd, 0, 0])
        x_a, x_b = der.column_front_x, back[0]
        if x_b > x_a + 1e-3:
            base.add_geom(name="chest_camera_bracket", type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[(x_a + x_b) / 2, back[1], back[2] - 0.004],
                          size=[(x_b - x_a) / 2, 0.02, 0.004],
                          material="bracket", contype=0, conaffinity=0, group=2)

    _apply_control(spec, cfg.get("control"))
    _add_background(spec, cfg, c["color"]["name"])

    # Keyframe "zero": khớp tay = 0, vật ở vị trí ban đầu (qpos0 chứa sẵn pose của freejoint).
    spec.add_key(name="zero", qpos=spec.compile().qpos0.tolist())
    der.extra["d435_rel"] = d435_rel
    der.extra["meshdir_rel"] = der_meshdir_rel
    return spec, der


def _apply_control(spec: mujoco.MjSpec, ctl: dict | None) -> None:
    """Đổi mọi actuator của tay thành servo vị trí PD (ctrl = vị trí khớp mong muốn), bù trọng lực thân robot.

    Actuator gốc của openarm_mujoco là motor mô-men (riêng ngón tay phải là servo vị trí). Servo PD giống chế độ MIT
    (kp, kd) của motor Damiao, nên lệnh góc khớp trong demo HDF5 phát lại trực tiếp được. Không có mục `control`
    trong config thì giữ nguyên actuator gốc.
    """
    if not ctl:
        return
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{ctl['integrator'].upper()}")
    for side in SIDES:
        names = [(f"{side}_joint{i}_ctrl", f"openarm_{side}_joint{i}", ctl["arm_kp"][i - 1], ctl["arm_kv"][i - 1])
                 for i in range(1, 8)]
        names += [(f"{side}_finger{i}_ctrl", f"openarm_{side}_finger_joint{i}", ctl["gripper_kp"], ctl["gripper_kv"])
                  for i in (1, 2)]
        for act_name, joint, kp, kv in names:
            a = spec.actuator(act_name)
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = kp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0:3] = [0, -kp, -kv]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrlrange = spec.joint(joint).range
    if ctl.get("gravity_compensation", True):
        for b in spec.bodies:
            if b.name.startswith("openarm_"):
                b.gravcomp = 1.0


def _add_background(spec: mujoco.MjSpec, cfg: dict, cam_name: str) -> None:
    """Phông nền ảnh thật cho camera ngực (mục `background` trong config).

    Một tấm phẳng vuông góc trục quang học, đặt xa `distance` m và phủ đúng trường nhìn của camera màu, dán ảnh nền
    (trung vị khung đầu các episode thật, kéo về 16:9 – xem scripts/make_background.py). Camera gắn cố định vào thân
    robot nên từ camera đó nhìn tấm phẳng giống hệt cảnh thật ở những chỗ không bị hộp/tay/vật trong sim che.
    Bàn và sàn của sim (nhóm BACKDROP_GROUP) bị ẩn khi render camera ngực để không che phông (openarm_sim/sensor.py).
    Không có file ảnh thì bỏ qua (cảnh như cũ).
    """
    bg = cfg.get("background")
    if not bg:
        return
    img = _abs(bg["image"])
    if not img.exists():
        print(f"Cảnh báo: không có ảnh nền {img} (chạy scripts/make_background.py) → bỏ qua phông nền")
        return
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = m.camera(cam_name).id
    C, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
    dist = float(bg.get("distance", 2.0))
    hh = dist * np.tan(np.radians(m.cam_fovy[cid]) / 2) * 1.02
    w, h = cfg["camera"]["color"]["resolution"]
    hw = hh * w / h
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    spec.add_texture(name="chest_background", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(img))
    mat = spec.add_material(name="chest_background", specular=0, shininess=0, reflectance=0,
                            emission=float(bg.get("emission", 0.0)))
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "chest_background"
    spec.worldbody.add_geom(name=BACKGROUND_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hw, hh, 0.001],
                            pos=(C - dist * R[:, 2]).tolist(), quat=q.tolist(), material="chest_background",
                            contype=0, conaffinity=0, group=BACKGROUND_GROUP)


def _xyaxes_quat(xaxis, yaxis) -> np.ndarray:
    x = np.asarray(xaxis, float)
    x /= np.linalg.norm(x)
    y = np.asarray(yaxis, float)
    y = y - x * (x @ y)
    y /= np.linalg.norm(y)
    mat = np.column_stack([x, y, np.cross(x, y)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def write_scene(cfg: dict, out_path: str | os.PathLike) -> Derived:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec, der = build_spec(cfg, out_path)
    xml = spec.to_xml()
    absdir = f'meshdir="{spec.meshdir}'
    if absdir not in xml:
        raise RuntimeError("Không tìm thấy meshdir trong XML xuất ra")
    xml = xml.replace(absdir, f'meshdir="{der.extra["meshdir_rel"]}')
    header = ("<!-- File SINH TỰ ĐỘNG bởi scripts/build_scene.py từ config/scene.yaml. Đừng sửa tay.\n"
              "     Robot: enactic/openarm_mujoco v1 (Apache-2.0), xem third_party/openarm_mujoco/SOURCE.md -->\n")
    out_path.write_text(header + xml, encoding="utf-8")
    return der


# ---------------------------------------------------------------- measurements

def camera_pose(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> dict:
    cid = m.camera(name).id
    R = d.cam_xmat[cid].reshape(3, 3)
    fwd, up, right = -R[:, 2], R[:, 1], R[:, 0]
    return {
        "pos": d.cam_xpos[cid].copy(),
        "forward": fwd, "up": up, "right": right,
        "pitch_deg": math.degrees(math.asin(-fwd[2])),
        "yaw_deg": math.degrees(math.atan2(fwd[1], fwd[0])),
        "roll_deg": math.degrees(math.asin(np.clip(-right[2], -1, 1))),
    }


def arm_geoms(m: mujoco.MjModel) -> list[int]:
    """Các geom va chạm thuộc hai tay (mọi body con của link0 trái/phải)."""
    roots = {m.body(f"openarm_{s}_link0").id for s in SIDES}
    out = []
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        b = m.geom_bodyid[g]
        while b != 0:
            if b in roots:
                out.append(g)
                break
            b = m.body_parentid[b]
    return out


def clearances(m: mujoco.MjModel, d: mujoco.MjData, targets: list[str], distmax: float = 0.5) -> dict:
    """Khoảng cách nhỏ nhất (m) từ geom va chạm của tay tới từng geom đích; âm = xuyên vào."""
    arms = arm_geoms(m)
    fromto = np.zeros(6)
    out = {}
    for t in targets:
        try:
            tg = m.geom(t).id
        except KeyError:
            continue
        best = (distmax, None)
        for g in arms:
            dist = mujoco.mj_geomDistance(m, d, g, tg, distmax, fromto)
            if dist < best[0]:
                best = (dist, m.geom(g).name)
        out[t] = best
    return out


def shoulder_reach(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    """Vị trí vai (gốc link2 = trục J2) và tầm với ước lượng = |vai → TCP| khi tay duỗi thẳng ở q=0."""
    out = {}
    for s in SIDES:
        sh = d.xpos[m.body(f"openarm_{s}_link2").id].copy()
        tcp = d.xpos[m.body(f"openarm_{s}_hand_tcp").id].copy()
        out[s] = {"link0": d.xpos[m.body(f"openarm_{s}_link0").id].copy(),
                  "shoulder": sh, "tcp": tcp, "reach": float(np.linalg.norm(tcp - sh))}
    return out


def box_top(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    g = m.geom(BOX_GEOM).id
    c, h = d.geom_xpos[g], m.geom_size[g]
    return {"z": float(c[2] + h[2]), "x_near": float(c[0] - h[0]), "x_far": float(c[0] + h[0]),
            "y": (float(c[1] - h[1]), float(c[1] + h[1])), "center": np.array([c[0], c[1], c[2] + h[2]])}


def load_scene(path: str | os.PathLike, key: str = "zero"):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    mujoco.mj_forward(m, d)
    return m, d
