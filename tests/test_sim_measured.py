"""Scene đo đạc 2026-10-13: bộ giới hạn tốc độ, quy ước camera đã hiệu chuẩn, scene load được."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")      # trước khi import mujoco (không có màn hình)
mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(ROOT))

from openarm_sim.actuation import RateLimitedTargets  # noqa: E402
from openarm_sim.calibrated_camera import add_camera, project  # noqa: E402


def test_rate_limiter_caps_speed_and_holds_on_nan():
    lim = RateLimitedTargets(np.full(16, 1.0), np.zeros(16))
    lim.set_command(np.full(16, 0.5))
    for _ in range(10):
        out = lim.step(0.01)
    assert np.allclose(out, 0.1)                  # 10 bước × 0.01 s × 1 rad/s
    cmd = np.full(16, np.nan)
    lim.set_command(cmd)                           # NaN: giữ lệnh cũ
    for _ in range(100):
        out = lim.step(0.01)
    assert np.allclose(out, 0.5)


def test_calibrated_camera_matches_pinhole():
    calib = {"fx": 455.0, "fy": 607.0, "cx": 300.0, "cy": 260.0,
             "R_cam_from_world": np.diag([1.0, -1.0, -1.0]).tolist(), "C_world": [0.0, 0.0, 1.0]}
    P = np.array([0.08, -0.05, 0.0])
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    spec.stat.extent = 2.0                         # mặt cắt xa tỉ lệ theo extent; model thử chỉ có một quả cầu
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
    spec.add_material(name="red", rgba=[1, 0, 0, 1], emission=1.0)
    spec.worldbody.add_body(pos=P.tolist()).add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0],
                                                      material="red")
    add_camera(spec.worldbody, "cam", calib)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    try:
        r = mujoco.Renderer(m, 480, 640)
    except Exception as exc:  # noqa: BLE001 (máy không có OpenGL)
        pytest.skip(f"không render được: {exc}")
    r.update_scene(d, "cam")
    im = r.render().astype(int)
    r.close()
    ys, xs = np.nonzero((im[..., 0] - im[..., 1]) > 60)
    uv, _ = project(calib, P)
    assert abs(xs.mean() - uv[0, 0]) < 1.0 and abs(ys.mean() - uv[0, 1]) < 1.0


def test_measured_scene_compiles():
    cfg_path = ROOT / "config" / "scene_20261013.yaml"
    if not (ROOT / "realdata" / "scene_20261013.json").exists():
        pytest.skip("chưa có số đo realdata")
    from openarm_sim.measured_scene import compile_measured, load_measured_config
    sys.path.insert(0, str(ROOT / "scripts"))
    cfg = load_measured_config(cfg_path)
    raw = Path("/media/space/New Volume/dataset/2026-10-13")
    if not raw.exists():
        pytest.skip("không có dataset để điền tư thế home/spawn")
    from build_measured_scene import resolve
    m = compile_measured(resolve(cfg, raw))
    assert m.camera("chest").id >= 0 and m.camera("front").id >= 0
    assert np.allclose(m.cam_intrinsic[m.camera("front").id][:2], [453.2, 604.3], atol=1)
    assert m.body_gravcomp.max() == 0.0
