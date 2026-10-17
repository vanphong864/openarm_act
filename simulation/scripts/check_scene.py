#!/usr/bin/env python3
"""Kiểm tra scene: in số đo hình học, va chạm/khe hở ở q=0, tầm với, và render ảnh camera ngực."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openarm_sim.render import OffscreenRenderer, camera_resolution, save_depth  # noqa: E402  (đặt MUJOCO_GL)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE, ROOT  # noqa: E402
from openarm_sim.scene import (BOX_GEOM, CAM_COLLISION_GEOM, PAD_GEOM, TABLE_GEOM, arm_geoms, body_front_profile,  # noqa: E402
                               box_top, camera_pose, clearances, load_config, load_scene,
                               measure_column_front, shoulder_reach)


def fmt(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def free_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "check"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    m, d = load_scene(args.scene)
    problems = []

    print("== Hình học (toạ độ thế giới, m; z = 0 là mặt bàn) ==")
    front, _ = measure_column_front(cfg)
    prof = body_front_profile(m, d, cfg["robot"]["base_body"], np.arange(0.30, 0.701, 0.05))
    print(f"Mặt trước cột (dải {cfg['robot']['column_probe_z']}): x = {front:.4f}")
    print("  profile x_max theo z (0.30–0.70): " + ", ".join(f"{z:.2f}:{x:.4f}" for z, x in prof))

    camera_configs = [cfg["camera"], *(cfg.get("additional_cameras") or [])]
    for name in (camera_cfg[key]["name"] for camera_cfg in camera_configs for key in ("color", "depth")):
        p = camera_pose(m, d, name)
        print(f"Camera {name}: tâm {fmt(p['pos'])}, nhìn {fmt(p['forward'])}, lên-ảnh {fmt(p['up'])}")
        print(f"  pitch {p['pitch_deg']:.2f}° (xuống), yaw {p['yaw_deg']:.2f}°, roll {p['roll_deg']:.2f}°, "
              f"cách mặt trước cột {p['pos'][0] - front:.4f}")
    bt = box_top(m, d)
    print(f"Mặt trên hộp: z = {bt['z']:.4f}, x [{bt['x_near']:.4f}, {bt['x_far']:.4f}], "
          f"y [{bt['y'][0]:.3f}, {bt['y'][1]:.3f}]; khe cột–hộp = {bt['x_near'] - front:.4f}")

    print("\n== Vai và tầm với (q = 0) ==")
    reach = shoulder_reach(m, d)
    targets = {}
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_pad_center") >= 0:
        targets["tâm ô trắng"] = "target_pad_center"
    for name in (cfg.get("objects") or {}):
        targets[f"tâm {name}"] = f"{name}_center"
    best = {}   # label -> (khoảng thiếu nhỏ nhất so với tầm với của tay gần hơn, tầm với)
    grid_x = np.linspace(bt["x_near"], bt["x_far"], 9)
    grid_y = np.linspace(*bt["y"], 13)
    for s, r in reach.items():
        sh = r["shoulder"]
        dc = np.linalg.norm(bt["center"] - sh)
        pts = np.array([[x, y, bt["z"]] for x in grid_x for y in grid_y])
        dist = np.linalg.norm(pts - sh, axis=1)
        frac = (dist <= r["reach"]).mean()
        print(f"Tay {s}: link0 {fmt(r['link0'])}, vai (link2/J2) {fmt(sh)}")
        print(f"  tầm với ước lượng (vai→TCP duỗi thẳng) {r['reach']:.3f}; vai→tâm mặt hộp {dc:.3f}; "
              f"vai→mép gần {np.linalg.norm([bt['x_near'] - sh[0], 0 - sh[1], bt['z'] - sh[2]]):.3f}; "
              f"phần mặt hộp trong bán kính với: {frac * 100:.0f}% (điều kiện cần, chưa tính IK/hướng kẹp)")
        if dc > r["reach"]:
            problems.append(f"Tâm mặt hộp ngoài tầm với tay {s} ({dc:.3f} > {r['reach']:.3f})")
        for label, site in targets.items():
            dt_ = np.linalg.norm(d.site_xpos[m.site(site).id] - sh)
            ok = "trong" if dt_ <= r["reach"] else "NGOÀI"
            print(f"  vai→{label}: {dt_:.3f} ({ok} tầm với)")
            best[label] = min(best.get(label, (np.inf,))[0], dt_ - r["reach"]), r["reach"]

    for label, (short, _) in best.items():
        if short > 0:
            problems.append(f"{label} ngoài tầm với của cả hai tay (thiếu {short:.3f} m)")

    print("\n== Va chạm ở q = 0 ==")
    arms = set(arm_geoms(m))
    env = [BOX_GEOM, TABLE_GEOM, CAM_COLLISION_GEOM, PAD_GEOM] + [f"{n}_geom" for n in (cfg.get("objects") or {})]
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
        both_arm = c.geom1 in arms and c.geom2 in arms
        kind = "nội bộ tay (model gốc)" if both_arm else ("TAY–MÔI TRƯỜNG" if (c.geom1 in arms or c.geom2 in arms)
                                                         else "vật–môi trường")
        print(f"  tiếp xúc [{kind}]: {n1} – {n2}, dist {c.dist:+.4f}")
        if kind == "TAY–MÔI TRƯỜNG":
            problems.append(f"Tay chạm {n1}–{n2} ở q=0")
    if d.ncon == 0:
        print("  không có tiếp xúc nào")
    for t, (dist, g) in clearances(m, d, env).items():
        print(f"  khe hở nhỏ nhất tay → {t}: {dist:.4f} m ({g})")
        if dist < 0.005:
            problems.append(f"Tay quá sát/xuyên {t} ở q=0: {dist:.4f} m")

    objs = list(cfg.get("objects") or {})
    if objs:
        print("\n== Vật nằm yên trên hộp? (mô phỏng 1 s, tay giữ q = 0) ==")
        d2 = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d2, m.key("zero").id)
        arm_dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        arm_qadr = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        q0 = d2.qpos[arm_qadr].copy()
        for _ in range(int(1.0 / m.opt.timestep)):
            d2.qpos[arm_qadr] = q0      # giữ tay cố định, chỉ để vật tự lắng
            d2.qvel[arm_dofs] = 0
            mujoco.mj_step(m, d2)
        for name in objs:
            b = m.body(name).id
            p0, p1 = d.xpos[b], d2.xpos[b]
            print(f"  {name}: đầu {fmt(p0)} → sau 1 s {fmt(p1)}, dịch {np.linalg.norm(p1 - p0) * 1000:.2f} mm")
            if np.linalg.norm(p1 - p0) > 0.002:
                problems.append(f"{name} không nằm yên (dịch {np.linalg.norm(p1 - p0) * 1000:.1f} mm)")

    print("\n== Render ==")
    for camera_cfg in camera_configs:
        for key in ("color", "depth"):
            name = camera_cfg[key]["name"]
            w, h = camera_resolution(m, name)
            with OffscreenRenderer(m, w, h) as r:
                if key == "color":
                    Image.fromarray(r.rgb(d, name)).save(out / f"{name}.png")
                    print(f"  {out / f'{name}.png'} ({w}x{h})")
                else:
                    dep = r.depth(d, name)
                    save_depth(dep, str(out / f"{name}_mm.png"), str(out / f"{name}_vis.png"))
                    Image.fromarray(r.rgb(d, name)).save(out / f"{name}_rgb.png")
                    print(f"  {out / f'{name}_mm.png'} (uint16 mm), {name}_vis.png, {name}_rgb.png ({w}x{h}); "
                          f"depth tâm ảnh {dep[h // 2, w // 2]:.3f} m")

    ow, oh = cfg["render"]["overview_size"]
    views = {
        "overview": free_cam([0.2, 0, 0.3], 2.0, 140, -20),
        "side": free_cam([0.2, 0, 0.35], 1.4, 90, 0),      # nhìn từ phía +y (bên trái robot)
        "camera_closeup": free_cam([0.06, 0, 0.58], 0.30, 205, -15),  # từ phía trước-trái nhìn về ngực
    }
    with OffscreenRenderer(m, ow, oh) as r:
        opt = mujoco.MjvOption()
        for name, cam in views.items():
            r.r.update_scene(d, camera=cam, scene_option=opt)
            Image.fromarray(r.r.render()).save(out / f"{name}.png")
            print(f"  {out / f'{name}.png'}")

    print("\n== Kết luận ==")
    if problems:
        for p in problems:
            print("  CẢNH BÁO:", p)
        return 1
    print("  OK: không va chạm ở q=0, mặt hộp/vật/ô đích trong tầm với.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
