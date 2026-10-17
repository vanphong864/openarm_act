#!/usr/bin/env python3
"""Hiệu chuẩn động học robot từ camera: offset zero của 7 khớp mỗi tay (q_thật = q_encoder + Δ).

Bằng chứng FK lệch: lúc kẹp cùng một hộp kẹo trên cùng mặt bàn, TCP tay trái ở z ≈ 0.178 m, tay phải ≈ 0.191 m
(lệch 1.3 cm); tâm hộp kẹo đo bởi hai camera (đã hiệu chuẩn nhất quán, intrinsics đo độc lập) lệch có hệ thống
3–5 cm so với FK theo phương ngang, ngược dấu ở hai tay. Ràng buộc ở đây, mỗi episode:
    FK_tcp(q_kẹp + Δ) + R_tcp(q_kẹp + Δ) · [0, 0, a] = tâm hộp kẹo (camera, khung đầu; z = mặt hộp + dày/2)
a: khoảng dọc trục bàn tay từ TCP tới tâm vật, cố định chung hai tay (mặc định 9 mm). Giải Gauss–Newton có prior Δ ~ N(0, σ²), σ = 2°.
Báo cáo sai số trước/sau trên episode giữ lại (chia theo tay).

Ví dụ:
  calibrate_joint_offsets.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --cameras simulation/realdata/cameras_20261013.json --scene simulation/realdata/scene_20261013.json \
      --out simulation/realdata/joint_offsets_20261013.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

ROBOT = Path(__file__).resolve().parent.parent / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"


class FK:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = {s: [m.joint(f"openarm_{s}_joint{i}").qposadr[0] for i in range(1, 8)] for s in ("left", "right")}
        self.tcp = {s: m.body(f"openarm_{s}_hand_tcp").id for s in ("left", "right")}

    def point(self, side, q7, a, base=np.zeros(3)):
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.qadr[side]] = q7
        mujoco.mj_kinematics(self.m, d)
        R = d.xmat[self.tcp[side]].reshape(3, 3)
        return d.xpos[self.tcp[side]] + R[:, 2] * a + base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tin-thickness", type=float, default=0.016)
    ap.add_argument("--sigma-deg", type=float, default=2.0, help="prior offset khớp vai/khuỷu (1–4)")
    ap.add_argument("--sigma-deg-wrist", type=float, default=1.0,
                    help="prior offset cổ tay (5–7): ít ảnh hưởng vị trí TCP nên dễ thành bù trừ suy biến (vd j7 −12°)")
    ap.add_argument("--sigma-mm", type=float, default=10.0, help="nhiễu mỗi quan sát (kẹp lệch tâm, đo camera)")
    ap.add_argument("--axial", type=float, default=0.009,
                    help="tâm vật nằm cách TCP dọc trục bàn tay (m), CHUNG hai tay: đầu ngón thò 15 mm sau TCP, kẹp hộp "
                         "dày 16 mm sát đáy → tâm ≈ +9 mm. Để tự do (--free-axial) thì a hấp thụ sai lệch chiều đứng.")
    ap.add_argument("--free-axial", action="store_true")
    ap.add_argument("--release-weight", type=float, default=0.67,
                    help="trọng số ràng buộc xy lúc nhả (vật rơi/trượt khi thả: nhiễu lớn hơn lúc kẹp); 0 = bỏ")
    ap.add_argument("--base-sigma-mm", type=float, default=20.0,
                    help="prior tịnh tiến bệ vai (link0) mỗi tay; 0 = không fit. Offset khớp nhỏ không mô tả được độ "
                         "lệch đứng hằng số (tay phải ~ +18 mm)")
    args = ap.parse_args()
    cal = json.loads(args.cameras.read_text())
    sc = json.loads(args.scene.read_text())
    held = set(cal["heldout_episodes"])
    h_center = cal["box_top_height_m"] + args.tin_thickness / 2
    start = {r["episode"]: r for r in sc["tin_poses"]["start"]}
    end = {r["episode"]: r for r in sc["tin_poses"]["end"]}
    fk = FK()
    obs = {"left": [], "right": []}
    for ep, r in start.items():
        with h5py.File(args.raw / f"{ep}.hdf5", "r") as f:
            q = f["observations/qpos"][:].astype(float)
            c = f["joint_commands"][:]
            e = f["observations/effort"][:]
        arm = "left" if q[:, 7].min() < 0.035 else "right"
        g = 7 if arm == "left" else 15
        hold = np.flatnonzero((q[:, g] > 0.018) & (q[:, g] < 0.032) & (e[:, g] > 0.8) & (c[:, g] < 0.01))
        if len(hold) < 10:
            continue
        cols = list(range(7)) if arm == "left" else list(range(8, 15))
        obs[arm].append((ep, q[hold[0], cols], np.array([*r["xy"], h_center]), 1.0))
        if ep in end and args.release_weight > 0:
            # Lúc nhả vật rơi gần như thẳng đứng: chỉ ràng buộc xy (z đặt bằng chính FK → phần dư z = 0).
            obs[arm].append((ep, q[hold[-1], cols], np.array([*end[ep]["xy"], np.nan]), args.release_weight))
    result = {"method": __doc__.split("\n")[0], "cameras": str(args.cameras), "sigma_deg": [args.sigma_deg, args.sigma_deg_wrist], "release_weight": args.release_weight, "arms": {}}
    sig_q = np.radians([args.sigma_deg] * 4 + [args.sigma_deg_wrist] * 3)
    for arm, rows in obs.items():
        fit = [x for x in rows if x[0] not in held]
        ho = [x for x in rows if x[0] in held]
        x = np.zeros(11)                      # Δ (7), a, tịnh tiến bệ vai (3)

        def resid(x, items):
            out = []
            for _, q7, tgt, wt in items:
                p_ = fk.point(arm, q7 + x[:7], x[7], x[8:11])
                e_ = p_ - np.where(np.isnan(tgt), p_, tgt)
                out.append(wt * e_)
            return np.concatenate(out)

        def report(x, items, kind="grasp"):
            sel = [it for it in items if (kind == "grasp") == (not np.isnan(it[2][2]))]
            if not sel:
                return None
            r = resid(x, [(a_, b_, c_, 1.0) for a_, b_, c_, _ in sel]).reshape(-1, 3) * 1000
            return {"mean_mm": r.mean(0).round(1).tolist(), "std_mm": r.std(0).round(1).tolist(),
                    "rms_mm": float(np.sqrt((r ** 2).sum(1).mean()).round(1)), "n": len(r)}

        x[7] = args.axial
        if args.free_axial:
            for a0 in np.linspace(-0.04, 0.04, 17):
                x0 = np.r_[np.zeros(7), a0, np.zeros(3)]
                if np.sum(resid(x0, fit) ** 2) < np.sum(resid(x, fit) ** 2):
                    x = x0
        z0 = np.r_[np.zeros(7), args.axial, np.zeros(3)]
        before = {"fit": report(z0, fit), "heldout": report(z0, ho), "heldout_release_xy": report(z0, ho, "release")}
        w = 1 / (args.sigma_mm * 1e-3)
        free = list(range(7)) + ([7] if args.free_axial else []) + ([8, 9, 10] if args.base_sigma_mm > 0 else [])
        sig_b = max(args.base_sigma_mm, 1e-6) * 1e-3

        def full(x):
            return np.r_[w * resid(x, fit), x[:7] / sig_q, x[8:11] / sig_b]

        for _ in range(30):
            r = full(x)
            J = np.zeros((len(r), 11))
            for k in free:
                h = np.zeros(11)
                h[k] = 1e-5
                J[:, k] = (full(x + h) - r) / 1e-5
            step = -np.linalg.lstsq(J, r, rcond=None)[0]
            x = x + step
            x[7] = x[7] if args.free_axial else args.axial
            if np.abs(step).max() < 1e-7:
                break
        after = {"fit": report(x, fit), "heldout": report(x, ho), "heldout_release_xy": report(x, ho, "release")}
        result["arms"][arm] = {"offsets_rad": x[:7].round(5).tolist(), "offsets_deg": np.degrees(x[:7]).round(2).tolist(),
                               "base_translation_m": x[8:11].round(5).tolist(),
                               "tcp_to_object_axial_m": float(x[7]), "before_no_offset": before,
                               "after": after}
        print(arm, json.dumps(result["arms"][arm]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("ghi", args.out)


if __name__ == "__main__":
    main()
