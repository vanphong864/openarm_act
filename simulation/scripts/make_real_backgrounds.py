#!/usr/bin/env python3
"""Ảnh nền thật (RGB + depth) cho từng camera cố định: trung vị theo pixel trên nhiều khung, bỏ robot và hộp kẹo.

Pixel bị robot che xác định bằng render robot (qpos thật) với camera đã hiệu chuẩn, nới rộng vài px; hộp kẹo theo màu.
Depth: trung vị các giá trị hợp lệ; pixel mất depth ở đa số khung (giấy trắng, mép xa) giữ 0 như cảm biến thật.
Kết quả dùng cho camera sim kiểu "ghép nền thật": sim chỉ render robot + vật, ghép theo z-buffer lên nền này.

Ví dụ:
  make_real_backgrounds.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --cameras simulation/realdata/cameras_20261013.json --out simulation/realdata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openarm_sim.calibrated_camera import add_camera  # noqa: E402
from openarm_sim.robot_surface import ARM16  # noqa: E402

ROBOT = Path(__file__).resolve().parent.parent / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"
H, W = 480, 640


def green(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 50) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80)).astype(np.uint8)
    return cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=25, help="số mẫu tối đa giữ cho mỗi pixel")
    ap.add_argument("--stride", type=int, default=40, help="lấy mỗi N khung trong episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", nargs="*", help="chỉ dùng các episode này (vd một phiên: hộp kê bị xê dịch giữa phiên)")
    ap.add_argument("--per-episode", action="store_true",
                    help="mỗi episode một nền riêng: background_<cam>_<episode>.png (hộp/giấy xê dịch giữa các phiên)")
    args = ap.parse_args()
    calib = json.loads(args.cameras.read_text())["cameras"]
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    for name, c in calib.items():
        add_camera(spec.worldbody, name, c)
    m = spec.compile()
    d = mujoco.MjData(m)
    qadr = [m.joint(f"openarm_{s}_joint{i}").qposadr[0] for s in ("left", "right") for i in range(1, 8)]
    fadr = {s: [m.joint(f"openarm_{s}_finger_joint{i}").qposadr[0] for i in (1, 2)] for s in ("left", "right")}
    r = mujoco.Renderer(m, H, W)
    files = []
    for p in sorted(args.raw.glob("*.hdf5")):
        if args.episodes and p.stem not in args.episodes:
            continue
        try:
            h5py.File(p, "r").close()
            files.append(p)
        except OSError:
            pass
    groups = [[p] for p in files] if args.per_episode else [files]
    report = {}
    for grp in groups:
        tag = f"_{grp[0].stem}" if args.per_episode else ""
        report[tag or "all"] = build(grp, tag, args, calib, m, d, r, qadr, fadr)
    (args.out / ("backgrounds_per_episode.json" if args.per_episode else "backgrounds.json")).write_text(
        json.dumps(report, indent=1))


def build(files, tag, args, calib, m, d, r, qadr, fadr):
    rng = np.random.default_rng(args.seed)
    stride = max(5, args.stride // 4) if tag else args.stride
    jobs = [(p, t) for p in files for t in range(0, 10 ** 6, stride)]
    K = args.samples
    buf = {c: (np.zeros((K, H, W, 3), np.uint8), np.zeros((K, H, W), np.uint16), np.zeros((H, W), np.int32))
           for c in calib}
    order = rng.permutation(len(files))
    kernel = np.ones((11, 11), np.uint8)
    used = 0
    for i in order:
        p = files[i]
        with h5py.File(p, "r") as f:
            q = f["observations/qpos"][:]
            ts = rng.choice(np.arange(len(q)), max(1, len(q) // stride), replace=False)
            for t in ts:
                d.qpos[:] = 0
                d.qpos[qadr] = q[t, ARM16]
                for s, g in (("left", 7), ("right", 15)):
                    d.qpos[fadr[s]] = q[t, g]
                mujoco.mj_forward(m, d)
                for c in calib:
                    r.enable_segmentation_rendering()
                    r.update_scene(d, c)
                    robot = r.render()[..., 0] >= 0
                    r.disable_segmentation_rendering()
                    r.enable_depth_rendering()
                    r.update_scene(d, c)
                    zr = r.render()
                    r.disable_depth_rendering()
                    rgb = f[f"observations/images/{c}_rgb"][t]
                    dep = f[f"observations/images/{c}_depth"][t]
                    # Chỉ coi là robot che khi robot sim ở TRƯỚC depth thật (hoặc depth thật mất): phần robot bị
                    # hộp kê che ngoài đời (cột thân sau hộp) vẫn được lấy mẫu làm nền.
                    zd = dep.astype(np.float32) * 1e-3
                    robot &= (zd == 0) | (zr < zd + 0.03)
                    occ = (cv2.dilate(robot.astype(np.uint8), kernel) > 0) | green(rgb)
                    rgbs, deps, cnt = buf[c]
                    sel = ~occ & (cnt < K)
                    ys, xs = np.nonzero(sel)
                    slot = cnt[ys, xs]
                    rgbs[slot, ys, xs] = rgb[ys, xs]
                    deps[slot, ys, xs] = dep[ys, xs]
                    cnt[ys, xs] += 1
                used += 1
        if all((b[2] >= K).mean() > 0.995 for b in buf.values()):
            break
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"frames_used": used, "cameras": {}}
    for c, (rgbs, deps, cnt) in buf.items():
        n = np.maximum(cnt, 1)
        valid_slot = np.arange(K)[:, None, None] < cnt[None]
        rgbf = np.where(valid_slot[..., None], rgbs.astype(np.float32), np.nan)
        rgb_bg = np.nan_to_num(np.nanmedian(rgbf, axis=0)).astype(np.uint8)
        df = deps.astype(np.float32)
        good = valid_slot & (df > 0)
        frac_valid = good.sum(0) / n
        df[~good] = np.nan
        with np.errstate(all="ignore"):
            dep_bg = np.nanmedian(df, axis=0)
        dep_bg = np.where((frac_valid >= 0.5) & np.isfinite(dep_bg), dep_bg, 0).astype(np.uint16)
        cv2.imwrite(str(args.out / f"background_{c}{tag}_rgb.png"), rgb_bg[..., ::-1])
        cv2.imwrite(str(args.out / f"background_{c}{tag}_depth.png"), dep_bg)
        cv2.imwrite(str(args.out / f"background_{c}{tag}_samples.png"), (np.minimum(cnt, K) * (255 // K)).astype(np.uint8))
        report["cameras"][c] = {"pixels_never_seen": int((cnt == 0).sum()), "median_samples": float(np.median(cnt)),
                                "depth_invalid_fraction": float((dep_bg == 0).mean())}
        print(tag, c, report["cameras"][c], flush=True)
    return report




if __name__ == "__main__":
    main()
