#!/usr/bin/env python3
"""Ảnh nền thật (RGB + depth) cho từng camera cố định: trung vị theo pixel trên nhiều khung, bỏ robot và hộp kẹo.

Pixel bị che được loại bằng ba cách: (1) render robot + hộp kẹo trong scene đo đạc (camera, offset khớp và bệ vai đã
hiệu chuẩn; hộp kẹo ở tư thế đầu đã đo tới lúc gắp và tư thế cuối từ lúc thả), nới rộng vài px; (2) màu hộp kẹo;
(3) depth: mẫu gần camera hơn trung vị depth của pixel đó quá --occluder-mm (tay robot lệch mô hình, hộp kẹo trong tay,
tay người) không được dùng.
Depth: trung vị các giá trị hợp lệ; pixel mất depth ở đa số khung (giấy trắng, mép xa) giữ 0 như cảm biến thật.
Kết quả dùng cho camera sim kiểu "ghép nền thật": sim chỉ render robot + vật, ghép theo z-buffer lên nền này.

Ví dụ:
  make_real_backgrounds.py --raw "/media/space/New Volume/dataset/2026-10-13" --out simulation/realdata
  make_real_backgrounds.py --raw "/media/space/New Volume/dataset/2026-10-13" --out simulation/realdata/bg_episodes \
      --per-episode
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

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from openarm_sim.measured_scene import DYNAMIC_GROUP, compile_measured  # noqa: E402
from openarm_sim.runner import Sim  # noqa: E402
from replay_real_20261013 import load_cfg, set_object  # noqa: E402

H, W = 480, 640


def green(rgb, max_blob_px=6000):
    """Mask of the (bright, saturated green) tin. The V threshold keeps out the dark table, which reads cyan with S > 80
    when the chest camera's white balance drifts (about half the episodes); without it the tin mask covered half the
    table and the per-episode background was inpainted there. Blobs larger than a tin (the light-blue box behind the
    table) are not the tin."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 50) & (hsv[..., 0] <= 105) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 150)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_blob_px
    return cv2.dilate(keep[labels], np.ones((15, 15), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config/scene_20261013.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=25, help="số mẫu tối đa giữ cho mỗi pixel")
    ap.add_argument("--stride", type=int, default=40, help="lấy mỗi N khung trong episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", nargs="*", help="chỉ dùng các episode này (vd một phiên: hộp kê bị xê dịch giữa phiên)")
    ap.add_argument("--per-episode", action="store_true",
                    help="mỗi episode một nền riêng: background_<cam>_<episode>.png (hộp/giấy xê dịch giữa các phiên)")
    ap.add_argument("--dilate", type=int, default=15, help="px nới rộng mask robot/hộp kẹo render")
    ap.add_argument("--occluder-mm", type=float, default=12.0,
                    help="mẫu có depth gần hơn trung vị của pixel quá ngưỡng này là vật che, không phải nền")
    ap.add_argument("--stage1-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/bg_stage1",
                    help="--per-episode: nền riêng từng episode (bước 1) lưu ở đây; nền cuối (bước 2) ghi vào --out")
    ap.add_argument("--neighbours", type=int, default=10, help="--per-episode: số episode cùng phiên dùng ở bước 2")
    ap.add_argument("--outlier", type=float, default=25.0,
                    help="--per-episode: pixel lệch trung vị các episode cùng phiên quá ngưỡng này (0-255) bị thay")
    ap.add_argument("--consensus-only", action="store_true", help="--per-episode: chỉ chạy bước 2 trên bước 1 có sẵn")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.raw)
    m = compile_measured(cfg)
    sim = Sim(m, cfg)
    calib = cfg["_measured"]["cameras"]["cameras"]
    tin = {"starts": {r["episode"]: r for r in cfg["_measured"]["scene"]["tin_poses"]["start"]},
           "ends": {r["episode"]: r for r in cfg["_measured"]["scene"]["tin_poses"]["end"]},
           "z": cfg["_measured"]["cameras"]["box_top_height_m"] + cfg["objects"]["mentos_tin"]["size"][2] / 2 + 1e-4,
           "geoms": [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("mentos_tin").id]}
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()   # robot + task objects, as CompositeCamera renders them
    opt.geomgroup[:] = 0
    opt.geomgroup[DYNAMIC_GROUP] = opt.geomgroup[2] = 1
    files = []
    for p in sorted(args.raw.glob("*.hdf5")):
        if args.episodes and p.stem not in args.episodes:
            continue
        try:
            h5py.File(p, "r").close()
            files.append(p)
        except OSError:
            pass
    if not args.per_episode:
        report = {"all": build(files, "", args, args.out, calib, m, sim, r, opt, tin)}
        (args.out / "backgrounds.json").write_text(json.dumps(report, indent=1))
        return
    report_path = args.out / "backgrounds_per_episode.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for p in files if not args.consensus_only else []:
        report[f"_{p.stem}"] = build([p], f"_{p.stem}", args, args.stage1_dir, calib, m, sim, r, opt, tin)
    for p in files:
        report.setdefault(f"_{p.stem}", {})["consensus"] = consensus(p.stem, args, calib, cfg["_measured"]["scene"])
    args.out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dict(sorted(report.items())), indent=1))


def sessions(scene: dict) -> list[list[str]]:
    """Episodes grouped into recording sessions: consecutive episodes whose measured pad (box) did not move."""
    pads = scene["pad"]["per_episode"]
    groups, previous = [], None
    for episode in sorted(pads):
        centre = np.array(pads[episode]["center"][:2])
        if previous is None or np.linalg.norm(centre - previous) > 0.006:
            groups.append([])
        groups[-1].append(episode)
        previous = centre
    return groups


def consensus(episode, args, calib, scene):
    """Step 2 of --per-episode: an episode's own background (step 1) still holds what was in view most of the time
    at some pixels: the arm's shadows where it hovered, a missed tin edge, an arm part the masks missed. The other
    episodes of the same session saw the same box and table with the arm elsewhere: their backgrounds, brought to
    this episode's colours (per-channel gains on the pixels both saw), give a per-pixel median; pixels of the
    episode's own background far from it, or never seen, take the median instead."""
    load = lambda c, e: (cv2.imread(str(args.stage1_dir / f"background_{c}_{e}_rgb.png"))[..., ::-1].astype(np.float32),  # noqa: E731
                         cv2.imread(str(args.stage1_dir / f"background_{c}_{e}_depth.png"), cv2.IMREAD_UNCHANGED),
                         cv2.imread(str(args.stage1_dir / f"background_{c}_{e}_samples.png"), cv2.IMREAD_UNCHANGED))
    group = next((g for g in sessions(scene) if episode in g), [episode])
    available = [e for e in group if e != episode and (args.stage1_dir / f"background_{list(calib)[0]}_{e}_rgb.png").exists()]
    index = {e: i for i, e in enumerate(group)}
    neighbours = sorted(available, key=lambda e: abs(index[e] - index.get(episode, 0)))[:args.neighbours]
    ok = lambda x: ((x > 20) & (x < 235)).all(-1)  # noqa: E731
    report = {"neighbours": len(neighbours)}
    for c in calib:
        rgb, dep, samples = load(c, episode)
        seen = samples > 0
        stack_rgb, stack_dep = [np.where(seen[..., None], rgb, np.nan)], [np.where(seen & (dep > 0), dep, np.nan)]
        stack_seen = [seen]
        for e in neighbours:
            rgb_e, dep_e, samples_e = load(c, e)
            seen_e = samples_e > 0
            common = seen & seen_e & ok(rgb) & ok(rgb_e)
            if common.sum() < 5000:
                continue
            g = np.clip(np.median(rgb[common] / np.maximum(rgb_e[common], 1.0), axis=0), 0.5, 2.0)
            stack_rgb.append(np.where(seen_e[..., None], np.clip(rgb_e * g, 0, 255), np.nan))
            stack_dep.append(np.where(seen_e & (dep_e > 0), dep_e, np.nan))
            stack_seen.append(seen_e)
        n_seen = np.sum(stack_seen, 0)
        with np.errstate(all="ignore"):
            median_rgb = np.nanmedian(np.stack(stack_rgb), 0)
            median_dep = np.nanmedian(np.stack(stack_dep), 0)
            dep_valid = np.sum([np.isfinite(d) for d in stack_dep], 0) >= 0.5 * np.maximum(n_seen, 1)
        usable = n_seen >= min(3, len(stack_rgb))
        outlier = seen & (np.abs(rgb - np.nan_to_num(median_rgb)).max(-1) > args.outlier)
        outlier = cv2.dilate(outlier.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        replace = usable & (outlier | ~seen)
        out_rgb = np.where(replace[..., None], np.nan_to_num(median_rgb), rgb).astype(np.uint8)
        out_dep = np.where(replace, np.where(dep_valid, np.nan_to_num(median_dep), 0), dep).astype(np.uint16)
        out_samples = np.where(replace, 250, samples).astype(np.uint8)
        args.out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out / f"background_{c}_{episode}_rgb.png"), out_rgb[..., ::-1])
        cv2.imwrite(str(args.out / f"background_{c}_{episode}_depth.png"), out_dep)
        cv2.imwrite(str(args.out / f"background_{c}_{episode}_samples.png"), out_samples)
        report[c] = {"episodes": len(stack_rgb), "replaced_fraction": round(float(replace.mean()), 4),
                     "pixels_never_seen": int((out_samples == 0).sum())}
    print(episode, "consensus", report, flush=True)
    return report


def tin_schedule(f, n):
    """Per frame: where the tin's measured pose is masked: 'start' until the gripper re-opens (the tin often stays
    put for a while after the close; masking its start footprint while it is carried only costs samples there),
    then 'end' (on the pad)."""
    cmd = f["joint_commands"][:] if "joint_commands" in f else f["action"][:]
    grip = 7 if cmd[:, 7].min() < cmd[:, 15].min() else 15
    closed = cmd[:, grip] < 0.03
    if not closed.any():
        return ["start"] * n
    close = int(np.argmax(closed))
    reopen = np.flatnonzero(~closed[close:])
    reopen = close + int(reopen[0]) if len(reopen) else n
    return ["start" if t < reopen - 10 else "end" for t in range(n)]


def build(files, tag, args, out_dir, calib, m, sim, r, opt, tin):
    rng = np.random.default_rng(args.seed)
    stride = max(5, args.stride // 4) if tag else args.stride
    K = args.samples
    buf = {c: (np.zeros((K, H, W, 3), np.uint8), np.zeros((K, H, W), np.uint16), np.zeros((H, W), np.int32))
           for c in calib}
    order = rng.permutation(len(files))
    kernel = np.ones((args.dilate, args.dilate), np.uint8)
    tin_kernel = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)
    used = 0
    ref = {}          # first frame of the group, per camera: (rgb, unoccluded) — the colour reference
    gains = {c: [] for c in calib}

    def observe(f, p, t, q, where):
        sim.reset(qpos16=q[t])
        pose = {"start": tin["starts"], "end": tin["ends"]}.get(where[t], {}).get(p.stem)
        if pose:
            set_object(m, sim.d, "mentos_tin", pose["xy"], pose["yaw_deg"], tin["z"])
        else:   # pose not measured: out of sight (colour and depth tests only)
            set_object(m, sim.d, "mentos_tin", [100.0, 100.0], 0, -100.0)
        mujoco.mj_forward(m, sim.d)
        out = {}
        for c in calib:
            r.enable_segmentation_rendering()
            r.update_scene(sim.d, c, scene_option=opt)
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            tin_px = np.isin(seg, tin["geoms"])
            robot = (seg >= 0) & ~tin_px
            r.enable_depth_rendering()
            r.update_scene(sim.d, c, scene_option=opt)
            zr = r.render()
            r.disable_depth_rendering()
            rgb = f[f"observations/images/{c}_rgb"][t]
            dep = f[f"observations/images/{c}_depth"][t]
            # Chỉ coi là robot che khi robot sim ở TRƯỚC depth thật (hoặc depth thật mất): phần robot bị
            # hộp kê che ngoài đời (cột thân sau hộp) vẫn được lấy mẫu làm nền.
            zd = dep.astype(np.float32) * 1e-3
            robot &= (zd == 0) | (zr < zd + 0.03)
            # the measured tin poses are ~1-2 cm off: a wider margin (its footprint gets samples after the release)
            out[c] = rgb, dep, ((cv2.dilate(robot.astype(np.uint8), kernel) > 0) | green(rgb)
                                | (cv2.dilate(tin_px.astype(np.uint8), tin_kernel) > 0))
        return out

    for i in order:
        p = files[i]
        with h5py.File(p, "r") as f:
            q = f["observations/qpos"][:]
            where = tin_schedule(f, len(q))
            ts = rng.choice(np.arange(len(q)), max(1, len(q) // stride), replace=False)
            for t in [0] + [t for t in ts if t != 0]:
                for c, (rgb, dep, occ) in observe(f, p, t, q, where).items():
                    # The camera's auto white balance / exposure drift within an episode (the dark arm entering the
                    # view): bring every frame to the reference frame's colours with per-channel gains measured on
                    # the background visible in both, so the median does not mix colour states (patches where the
                    # background was only seen late in the episode).
                    if c not in ref:
                        ref[c] = rgb.astype(np.float32), ~occ
                    ok = lambda x: ((x > 20) & (x < 235)).all(-1)  # noqa: E731
                    common = ~occ & ref[c][1] & ok(rgb) & ok(ref[c][0])
                    if common.sum() > 2000:
                        g = np.clip(np.median(ref[c][0][common] / np.maximum(rgb[common], 1.0), axis=0), 0.6, 1.6)
                        gains[c].append(g)
                        rgb = np.clip(rgb * g, 0, 255).astype(np.uint8)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"frames_used": used, "cameras": {}}
    for c, (rgbs, deps, cnt) in buf.items():
        valid_slot = np.arange(K)[:, None, None] < cnt[None]
        # Occluders the masks missed sit in front of the surface: drop samples clearly nearer than the pixel's median.
        with np.errstate(all="ignore"):
            z_med = np.nanmedian(np.where(valid_slot & (deps > 0), deps.astype(np.float32), np.nan), axis=0)
        near = valid_slot & (deps > 0) & (deps.astype(np.float32) < z_med[None] - args.occluder_mm)
        # The black arm often reads no depth at all: where the surface itself usually has depth, so is suspect.
        surface = (valid_slot & (deps > 0)).sum(0) >= 0.5 * np.maximum(cnt, 1)
        near |= valid_slot & (deps == 0) & surface[None]
        keep = valid_slot & ~near
        order_k = np.argsort(~keep, axis=0, kind="stable")   # kept samples first
        rgbs = np.take_along_axis(rgbs, order_k[..., None], 0)
        deps = np.take_along_axis(deps, order_k, 0)
        cnt = keep.sum(0).astype(np.int32)
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
        cv2.imwrite(str(out_dir / f"background_{c}{tag}_rgb.png"), rgb_bg[..., ::-1])
        cv2.imwrite(str(out_dir / f"background_{c}{tag}_depth.png"), dep_bg)
        cv2.imwrite(str(out_dir / f"background_{c}{tag}_samples.png"), (np.minimum(cnt, K) * (255 // K)).astype(np.uint8))
        g = np.array(gains[c]) if gains[c] else np.ones((1, 3))
        report["cameras"][c] = {"pixels_never_seen": int((cnt == 0).sum()), "median_samples": float(np.median(cnt)),
                                "depth_invalid_fraction": float((dep_bg == 0).mean()),
                                "colour_gain_p5_p95": np.percentile(g, [5, 95], axis=0).round(3).tolist()}
        print(tag, c, report["cameras"][c], flush=True)
    return report




if __name__ == "__main__":
    main()
