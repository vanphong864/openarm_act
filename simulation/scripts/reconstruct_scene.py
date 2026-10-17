#!/usr/bin/env python3
"""Dựng lại cảnh (mặt hộp kê, ô giấy trắng, hộp kẹo) trong hệ robot từ hai camera đã hiệu chuẩn chung.

Đầu vào: cameras_*.json của calibrate_joint.py (có độ cao mặt hộp kê và đỉnh hộp kẹo) và các episode HDF5.
  - Mặt hộp kê: điểm depth khung đầu của hai camera gần mặt z = h_bàn → mặt phẳng bình phương tối thiểu (độ nghiêng
    còn lại, chênh giữa hai camera) và biên vùng nhìn thấy.
  - Ô giấy: pixel trắng → giao tia với mặt hộp kê → đa giác mỗi camera; lấy giao hai đa giác → hình chữ nhật.
  - Hộp kẹo: bao lồi vùng xanh → giao tia với mặt z = đỉnh hộp kẹo. Mặt bên nhìn thấy làm vết phình về phía camera;
    hai camera nhìn từ hai phía nên GIAO hai vết ≈ mặt trên → dài × rộng, tâm, yaw. Khung đầu = vị trí sinh,
    khung cuối = vị trí đặt.
Kết quả: JSON số đo kèm độ phân tán giữa các episode, dùng cho build_measured_scene.py.

Ví dụ:
  reconstruct_scene.py --raw "/media/space/New Volume/dataset/2026-10-13" \
      --cameras simulation/realdata/cameras_20261013.json --out simulation/realdata/scene_20261013.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

H, W = 480, 640


class Cam:
    def __init__(self, c: dict):
        self.fx, self.fy, self.cx, self.cy = c["fx"], c["fy"], c["cx"], c["cy"]
        self.R, self.C = np.array(c["R_cam_from_world"]), np.array(c["C_world"])

    def rays(self, u, v):
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, float)], -1) @ self.R

    def backproject(self, u, v, z):
        return self.rays(u, v) * z[:, None] + self.C

    def on_plane(self, u, v, h):
        r = self.rays(u, v)
        return self.C + r * ((h - self.C[2]) / r[:, 2])[:, None]


def green_hull(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 50) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    b = 1 + int(np.argmax(st[1:, 4]))
    if st[b, 4] < 150:
        return None
    keep = [i for i in range(1, n) if st[i, 4] >= 15 and np.linalg.norm(cen[i] - cen[b]) < 70]
    pts = np.column_stack(np.nonzero(np.isin(lab, keep)))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if hull[:, 0].min() <= 2 or hull[:, 1].min() <= 2 or hull[:, 0].max() >= W - 3 or hull[:, 1].max() >= H - 3:
        return None                     # bị cắt ở mép ảnh
    return hull


def white_hull(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 1] < 50) & (hsv[..., 2] > 170)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        if st[i, 4] > 1500 and (best is None or st[i, 4] > st[best, 4]):
            best = i
    if best is None:
        return None
    ys, xs = np.nonzero(lab == best)
    return cv2.convexHull(np.c_[xs, ys].astype(np.int32)).reshape(-1, 2)


def footprint(cam, hull, h):
    """Đa giác (xy, float32) của bao lồi pixel chiếu lên mặt z = h."""
    P = cam.on_plane(hull[:, 0].astype(float), hull[:, 1].astype(float), h)
    return cv2.convexHull(P[:, :2].astype(np.float32)).reshape(-1, 2)


def rect(poly):
    (cx, cy), (a, b), ang = cv2.minAreaRect(poly.astype(np.float32))
    yaw = np.radians(ang)
    if a < b:
        a, b, yaw = b, a, yaw + np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return [float(cx), float(cy)], [float(a), float(b)], float(np.degrees(yaw))


def intersect(p, q):
    area, inter = cv2.intersectConvexConvex(p.astype(np.float32), q.astype(np.float32))
    return (inter.reshape(-1, 2) if inter is not None and area > 0 else None), float(area)


def tin_top_pose(cam, rgb, depth_mm, h_box):
    """Tâm xy + yaw hộp kẹo từ điểm depth của MẶT TRÊN (đứng trên mặt hộp 4–30 mm); không bị phồng như vết RGB."""
    hull = green_hull(rgb)
    if hull is None:
        return None
    m = np.zeros(depth_mm.shape, np.uint8)
    cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
    m = cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0
    z = depth_mm.astype(float) * 1e-3
    vv, uu = np.nonzero(m & (z > 0.2) & (z < 1.2))
    P = cam.backproject(uu.astype(float), vv.astype(float), z[vv, uu])
    hgt = P[:, 2] - h_box
    top = (hgt > 0.004) & (hgt < 0.03)
    if top.sum() < 80:
        return None
    th = np.median(hgt[top])
    T = P[top & (np.abs(hgt - th) < 0.004)]
    if len(T) < 60:
        return None
    c, size, yaw = rect(T[:, :2])
    return {"xy": [float(np.median(T[:, 0])), float(np.median(T[:, 1]))], "yaw_deg": yaw, "top_height_m": float(th),
            "size_depth": size}


def summ(a):
    a = np.asarray(a, float)
    return {"median": float(np.median(a)), "mean": float(a.mean()), "std": float(a.std()), "n": int(len(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args()
    info = json.loads(args.cameras.read_text())
    cams = {k: Cam(v) for k, v in info["cameras"].items()}
    h_box, h_tin = info["box_top_height_m"], info["tin_top_height_m"]
    files = []
    for p in sorted(args.raw.glob("*.hdf5")):
        try:
            h5py.File(p, "r").close()
            files.append(p)
        except OSError:
            pass
    files = files[:args.max_episodes] if args.max_episodes else files
    vv, uu = np.mgrid[0:H:3, 0:W:3]

    # ---- mặt hộp kê (kiểm tra độ phẳng/nghiêng theo từng camera, biên vùng nhìn thấy)
    plane, pts_all = {}, []
    for c, cam in cams.items():
        P = []
        for p in files[::max(1, len(files) // 12)]:
            with h5py.File(p, "r") as f:
                z = f[f"observations/images/{c}_depth"][0][::3, ::3].astype(np.float64) * 1e-3
            ok = (z > 0.2) & (z < 1.2)
            Q = cam.backproject(uu[ok].astype(float), vv[ok].astype(float), z[ok])
            P.append(Q[np.abs(Q[:, 2] - h_box) < 0.015])
        P = np.concatenate(P)
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        k = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        plane[c] = {"dz_dx": float(k[0]), "dz_dy": float(k[1]), "z_at_origin": float(k[2]),
                    "z_at_work_center": float(k[0] * 0.3 + k[2]), "resid_mm": float((P[:, 2] - A @ k).std() * 1e3),
                    "x_range_seen": [float(np.percentile(P[:, 0], 0.5)), float(np.percentile(P[:, 0], 99.5))],
                    "y_range_seen": [float(np.percentile(P[:, 1], 0.5)), float(np.percentile(P[:, 1], 99.5))]}
        pts_all.append(P)
    P = np.concatenate(pts_all)
    surface = {"height_m": h_box, "per_camera": plane,
               "x_range_seen": [float(np.percentile(P[:, 0], 0.2)), float(np.percentile(P[:, 0], 99.8))],
               "y_range_seen": [float(np.percentile(P[:, 1], 0.2)), float(np.percentile(P[:, 1], 99.8))],
               "note": "Biên chỉ là phần mặt hộp nhìn thấy (gộp hai camera); mép thật có thể nằm ngoài khung."}
    print("mặt hộp kê:", json.dumps(surface), flush=True)

    # ---- ô giấy và hộp kẹo
    pads, tins = [], {"start": [], "end": []}
    for p in files:
        with h5py.File(p, "r") as f:
            T = f["observations/qpos"].shape[0]
            for phase, t in (("start", 0), ("end", T - 1)):
                imgs = {c: f[f"observations/images/{c}_rgb"][t] for c in cams}
                if phase == "start":
                    hulls = {c: white_hull(imgs[c]) for c in cams}
                    fps = {c: footprint(cams[c], hh, h_box) for c, hh in hulls.items() if hh is not None}
                    rec = {"episode": p.stem, "per_camera": {c: rect(fp) for c, fp in fps.items()},
                           "image_center_chest": (hulls["chest"].mean(0).tolist() if hulls.get("chest") is not None
                                                  else None)}
                    if len(fps) == 2:
                        inter, area = intersect(fps["chest"], fps["front"])
                        if inter is not None:
                            rec["fused"] = rect(inter)
                    pads.append(rec)
                fps = {c: footprint(cam, hh, h_tin) for c, cam in cams.items()
                       if (hh := green_hull(imgs[c])) is not None}
                rec = {"episode": p.stem, "per_camera": {c: rect(fp) for c, fp in fps.items()}}
                for c in ("front", "chest"):
                    tp = tin_top_pose(cams[c], imgs[c], f[f"observations/images/{c}_depth"][t], h_box)
                    if tp:
                        rec["depth_pose"] = {**tp, "camera": c}
                        break
                if len(fps) == 2:
                    inter, area = intersect(fps["chest"], fps["front"])
                    if inter is not None and len(inter) >= 3:
                        rec["fused"] = rect(inter)
                tins[phase].append(rec)
        print(p.stem, "ok", flush=True)

    # Hộp kê + ô giấy bị xê dịch vài lần trong ngày (vị trí ô giấy trong ảnh nhảy cùng lúc ở hai camera):
    # gom episode liên tiếp có tâm ô giấy (ảnh camera ngực) lệch < 4 px thành một đoạn.
    segments, cur = [], None
    for r in pads:
        c = r.get("image_center_chest")
        if c is None:
            continue
        if cur is None or np.linalg.norm(np.array(c) - cur["ref"]) > 4.0:
            cur = {"ref": np.array(c), "episodes": []}
            segments.append(cur)
        cur["episodes"].append(r["episode"])
    seg_out = []
    for k, sg in enumerate(segments):
        rows = [r["fused"] for r in pads if r["episode"] in sg["episodes"] and "fused" in r]
        if not rows:
            continue
        seg_out.append({"id": k, "episodes": sg["episodes"], "pad_center": np.median([x[0] for x in rows], 0).tolist(),
                        "pad_size": np.median([x[1] for x in rows], 0).tolist(),
                        "pad_yaw_deg": float(np.median([x[2] for x in rows])),
                        "pad_center_std_mm": (np.std([x[0] for x in rows], 0) * 1000).round(2).tolist()})
    pf = [r["fused"] for r in pads if "fused" in r]
    pad = {"center": np.median([x[0] for x in pf], 0).tolist(), "size": np.median([x[1] for x in pf], 0).tolist(),
           "yaw_deg": float(np.median([x[2] for x in pf])), "n": len(pf),
           "center_std_mm": (np.std([x[0] for x in pf], 0) * 1000).tolist(), "segments": seg_out,
           "per_camera_center": {c: np.median([r["per_camera"][c][0] for r in pads if c in r["per_camera"]],
                                              0).tolist() for c in cams},
           "per_camera_size": {c: np.median([r["per_camera"][c][1] for r in pads if c in r["per_camera"]],
                                            0).tolist() for c in cams}}
    print("ô giấy:", json.dumps(pad), flush=True)
    ts = [r["fused"] for r in tins["start"] + tins["end"] if "fused" in r]
    tin = {"length": summ([x[1][0] for x in ts]), "width": summ([x[1][1] for x in ts]),
           "thickness_m": h_tin - h_box,
           "per_camera_length": {c: summ([r["per_camera"][c][1][0] for r in tins["start"] if c in r["per_camera"]])
                                 for c in cams}}
    print("hộp kẹo:", json.dumps(tin), flush=True)
    poses = {}
    for phase in ("start", "end"):
        poses[phase] = [{"episode": r["episode"], **r["depth_pose"]} for r in tins[phase] if "depth_pose" in r]
    pad["per_episode"] = {r["episode"]: {"center": r["fused"][0], "size": r["fused"][1], "yaw_deg": r["fused"][2]}
                          for r in pads if "fused" in r}
    result = {"source": str(args.raw), "cameras": str(args.cameras), "surface": surface, "pad": pad,
              "tin": tin, "tin_poses": poses, "pad_detections": pads, "tin_detections": tins}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("ghi", args.out)


if __name__ == "__main__":
    main()
