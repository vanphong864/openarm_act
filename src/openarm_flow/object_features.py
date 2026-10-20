"""Object-centric policy inputs: where the tin and the target pad are, in the robot frame.

No learned detector. At the first frame of an episode (robot at rest, nothing occluded) the scene is
estimated from the two calibrated RGB views:

  - colours are white-balanced on the white pad (the cameras' auto white balance drifts strongly);
  - pad: the white blob on the box top, centre and yaw from its outline projected onto the box plane;
  - tin: a 3D box of known size is fitted so its projected silhouette best overlaps the tin-coloured pixels
    (green rim + white label) in both views jointly.

Measured on the 88 recorded 2026-10-13 episodes against the camera-measured tin poses: ~10 mm random error
(the cross-camera calibration limit) plus a stable bias of 10-15 mm, yaw ~3 deg; pad 4-5 mm.

Per-frame features combine that static scene with the current joint state through forward kinematics
(TCP -> tin and TCP -> pad for each arm), so they stay meaningful when the arm drifts off a demonstrated
path. Calibration lives in the dataset/release metadata: train and deployment use identical numbers.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np
import torch

FEATURE_NAMES = (
    "valid", "tin_x", "tin_y", "tin_cos2yaw", "tin_sin2yaw", "tin_fit",
    "pad_x", "pad_y", "pad_cos2yaw", "pad_sin2yaw", "tin_to_pad_x", "tin_to_pad_y",
    "left_tcp_to_tin_x", "left_tcp_to_tin_y", "left_tcp_to_tin_z", "left_tcp_to_pad_x", "left_tcp_to_pad_y",
    "right_tcp_to_tin_x", "right_tcp_to_tin_y", "right_tcp_to_tin_z", "right_tcp_to_pad_x", "right_tcp_to_pad_y",
)
FEATURE_DIM = len(FEATURE_NAMES)
REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ObjectConfig:
    cameras: dict = field(default_factory=dict)        # base camera name -> fx, fy, cx, cy, R_cam_from_world, C_world
    box_top_m: float = 0.154
    box_xy: tuple = (0.15, 0.50, -0.40, 0.41)         # x0, x1, y0, y1 of the searched box top region
    tin_size_m: tuple = (0.088, 0.044, 0.016)
    joint_offsets_rad: tuple = (0.0,) * 16              # encoder -> URDF joint zero (calibrated)
    base_translation_m: tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    @classmethod
    def from_measurements(cls, cameras=REPO / "simulation/realdata/cameras_20261013.json",
                          offsets=REPO / "simulation/realdata/joint_offsets_20261013.json"):
        cal = json.loads(Path(cameras).read_text())
        jo = json.loads(Path(offsets).read_text())
        off = [0.0] * 16
        for i, side in enumerate(("left", "right")):
            off[8 * i:8 * i + 7] = jo["arms"][side]["offsets_rad"]
        keys = ("fx", "fy", "cx", "cy", "R_cam_from_world", "C_world")
        return cls(cameras={c: {k: v[k] for k in keys} for c, v in cal["cameras"].items()},
                   box_top_m=float(cal["box_top_height_m"]), joint_offsets_rad=tuple(off),
                   base_translation_m=tuple(tuple(jo["arms"][s]["base_translation_m"]) for s in ("left", "right")))

    def metadata(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_metadata(cls, meta):
        meta = dict(meta)
        for key in ("box_xy", "tin_size_m", "joint_offsets_rad"):
            meta[key] = tuple(meta[key])
        meta["base_translation_m"] = tuple(tuple(v) for v in meta["base_translation_m"])
        return cls(**meta)


def _project(cam, points):
    R, C = np.asarray(cam["R_cam_from_world"]), np.asarray(cam["C_world"])
    pc = (np.atleast_2d(points) - C) @ R.T
    return np.c_[cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"], cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]], pc[:, 2]


def _ray_plane(cam, u, v, z):
    d = np.array([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], 1.0])
    R, C = np.asarray(cam["R_cam_from_world"]), np.asarray(cam["C_world"])
    w = R.T @ d
    return (C + (z - C[2]) / w[2] * w)[:2]


class SceneEstimator:
    def __init__(self, cfg: ObjectConfig, image_size=(480, 640)):
        self.cfg, self.shape = cfg, image_size
        x0, x1, y0, y1 = cfg.box_xy
        self.region = {}
        for name, cam in cfg.cameras.items():
            uv, _ = _project(cam, np.array([[x0, y0, cfg.box_top_m], [x1, y0, cfg.box_top_m],
                                            [x1, y1, cfg.box_top_m], [x0, y1, cfg.box_top_m]]))
            m = np.zeros(image_size, np.uint8)
            cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
            self.region[name] = m > 0
        sx, sy, sz = cfg.tin_size_m
        self.corners = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0, sz)])

    def _white_balance(self, rgb, name):
        f = rgb.astype(np.float32)
        lum = f.mean(-1)
        region = self.region[name]
        cand = ((lum > np.percentile(lum[region], 97)) & region).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
        if n < 2:
            return rgb, None
        i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        ref = f[lab == i].mean(0)
        return np.clip(f * (ref.mean() / np.maximum(ref, 1)), 0, 255).astype(np.uint8), lab == i

    def _silhouette(self, name, x, y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        uv, _ = _project(self.cfg.cameras[name], self.corners @ R.T + [x, y, self.cfg.box_top_m])
        m = np.zeros(self.shape, np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        return m > 0

    def estimate(self, rgb_by_camera):
        """{camera: uint8 [H,W,3]} -> dict(tin=(x, y, yaw, fit) | None, pad=(x, y, yaw) | None)."""
        cfg, top = self.cfg, self.cfg.box_top_m
        masks, pads = {}, []
        for name, rgb in rgb_by_camera.items():
            wb, _ = self._white_balance(np.asarray(rgb), name)
            hsv = cv2.cvtColor(wb, cv2.COLOR_RGB2HSV)
            green = (hsv[..., 0] >= 40) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 50)
            white = (hsv[..., 1] < 50) & (hsv[..., 2] > 170)
            region = self.region[name]
            # pad: the largest white blob on the box top (the tin label is white too, but much smaller)
            n, lab, st, cent = cv2.connectedComponentsWithStats(
                cv2.morphologyEx((white & region).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
            pad = None
            if n > 1:
                i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
                if st[i, cv2.CC_STAT_AREA] >= 1500:
                    pad = cv2.dilate((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                    contour = max(cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)[:, 0]
                    plane = np.array([_ray_plane(cfg.cameras[name], u, v, top) for u, v in contour], np.float32)
                    _, _, angle = cv2.minAreaRect(plane * 1000)
                    px, py = _ray_plane(cfg.cameras[name], *cent[i], top)
                    pads.append((float(px), float(py), np.radians(angle)))
            masks[name] = (green & region, white & region & ~(pad if pad is not None else False))
        pad = None
        if pads:
            xy = np.mean([p[:2] for p in pads], 0)
            yaw = np.angle(np.mean([np.exp(2j * p[2]) for p in pads])) / 2   # rectangle yaw is 90/180 deg ambiguous
            pad = (float(xy[0]), float(xy[1]), float(yaw))
        coarse = None
        for name in sorted(masks, key=lambda n: n != "front"):       # front view first: most reliable centroid
            g = cv2.morphologyEx(masks[name][0].astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, lab, st, cent = cv2.connectedComponentsWithStats(g)
            if n > 1:
                i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
                if st[i, cv2.CC_STAT_AREA] >= 80:
                    coarse = _ray_plane(cfg.cameras[name], *cent[i], top + cfg.tin_size_m[2])
                    break
        if coarse is None:
            return {"tin": None, "pad": pad}
        target = {}
        for name, (g, w) in masks.items():
            near = self._silhouette(name, *coarse, 0.0) | self._silhouette(name, *coarse, np.pi / 2)
            near = cv2.dilate(near.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
            target[name] = (g | w) & near

        def score(x, y, yaw):
            total = 0.0
            for name, t in target.items():
                sil = self._silhouette(name, x, y, yaw)
                total += (sil & t).sum() / max(1, (sil | t).sum())
            return total / len(target)

        best = max(((score(x, y, a), (x, y, a)) for x in coarse[0] + np.linspace(-0.04, 0.04, 9)
                    for y in coarse[1] + np.linspace(-0.04, 0.04, 9) for a in np.radians(np.arange(0, 180, 15))),
                   key=lambda t: t[0])
        for step, astep in ((0.005, np.radians(5)), (0.002, np.radians(2))):
            for _ in range(3):
                x0, y0, a0 = best[1]
                for dx in (-step, 0, step):
                    for dy in (-step, 0, step):
                        for da in (-astep, 0, astep):
                            v = score(x0 + dx, y0 + dy, a0 + da)
                            if v > best[0]:
                                best = (v, (x0 + dx, y0 + dy, a0 + da))
        x, y, yaw = best[1]
        return {"tin": (float(x), float(y), float(yaw % np.pi), float(best[0])), "pad": pad}


class ObjectFeatures:
    """Static scene (estimated once per episode) + joint state -> FEATURE_DIM numbers per frame."""

    def __init__(self, cfg: ObjectConfig):
        from openarm_flow.kinematics import OpenArmKinematics
        self.cfg = cfg
        self.fk = OpenArmKinematics()
        self.offsets = np.asarray(cfg.joint_offsets_rad, np.float32)
        self.base = np.asarray(cfg.base_translation_m, np.float32)

    def tcp(self, states):
        q = torch.from_numpy(np.asarray(states, np.float32) + self.offsets)
        with torch.no_grad():
            pos, _ = self.fk(q)
        return pos.numpy() + self.base                                   # [T, 2 arms, 3]

    def __call__(self, scene, states):
        states = np.atleast_2d(np.asarray(states, np.float32))
        out = np.zeros((len(states), FEATURE_DIM), np.float32)
        tin, pad = scene.get("tin"), scene.get("pad")
        if tin is None or pad is None:
            return out                                                   # valid = 0: the policy relies on images
        tz = self.cfg.box_top_m + self.cfg.tin_size_m[2]
        tcp = self.tcp(states)
        out[:, 0] = 1
        out[:, 1:6] = [tin[0], tin[1], np.cos(2 * tin[2]), np.sin(2 * tin[2]), tin[3]]
        out[:, 6:10] = [pad[0], pad[1], np.cos(2 * pad[2]), np.sin(2 * pad[2])]
        out[:, 10:12] = [pad[0] - tin[0], pad[1] - tin[1]]
        for arm, col in ((0, 12), (1, 17)):
            out[:, col:col + 3] = np.array([tin[0], tin[1], tz]) - tcp[:, arm]
            out[:, col + 3:col + 5] = np.array([pad[0], pad[1]]) - tcp[:, arm, :2]
        return out
