"""Policy viết tay (scripted) gắp-đặt một vật: nội suy Cartesian min-jerk qua các điểm mốc + IK mỗi bước.

Dùng để sinh demo giả khi chưa có dữ liệu thật. Các biến thể lỗi cố ý ("variant") tạo ra demo hỏng
để thử bộ lọc demo tốt/xấu.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import GRIP_CLOSED, GRIP_OPEN, Arm, grasp_rot

NOMINAL_Q = np.array([0, 0, 0, 1.0, 0, 0, 0])
# Cấu hình khớp tay phải ở lúc kẹp / nhả vật, trung vị 37 episode thật (2026-10-08). Dùng làm hạt giống IK: người điều
# khiển thật với tới ô bằng cánh tay gần duỗi (khuỷu j4 ≈ 0.4 rad) và cẳng tay xoay j3 ≈ −1.2 rad, một cấu hình mà
# IK xuất phát từ NOMINAL_Q không tìm ra.
REAL_GRASP_Q = np.array([0.43, 0.10, -0.08, 0.42, 0.35, 0.22, -0.23])
REAL_RELEASE_Q = np.array([0.51, -0.10, -1.20, 0.41, 1.18, -0.50, -0.01])
MIRROR_LEFT = np.array([-1, -1, -1, 1, -1, -1, -1])
TIP_BELOW_TCP = 0.016      # đầu ngón kẹp thấp hơn TCP ~15.4 mm (đo từ mesh va chạm của ngón)
GRASP_AXIAL = 0.024      # m, TCP lệch khỏi tâm vật dài dọc trục dài (đo từ dữ liệu thật)…
GRASP_AXIAL_MAX_FRAC = 0.8   # …nhưng không quá 80% nửa chiều dài vật (ngón còn ôm được vật)
TIP_CLEARANCE = 0.003      # chừa khe giữa đầu ngón và mặt hộp khi gắp
RELEASE_DROP = 0.02        # thả vật cao hơn độ cao kẹp (m), đo từ dữ liệu thật
VARIANTS = ("good", "grasp_offset", "early_release", "miss_target", "jerky")


@dataclass
class Waypoint:
    pos: np.ndarray
    grip: float
    yaw: float               # góc xoay kẹp quanh trục z thế giới (rad)
    tilt: np.ndarray         # (roll, pitch) nghiêng kẹp (rad), xem arm.grasp_rot
    duration: float          # giây để đi từ mốc trước tới mốc này


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


DEG = np.pi / 180


def grasp_tilts(sign: float = 1.0, max_pitch: float = 45.0) -> list[tuple[float, float]]:
    """Các góc nghiêng (roll, pitch) thử theo thứ tự tăng dần độ nghiêng: thẳng đứng, nghiêng tới trước, rồi kèm nghiêng
    ngang (sign > 0 nghiêng về +y, tay phải nghiêng vào giữa; sign < 0 cho tay trái)."""
    out = [(0.0, 0.0)] + [(0.0, p * DEG) for p in range(10, int(max_pitch) + 1, 10)]
    out += [(sign * r * DEG, p * DEG) for p in (10, 20, 30) for r in (15, 25)]
    return out


def choose_orientation(arm: Arm, d, pos, low, yaws, rolls=(0.0,), seeds=()) -> tuple[float, tuple, np.ndarray]:
    """Chọn (yaw, tilt) để IK tới được cả `pos` và điểm thấp `low`, ưu tiên nghiêng ít (kẹp gần thẳng đứng)
    (`rolls` là danh sách số = roll, hoặc cặp (roll, pitch))
    rồi tới khớp xa giới hạn nhất. Trả (yaw, (roll, pitch), q tại pos)."""
    best = None
    span = arm.range[:, 1] - arm.range[:, 0]

    def margin_of(*qs):   # lề nhỏ nhất của MỌI khớp (theo % tầm quay), không chỉ khớp 7
        return min(float((np.minimum(q - arm.range[:, 0], arm.range[:, 1] - q) / span).min()) for q in qs)

    for ri, tilt in enumerate(rolls):
        roll, pitch = (tilt, 0.0) if np.isscalar(tilt) else tilt
        for yaw in yaws:
            R = grasp_rot(yaw, roll, pitch)
            # (q, sai số vị trí, sai số hướng) của seed tốt nhất — sai số hướng phải lấy ngay sau từng lần gọi ik,
            # vì arm.last_rot_err chỉ giữ kết quả của lần gọi gần nhất
            q, err, rot_err = min(((*arm.ik(d, pos, R, q_init=sd), arm.last_rot_err) for sd in seeds),
                                  key=lambda r: r[1])
            q_low, err_low = arm.ik(d, low, R, q_init=q)
            err = max(err, err_low)
            if max(rot_err, arm.last_rot_err) > 0.05:   # hướng kẹp lệch (bị chặn bởi lề giới hạn khớp)
                err = max(err, 1.0)
            margin = margin_of(q, q_low)
            score = (err > 2e-3, ri if err <= 2e-3 else 0, -margin if err <= 2e-3 else err)
            if best is None or score < best[0]:
                best = (score, yaw, (roll, pitch), q)
        if best[0][0] is False:
            break
    return best[1], best[2], best[3]


class ScriptedPickPlace:
    # (tay, điểm đích làm tròn mm) → điểm đặt khả thi đã tìm được; mọi episode của cùng bố trí dùng chung kết quả,
    # tránh tìm lại vài chục giây mỗi demo.
    _place_cache: dict = {}

    def __init__(self, arm: Arm, d, obj_pos, obj_yaw: float, target_pos, fps: int,
                 variant: str = "good", rng: np.random.Generator | None = None, max_joint_vel: float = 2.5,
                 obj_half=(0.02, 0.02, 0.02), target_half=None, obj_body: int | None = None):
        """obj_body: id body của vật. Có thì sau khi nhấc vật lên, policy đo vật thực sự nằm đâu trong tay và chỉnh điểm
        đặt cho tâm vật rơi đúng chỗ đã chọn (như người điều khiển nhìn rồi chỉnh); không có thì chạy hở hoàn toàn."""
        rng = rng or np.random.default_rng()
        self.arm, self.fps, self.variant, self.rng = arm, fps, variant, rng
        hx, hy, hz = obj_half
        noise = lambda s: rng.normal(0, s, 3) * [1, 1, 0]  # noqa: E731
        grasp = np.asarray(obj_pos, float) + noise(0.003)
        place = np.asarray(target_pos, float) + noise(0.008)
        # Người điều khiển thật không kẹp đúng tâm vật dài: TCP lệch ~2.4 cm dọc trục dài về phía đầu gần robot
        # (tâm vật nằm xa robot hơn TCP; dữ liệu thật: tâm vật − TCP ≈ 0.024 m dọc trục).
        # Nhờ vậy đặt tâm vật vào ô mà TCP không phải vươn quá xa.
        self.grasp_axial = np.zeros(2)
        self.grasp_long_side = False
        if max(hx, hy) > 1.3 * min(hx, hy):
            ux, uy = (np.cos(obj_yaw), np.sin(obj_yaw)) if hx > hy else (-np.sin(obj_yaw), np.cos(obj_yaw))
            toward_robot = -np.asarray(obj_pos[:2], float)
            sgn = 1.0 if ux * toward_robot[0] + uy * toward_robot[1] > 0 else -1.0
            self.grasp_axial = sgn * min(GRASP_AXIAL, GRASP_AXIAL_MAX_FRAC * max(hx, hy)) * np.array([ux, uy])
            grasp[:2] += self.grasp_axial
        if variant == "grasp_offset":
            ang = rng.uniform(0, 2 * np.pi)
            # lệch 1.8–2.5 lần nửa cạnh lớn của vật (tức kẹp trượt ra mép/ngoài vật)
            grasp[:2] += rng.uniform(1.8, 2.5) * max(hx, hy) * np.array([np.cos(ang), np.sin(ang)])
        if variant == "miss_target":   # đặt lệch 8–11 cm về phía tay (vẫn trong vùng với tới)
            place[:2] += [rng.uniform(-0.03, 0.03), -rng.uniform(0.08, 0.11)]
        speed = rng.uniform(0.85, 1.2)
        # Vật mỏng (nửa cao < ~19 mm): TCP phải cao hơn tâm vật để đầu ngón không đâm xuống hộp.
        z_grasp = max(0.0, TIP_BELOW_TCP + TIP_CLEARANCE - hz)
        # Thả vật cao hơn chỗ kẹp ~2 cm như người điều khiển thật (TCP lúc nhả ≈ 0.21, lúc kẹp ≈ 0.19): vật rơi nhẹ xuống ô,
        # và ở độ cao đó tay phải với xa hơn (ô gần sát giới hạn tầm với).
        up, z_place = 0.10, z_grasp + RELEASE_DROP
        # Vật dài (vd hộp thiếc 60×40): kẹp ngang cạnh ngắn, tức trục đóng của kẹp (trục y của TCP) song song cạnh ngắn.
        if max(hx, hy) > 1.3 * min(hx, hy):
            offs = (0.0, np.pi) if hx > hy else (np.pi / 2, -np.pi / 2)
        else:
            offs = tuple(k * np.pi / 2 for k in range(4))
        # Hướng kẹp khi gắp theo cạnh vật; khi đặt chọn riêng (vật đang nằm trong tay nên xoay cổ tay tuỳ ý),
        # vì cùng một hướng có thể không với tới được cả hai chỗ.
        q0 = arm.qpos(d)[:7]
        sign = 1.0 if arm.side == "right" else -1.0
        # Tay trái: cấu hình đối xứng gương của tay phải (dấu khớp 1,2,3,5,6,7 đảo; kiểm bằng FK, TCP lệch gương < 1 cm)
        real_seeds = (REAL_RELEASE_Q, REAL_GRASP_Q) if arm.side == "right" else \
            (MIRROR_LEFT * REAL_RELEASE_Q, MIRROR_LEFT * REAL_GRASP_Q)
        gy, gt, self._q_seed = choose_orientation(
            arm, d, grasp + [0, 0, up], grasp + [0, 0, z_grasp],
            yaws=[wrap(obj_yaw + o) for o in offs], rolls=grasp_tilts(sign, 40.0), seeds=(q0, NOMINAL_Q, *real_seeds))
        gt = np.asarray(gt, float)
        # Khi đặt: thử nghiêng dần mũi kẹp về phía trước và vào giữa (tay phải → +y, tay trái → −y).
        # Nếu tâm ô ngoài tầm với (thực tế: tay phải không với tới y ≥ 0 ở x ≈ 0.33), đặt ở điểm gần tâm nhất mà
        # với tới được và tâm vật còn trong ô (lề 3 mm), đúng như người điều khiển thật làm.
        offsets = [(0.0, 0.0)]
        if target_half is not None:
            inset = 0.003     # thành công tính theo TÂM vật trong ô (như dữ liệu thật: vật có thể thò ra mép giấy)
            mx, my = max(0.0, target_half[0] - inset), max(0.0, target_half[1] - inset)
            offsets = sorted(((dx, dy) for dx in np.arange(-mx, mx + 1e-9, 0.01) for dy in np.arange(-my, my + 1e-9, 0.01)),
                             key=lambda o: o[0] ** 2 + o[1] ** 2)
        self.place_offset = (0.0, 0.0)
        g = self.grasp_axial
        key = (arm.side, tuple(np.round(np.asarray(target_pos, float), 3)), tuple(np.round(obj_half, 3)),
               tuple(np.round(g, 3)))
        # ứng viên: (độ lệch trong ô, hệ số xoay cổ tay k). Xoay cổ tay Δ = k·30° xoay luôn vật quanh TCP, nên tâm vật
        # = TCP + Rz(Δ)(−g): TCP đích phải = điểm đặt + Rz(Δ) g.
        cands = [(off, k) for off in offsets for k in sorted(range(-3, 4), key=abs)]
        if key in self._place_cache and self._place_cache[key] in cands:
            cands = [self._place_cache[key]] + [c for c in cands if c != self._place_cache[key]]
        found = None
        for off, k in cands:
            dyaw = k * np.pi / 6
            c, sn = np.cos(dyaw), np.sin(dyaw)
            pl = place + [off[0] + c * g[0] - sn * g[1], off[1] + sn * g[0] + c * g[1], 0.0]
            # Loại nhanh: chỉ vị trí cũng không tới được thì khỏi thử hướng kẹp (rẻ hơn ~100 lần).
            if min(arm.ik(d, pl + [0, 0, z_place], None, q_init=sd, iters=120)[1] for sd in (self._q_seed, q0, *real_seeds)) > 2e-3:
                continue
            py, pt, qp = choose_orientation(
                arm, d, pl + [0, 0, up], pl + [0, 0, z_place],
                yaws=[wrap(gy + dyaw)], rolls=grasp_tilts(sign, 45.0), seeds=(self._q_seed, q0, NOMINAL_Q, *real_seeds))
            q_hi, e_hi = arm.ik(d, pl + [0, 0, up], grasp_rot(py, *pt), q_init=qp)
            q_lo, e_lo = arm.ik(d, pl + [0, 0, z_place], grasp_rot(py, *pt), q_init=q_hi)
            if max(e_hi, e_lo) < 2e-3:
                obj_goal = (place + [off[0], off[1], 0.0])[:2]
                place, self.place_offset, found = pl, (off[0], off[1]), (off, k)
                self._place_cache[key] = (off, k)
                break
        if found is None:       # không có điểm đặt khả thi: giữ tâm ô (demo sẽ không thành công)
            obj_goal = place[:2].copy()
            py, pt, qp = choose_orientation(arm, d, place + [0, 0, up], place + [0, 0, z_place], yaws=[gy],
                                            rolls=grasp_tilts(sign, 45.0), seeds=(self._q_seed, q0, NOMINAL_Q, *real_seeds))
        pt = np.asarray(pt, float)
        py = gy + wrap(py - gy)
        tcp0, _ = arm.tcp_pose(d)
        wp = [
            Waypoint(tcp0, GRIP_OPEN, gy, np.zeros(2), 0.0),
            Waypoint(grasp + [0, 0, up], GRIP_OPEN, gy, gt, 1.6),
            Waypoint(grasp + [0, 0, z_grasp], GRIP_OPEN, gy, gt, 1.0),
            Waypoint(grasp + [0, 0, z_grasp], GRIP_CLOSED, gy, gt, 0.6),
            Waypoint(grasp + [0, 0, up], GRIP_CLOSED, gy, gt, 1.0),
            Waypoint(place + [0, 0, up], GRIP_CLOSED, py, pt, 1.4),
            Waypoint(place + [0, 0, z_place], GRIP_CLOSED, py, pt, 1.0),
            Waypoint(place + [0, 0, z_place], GRIP_OPEN, py, pt, 0.5),
            Waypoint(place + [0, 0, up], GRIP_OPEN, py, pt, 1.0),
            Waypoint(place + [0, 0, up], GRIP_OPEN, py, pt, 0.5),
        ]
        for w in wp:
            w.duration /= speed
        self.release_step = None
        if variant == "early_release":   # nhả vật giữa đường mang tới ô đích
            t_transport = sum(w.duration for w in wp[:5]) + rng.uniform(0.05, 0.45) * wp[5].duration
            self.release_step = int(t_transport * fps)
        self.wp = wp
        self.times = np.cumsum([w.duration for w in wp])
        self.n_steps = int(self.times[-1] * fps) + 1
        self.q0 = q0
        self._q = q0
        self._t = 0
        self.obj_body, self._obj_goal, self._replanned = obj_body, obj_goal, False
        self.max_step = max_joint_vel / fps

    def target(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        """(vị trí TCP, độ mở kẹp, ma trận hướng TCP) mong muốn tại thời điểm t."""
        i = int(np.searchsorted(self.times, t, side="right"))
        if i >= len(self.wp):
            w = self.wp[-1]
            return w.pos, w.grip, grasp_rot(w.yaw, *w.tilt)
        a, b = self.wp[i - 1], self.wp[i]
        s = min_jerk((t - self.times[i - 1]) / b.duration)
        lerp = lambda u, v: u + s * (v - u)  # noqa: E731
        return lerp(a.pos, b.pos), lerp(a.grip, b.grip), grasp_rot(lerp(a.yaw, b.yaw), *lerp(a.tilt, b.tilt))

    def __call__(self, d) -> np.ndarray:
        """Lệnh 8 chiều cho bước hiện tại (gọi đúng một lần mỗi bước điều khiển)."""
        t = self._t / self.fps
        if self.obj_body is not None and not self._replanned and t >= self.times[4]:
            self._replan_place(d)
        pos, grip, rot = self.target(t)
        if t < self.times[1]:
            # Đoạn đầu đi trong không gian khớp tới tư thế trên vật (tránh điểm kỳ dị tay duỗi thẳng ở q = 0).
            q = self.q0 + min_jerk(t / self.times[1]) * (self._q_seed - self.q0)
        else:
            q, _ = self.arm.ik(d, pos, rot, q_init=self._q, iters=40)
        q = self._q + np.clip(q - self._q, -self.max_step, self.max_step)   # giới hạn tốc độ khớp
        self._q = q
        if self.release_step is not None and self._t >= self.release_step:
            grip = GRIP_OPEN
        # Servo mềm không bù trọng lực (như robot thật): bù võng tĩnh để tay đứng đúng tại q, như người điều khiển bù bằng mắt.
        act = np.append(self.arm.sag_compensated(d, q), grip)
        if self.variant == "jerky":
            act[:7] += self.rng.normal(0, 0.04, 7)
        self._t += 1
        return act

    def _replan_place(self, d) -> None:
        """Vật đã nhấc lên: đo vị trí vật trong khung TCP, dời các điểm đặt sao cho tâm vật rơi đúng `_obj_goal`."""
        self._replanned = True
        tcp, R = self.arm.tcp_pose(d)
        o_local = R.T @ (d.xpos[self.obj_body] - tcp)
        w = self.wp[6]                                     # điểm hạ vật xuống ô
        pred = w.pos + grasp_rot(w.yaw, *w.tilt) @ o_local
        delta = self._obj_goal - pred[:2]
        if np.linalg.norm(delta) > 0.05:                   # lệch quá xa = không thực sự cầm vật: giữ kế hoạch
            return
        for w in self.wp[5:]:
            w.pos = w.pos.copy()
            w.pos[:2] += delta

    @property
    def done(self) -> bool:
        return self._t >= self.n_steps
