"""Rule-based gripper for diagnosis: the policy moves the arms, this rule opens and closes the grippers.

Close when the fingertips sit around the tin (closing axis across its short side, tips below its top, anywhere
along the long side the fingers still hold) and the hand has almost stopped; stay closed
until the tin is lowered onto the pad, then open for good. A grasp that closes on air (opening below
MISSED_M) reopens so the policy can try again. Reads the true tin pose and TCP from the simulator, so it answers
"can the arm policy carry the tin when the gripper is right?", not how a deployed robot would decide.
"""

import numpy as np

from .arm import GRIP_CLOSED, GRIP_OPEN
from .policy import TIP_BELOW_TCP

COLUMNS = {"left": 7, "right": 15}
CLOSED_COMMAND = 0.03      # recorded commands below this are "close" (same threshold as the dataset)
# Where the demonstrator's fingertips were when it closed (recorded commands replayed in the sim, 11 val episodes):
# across 4-34 mm (median ~18), along up to 40 mm, height -8..+12 mm, closing axis within 18 deg of the short side.
ACROSS_TOL_M = 0.025       # fingertips' offset along the closing axis
ALONG_MARGIN_M = 0.002     # fingertips inside the tin's long half-length (44 mm)
HEIGHT_M = (-0.012, 0.013) # fingertips vs the tin centre (expert: 5 mm below; the tin top is 8 mm above)
ALIGN_DEG = 25.0           # closing axis vs the tin's short side
STILL_M_S = 0.05           # TCP speed below this
MISSED_M = 0.008           # opening below this while closed: nothing between the fingers
MISSED_S, RETRY_S = 0.3, 1.0
RELEASE_LIFT_M = 0.03      # the tin must have been lifted this much before a release counts
RELEASE_HEIGHT_M = 0.035   # release once the tin is this close above the pad surface, inside the pad
RELEASE_MARGIN_M = 0.005


class GripperRule:
    def __init__(self, sim, m, tin_body, tin_size, pad_geom, fps):
        self.sim, self.m, self.tin_body, self.pad, self.fps = sim, m, tin_body, pad_geom, fps
        self.half = np.asarray(tin_size, float) / 2
        self.long_axis = 0 if self.half[0] >= self.half[1] else 1
        self.events = []

    def reset(self, last_command):
        """Start of an episode or at the handover: a gripper commanded closed counts as holding."""
        self.state = {s: "closed" if last_command[c] < CLOSED_COMMAND else "open" for s, c in COLUMNS.items()}
        self.prev_tcp = {s: None for s in COLUMNS}
        self.missed = {s: 0 for s in COLUMNS}
        self.wait = {s: 0 for s in COLUMNS}
        self.tin_rest_z = float(self.sim.d.xpos[self.tin_body][2])
        self.lifted = False
        self.events = []

    def grasp_offset(self, side):
        """Fingertip centre in the tin frame: (across the closing axis, along the long side, height), and the
        closing-axis angle to the short side (deg). The tips are TIP_BELOW_TCP along the TCP z axis (R_DOWN)."""
        d = self.sim.d
        tcp, R = self.sim.arms[side].tcp_pose(d)
        tip = tcp + TIP_BELOW_TCP * R[:, 2]
        Rt = d.xmat[self.tin_body].reshape(3, 3)
        long_ax, short_ax = Rt[:, self.long_axis], Rt[:, 1 - self.long_axis]
        rel = tip - d.xpos[self.tin_body]
        closing = R[:, 1]
        cos = abs(closing[:2] @ short_ax[:2]) / max(1e-9, np.linalg.norm(closing[:2]) * np.linalg.norm(short_ax[:2]))
        return np.array([rel @ short_ax, rel @ long_ax, rel[2]]), float(np.degrees(np.arccos(min(1.0, cos)))), tcp

    def __call__(self, step, action):
        action = np.array(action, float)
        d = self.sim.d
        tin = d.xpos[self.tin_body]
        self.lifted |= bool(tin[2] - self.tin_rest_z > RELEASE_LIFT_M)
        q = self.sim.qpos16()
        for side, col in COLUMNS.items():
            off, angle, tcp = self.grasp_offset(side)
            speed = 0.0 if self.prev_tcp[side] is None else np.linalg.norm(tcp - self.prev_tcp[side]) * self.fps
            self.prev_tcp[side] = tcp
            if self.state[side] == "open":
                self.wait[side] = max(0, self.wait[side] - 1)
                ready = (abs(off[0]) < ACROSS_TOL_M and abs(off[1]) < self.half[self.long_axis] - ALONG_MARGIN_M
                         and HEIGHT_M[0] < off[2] < HEIGHT_M[1] and angle < ALIGN_DEG and speed < STILL_M_S)
                if ready and not self.wait[side]:
                    self.state[side], self.missed[side] = "closed", 0
                    self.events.append((step, side, "close"))
            elif self.state[side] == "closed":
                self.missed[side] = self.missed[side] + 1 if q[col] < MISSED_M else 0
                if self.missed[side] > MISSED_S * self.fps:
                    self.state[side], self.wait[side] = "open", round(RETRY_S * self.fps)
                    self.events.append((step, side, "missed"))
                elif self.lifted and self._over_pad(tin):
                    self.state[side] = "released"
                    self.events.append((step, side, "release"))
            action[col] = GRIP_CLOSED if self.state[side] == "closed" else GRIP_OPEN
        return action

    def _over_pad(self, tin):
        d = self.sim.d
        local = (tin - d.geom_xpos[self.pad]) @ d.geom_xmat[self.pad].reshape(3, 3)
        half = self.m.geom_size[self.pad]
        inside = abs(local[0]) < half[0] - RELEASE_MARGIN_M and abs(local[1]) < half[1] - RELEASE_MARGIN_M
        return bool(inside and local[2] - half[2] - self.half[2] < RELEASE_HEIGHT_M)
