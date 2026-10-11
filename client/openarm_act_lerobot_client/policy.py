"""An ACT policy for openarm_act, served by openarm_act_lerobot over websocket.

openarm_act loads a policy by its import path, so this runs on the robot
without touching openarm_act's code:

    ros2 launch openarm_act act.launch.py \\
        policy:=openarm_act_lerobot_client.policy:RemoteActPolicy \\
        checkpoint:=ws://gpu-server:8000

It keeps ACT's contract (openarm_act/policy.py) exactly:

    predict(rgb uint8 [H, W, 3], depth uint16 [H, W] mm, qpos float32 [16])
        -> float32 [chunk, 16]  absolute targets, left j1..j7, left gripper,
                                right j1..j7, right gripper; rad, grippers in m;
                                actions[0] for the step the observation was taken at

and sends the server the raw frames: depth encoding, resizing, normalising
and delta-to-absolute all happen there, in the same code that trained it.

ACT reads no prompt. Predicted targets are clipped to OpenArm V1's joint limits
($OPENARM_ACT_CLIP=0 turns that off).

Python 3.10 (ROS 2 Humble), numpy 1, websockets and msgpack; no torch.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import time
from urllib.parse import urlparse

import numpy as np

from openarm_act_lerobot_client import msgpack_numpy

NUM_JOINTS = 16
JOINT_NAMES = (
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
)  # fmt: skip
# OpenArm V1 limits in JOINT_NAMES order; the same table as
# openarm_act_lerobot.constants.V1_JOINT_LIMITS (this package cannot import it).
V1_JOINT_LIMITS = np.asarray(
    [
        (-3.490659, 1.396263), (-3.316125, 0.174533), (-1.570796, 1.570796), (0.0, 2.443461),
        (-1.570796, 1.570796), (-0.785398, 0.785398), (-1.570796, 1.570796), (0.0, 0.043),
        (-1.396263, 3.490659), (-0.174533, 3.316125), (-1.570796, 1.570796), (0.0, 2.443461),
        (-1.570796, 1.570796), (-0.785398, 0.785398), (-1.570796, 1.570796), (0.0, 0.043),
    ],
    dtype=np.float32,
)  # fmt: skip
CAMERA_RGB = "chest"
CAMERA_DEPTH = "chest_depth"
DEFAULT_PORT = 8000

log = logging.getLogger(__name__)


def parse_server(address: str) -> tuple[str, int]:
    """"ws://host:port", "host:port" or "host" -> (host, port)."""
    address = address.strip()
    if not address:
        raise ValueError("no server: pass checkpoint:=ws://<host>:<port> (or set OPENARM_ACT_SERVER)")
    parsed = urlparse(address if "://" in address else f"ws://{address}")
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
        raise ValueError(f"cannot read a websocket server from {address!r}")
    return parsed.hostname, parsed.port or DEFAULT_PORT


class RemoteActPolicy:
    """openarm_act's Policy interface (predict / reset / describe), remote."""

    def __init__(self, checkpoint: str = "", connect_timeout: float = 10.0, **_ignored):
        self.host, self.port = parse_server(checkpoint or os.environ.get("OPENARM_ACT_SERVER", ""))
        self.connect_timeout = float(connect_timeout)
        self.ws = None
        self.metadata = self._connect()

        if self.metadata.get("action_dim") != NUM_JOINTS or tuple(self.metadata.get("joint_names", ())) != JOINT_NAMES:
            raise ValueError(
                f"the server at {self.host}:{self.port} is not an OpenArm policy "
                f"(action_dim {self.metadata.get('action_dim')}, joints {self.metadata.get('joint_names')})"
            )
        self.chunk_size = int(self.metadata["action_horizon"])
        # openarm_act refuses a model trained on other frames than the camera gives.
        size = self.metadata.get("image_size")
        self.image_size = tuple(int(v) for v in size) if size else None
        self.send_depth = CAMERA_DEPTH in (self.metadata.get("cameras") or ())
        self.clip = os.environ.get("OPENARM_ACT_CLIP", "1") != "0"
        self.server_ms = None
        self.round_trip_ms = None

    def _connect(self) -> dict:
        """Open the websocket; -> the server's metadata. Refuses quickly when
        nothing listens rather than waiting for ever."""
        try:
            socket.create_connection((self.host, self.port), timeout=self.connect_timeout).close()
        except OSError as error:
            raise ConnectionError(
                f"no policy server at {self.host}:{self.port} ({error}); start scripts/serve.py there"
            ) from error
        from websockets.sync.client import connect

        # As a context manager: websockets >= 16 warns otherwise, and every
        # version since 11 (ROS's pip may bring any) supports it.
        self._exit = contextlib.ExitStack()
        self.ws = self._exit.enter_context(
            connect(f"ws://{self.host}:{self.port}", compression=None, max_size=None, open_timeout=self.connect_timeout)
        )
        return dict(msgpack_numpy.unpackb(self.ws.recv()))

    def _infer(self, observation: dict) -> dict:
        self.ws.send(msgpack_numpy.packb(observation))
        response = self.ws.recv()
        if isinstance(response, str):  # the server sends its traceback as text
            raise RuntimeError(f"error in the policy server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def predict(self, rgb, depth, qpos):
        if self.ws is None:  # the last call lost the connection
            self._connect()
        observation = {
            "images": {CAMERA_RGB: np.ascontiguousarray(rgb, dtype=np.uint8)},
            "state": np.asarray(qpos, dtype=np.float32).reshape(NUM_JOINTS),
        }
        if self.send_depth:
            observation["images"][CAMERA_DEPTH] = np.ascontiguousarray(depth, dtype=np.uint16)
        started = time.perf_counter()
        try:
            result = self._infer(observation)
        except Exception:
            self.close()  # reconnect on the next call
            raise
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS:
            raise ValueError(f"server returned actions of shape {actions.shape}, expected [chunk, {NUM_JOINTS}]")
        timing = result.get("server_timing") or {}
        self.server_ms = timing.get("infer_ms")
        self.round_trip_ms = (time.perf_counter() - started) * 1e3
        if self.clip:
            actions = np.clip(actions, V1_JOINT_LIMITS[:, 0], V1_JOINT_LIMITS[:, 1])
        return actions[: self.chunk_size]

    def reset(self):
        """Nothing to drop: every chunk comes from one observation alone."""

    def close(self):
        if self.ws is not None:
            try:
                self._exit.close()
            except Exception:  # noqa: BLE001 -- already broken
                pass
            self.ws = None

    def describe(self):
        meta = self.metadata
        step = f" step {meta['step']}" if meta.get("step") is not None else ""
        return (
            f"RemoteActPolicy: {meta.get('model')}/{meta.get('backbone')}/{meta.get('finetune')}{step} "
            f"at ws://{self.host}:{self.port}, chunk {self.chunk_size} at {meta.get('fps')} Hz, "
            f"{'RGB-D' if self.send_depth else 'RGB'}"
            + (" (depth as 4th channel)" if meta.get("depth_as") == "channel" else "")
            + ("" if self.clip else ", V1 limits NOT enforced")
        )
