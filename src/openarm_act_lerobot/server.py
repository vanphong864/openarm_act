"""The policy over websocket, for openarm_act_lerobot_client.

openpi's protocol: on connecting, the server sends its metadata; then each
message from the client is an observation and each reply the actions, all
msgpack with NumPy arrays (openarm_act_lerobot_client.msgpack_numpy). A failed
inference sends the traceback as text and closes the connection. GET
/healthz answers 200 for load balancers and scripts.

Inference runs in a worker thread, one observation at a time: the GPU is
shared, and the delta-action step keeps the state of the observation in flight.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback

import websockets
import websockets.asyncio.server as ws_server
from openarm_act_lerobot_client import msgpack_numpy

log = logging.getLogger(__name__)


class PolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata: dict | None = None):
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.metadata = dict(metadata or {})
        self._lock: asyncio.Lock | None = None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        self._lock = asyncio.Lock()
        async with ws_server.serve(
            self._handler, self.host, self.port, compression=None, max_size=None, process_request=_health_check
        ) as server:
            log.info("listening on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        log.info("connection from %s", websocket.remote_address)
        await websocket.send(msgpack_numpy.packb(self.metadata))
        previous_total = None
        while True:
            try:
                started = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                async with self._lock:
                    infer_started = time.monotonic()
                    result = await asyncio.to_thread(self.policy.infer, obs)
                    infer_s = time.monotonic() - infer_started
                result["server_timing"] = {"infer_ms": infer_s * 1000}
                if previous_total is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total * 1000
                await websocket.send(msgpack_numpy.packb(result))
                previous_total = time.monotonic() - started
            except websockets.ConnectionClosed:
                log.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                log.exception("inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                break


def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
