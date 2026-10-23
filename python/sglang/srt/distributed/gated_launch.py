import json
import logging
import threading
import time
import urllib.request
from typing import Optional

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1.0
LOG_INTERVAL_SECONDS = 10.0

_instance: Optional["_GatedLaunchServer"] = None


def maybe_wait_for_gated_launch(*, host: str, port: Optional[int]) -> None:
    global _instance

    if port is None or _instance is not None:
        return

    world_group = get_world_group()
    parallel = get_parallel()
    ordinary_dp_follower = (
        parallel.dp_size > 1
        and not parallel.enable_dp_attention
        and parallel.dp_rank != 0
    )

    _instance = _GatedLaunchServer()
    status_url = None
    if world_group.rank_in_group == 0:
        if ordinary_dp_follower:
            # Ordinary DP groups have independent WORLDs but share one gate port.
            connect_host = {"0.0.0.0": "127.0.0.1", "::": "::1", "[::]": "::1"}.get(
                host, host
            )
            status_url = NetworkAddress(connect_host, port).to_url() + "/gate/status"
        else:
            _instance.serve(host=host, port=port)

    logger.info(f"Gated launch waiting for activation. rank={world_group.rank}")
    tic = time.perf_counter()
    _wait_until_activated(
        world_group=world_group, server=_instance, status_url=status_url
    )
    logger.info(f"Gated launch activated. elapsed={time.perf_counter() - tic:.2f} s")


def _wait_until_activated(
    *, world_group, server: "_GatedLaunchServer", status_url: Optional[str] = None
) -> None:
    activated = torch.zeros(1, dtype=torch.int32)
    started_at = time.perf_counter()
    next_log_at = started_at + LOG_INTERVAL_SECONDS

    while True:
        activated[0] = int(
            _poll_gate_status(status_url) if status_url is not None else server.activated
        )

        if world_group.world_size > 1:
            dist.broadcast(
                activated,
                src=world_group.ranks[0],
                group=world_group.cpu_group,
            )

        if bool(activated[0]):
            return

        if (now := time.perf_counter()) >= next_log_at:
            logger.info(
                f"Gated launch still waiting for activation. "
                f"rank={world_group.rank} elapsed={now - started_at:.0f} s"
            )
            next_log_at = now + LOG_INTERVAL_SECONDS

        time.sleep(POLL_INTERVAL_SECONDS)


def _poll_gate_status(url: str) -> bool:
    # Gate traffic must not go through an environment-configured HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=POLL_INTERVAL_SECONDS) as response:
            return json.load(response)["activated"] is True
    except (OSError, ValueError):
        return False


class _GatedLaunchServer:
    def __init__(self):
        self.activated = False
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def serve(self, *, host: str, port: int) -> None:
        config = uvicorn.Config(
            _build_app(self), host=host, port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info(f"Gated launch control server started on {host}:{port}")


def _build_app(server: _GatedLaunchServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return PlainTextResponse("OK")

    @app.get("/gate/status")
    def status():
        return {"activated": server.activated}

    @app.post("/gate/activate")
    def activate():
        server.activated = True
        return PlainTextResponse("OK")

    return app
