from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

import multiprocessing
import os
import queue
import threading
import time
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch.distributed as dist

from sglang.srt.distributed import gated_launch
from sglang.srt.distributed.gated_launch import (
    POLL_INTERVAL_SECONDS,
    _GatedLaunchServer,
    _wait_until_activated,
    maybe_wait_for_gated_launch,
)
from sglang.srt.utils.network import get_open_port
from sglang.test.test_utils import CustomTestCase

_JOIN_TIMEOUT_SECONDS = 120


def _fake_world_group(*, rank: int, world_size: int, cpu_group=None):
    return SimpleNamespace(
        rank=rank,
        rank_in_group=rank,
        ranks=list(range(world_size)),
        world_size=world_size,
        cpu_group=cpu_group,
    )


def _activate_over_http(base_url: str, delay: float):
    time.sleep(delay)
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        try:
            if requests.post(f"{base_url}/gate/activate", timeout=1).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.1)


def _gated_launch_worker(
    rank: int, dist_port: int, gate_port: int, activate_after: float, out,
    enable_dp_attention: bool,
):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{dist_port}",
        rank=rank,
        world_size=2,
    )
    try:
        if rank == 0:
            threading.Thread(
                target=_activate_over_http,
                args=(f"http://127.0.0.1:{gate_port}", activate_after),
                daemon=True,
            ).start()

        world_group = _fake_world_group(
            rank=rank, world_size=2, cpu_group=dist.group.WORLD
        )
        started_at = time.perf_counter()
        with (
            patch.object(gated_launch, "get_world_group", return_value=world_group),
            patch.object(
                gated_launch,
                "get_parallel",
                return_value=SimpleNamespace(
                    dp_size=2 if enable_dp_attention else 1,
                    dp_rank=rank if enable_dp_attention else 0,
                    enable_dp_attention=enable_dp_attention,
                ),
            ),
        ):
            maybe_wait_for_gated_launch(host="127.0.0.1", port=gate_port)
        out.put((rank, time.perf_counter() - started_at))
    finally:
        dist.destroy_process_group()


def _ordinary_dp_worker(dp_rank, tp_rank, tp_size, dist_port, gate_port, out, release):
    if tp_size > 1:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{dist_port}",
            rank=tp_rank,
            world_size=tp_size,
        )
    try:
        world_group = _fake_world_group(
            rank=tp_rank,
            world_size=tp_size,
            cpu_group=dist.group.WORLD if tp_size > 1 else None,
        )
        parallel = SimpleNamespace(dp_size=4, dp_rank=dp_rank, enable_dp_attention=False)
        with (
            patch.object(gated_launch, "get_world_group", return_value=world_group),
            patch.object(gated_launch, "get_parallel", return_value=parallel, create=True),
        ):
            out.put(("waiting", dp_rank, tp_rank))
            maybe_wait_for_gated_launch(host="0.0.0.0", port=gate_port)
            out.put(("activated", dp_rank, tp_rank))
            release.wait(timeout=90)
    finally:
        if tp_size > 1:
            dist.destroy_process_group()


class TestGatedLaunchServer(CustomTestCase):
    def setUp(self):
        self.server = _GatedLaunchServer()
        self.port = get_open_port()
        self.server.serve(host="127.0.0.1", port=self.port)
        self.addCleanup(self._stop_server, self.server)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._wait_until_listening()

    @staticmethod
    def _stop_server(server):
        server._server.should_exit = True
        server._thread.join(timeout=5)

    def _wait_until_listening(self):
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline:
            try:
                requests.get(f"{self.base_url}/health", timeout=1)
                return
            except requests.exceptions.RequestException:
                time.sleep(0.1)
        self.fail(f"control server did not start listening on port {self.port}")

    def test_health_is_served_while_the_gate_is_still_closed(self):
        """The control port answers before activation so a caller can find it."""
        response = requests.get(f"{self.base_url}/health", timeout=5)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.server.activated)

    def test_activate_flips_the_flag_and_stays_successful_when_repeated(self):
        """A retried activation still succeeds instead of erroring or toggling back."""
        first = requests.post(f"{self.base_url}/gate/activate", timeout=5)
        self.assertEqual(first.status_code, 200)
        self.assertTrue(self.server.activated)

        second = requests.post(f"{self.base_url}/gate/activate", timeout=5)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(self.server.activated)

    def test_activating_one_server_leaves_another_one_closed(self):
        """The route acts on its own server instead of process wide state."""
        other = _GatedLaunchServer()
        other_port = get_open_port()
        other.serve(host="127.0.0.1", port=other_port)
        self.addCleanup(self._stop_server, other)

        requests.post(f"{self.base_url}/gate/activate", timeout=5)

        self.assertTrue(self.server.activated)
        self.assertFalse(other.activated)

    def test_status_does_not_activate_and_bypasses_environment_proxies(self):
        """Followers must wait for POST activation, not merely a healthy listener."""
        url = f"{self.base_url}/gate/status"
        with patch.dict(
            os.environ,
            {
                "http_proxy": "http://127.0.0.1:1",
                "HTTP_PROXY": "http://127.0.0.1:1",
                "no_proxy": "",
                "NO_PROXY": "",
            },
        ):
            self.assertFalse(gated_launch._poll_gate_status(url))
            self.assertFalse(self.server.activated)
        requests.post(f"{self.base_url}/gate/activate", timeout=5).raise_for_status()
        self.assertTrue(gated_launch._poll_gate_status(url))

    def test_unavailable_or_timed_out_status_keeps_the_gate_closed(self):
        """Followers starting before DP0 must retry instead of escaping or failing."""
        self.assertFalse(
            gated_launch._poll_gate_status(f"http://127.0.0.1:{get_open_port()}/gate/status")
        )
        with patch.object(
            urllib.request.OpenerDirector, "open", side_effect=TimeoutError
        ):
            self.assertFalse(gated_launch._poll_gate_status(f"{self.base_url}/gate/status"))


class TestOrdinaryDPGatedLaunch(CustomTestCase):
    def _check_independent_groups(self, tp_size):
        context = multiprocessing.get_context("spawn")
        out = context.Queue()
        release = context.Event()
        processes = []
        gate_port = get_open_port()
        dp_ranks = range(4) if tp_size == 1 else range(2)

        def start_group(dp_rank):
            dist_port = get_open_port()
            for tp_rank in range(tp_size):
                process = context.Process(
                    target=_ordinary_dp_worker,
                    args=(dp_rank, tp_rank, tp_size, dist_port, gate_port, out, release),
                )
                processes.append(process)
                process.start()

        try:
            # Followers arrive before the one owner; no worker may pass the gate.
            for dp_rank in list(dp_ranks)[1:]:
                start_group(dp_rank)
            for _ in processes:
                self.assertEqual(out.get(timeout=60)[0], "waiting")
            with self.assertRaises(queue.Empty):
                out.get(timeout=1.5)
            start_group(0)
            for _ in range(tp_size):
                self.assertEqual(out.get(timeout=60)[0], "waiting")

            base_url = f"http://127.0.0.1:{gate_port}"
            deadline = time.monotonic() + 30
            while True:
                try:
                    response = requests.get(f"{base_url}/health", timeout=1)
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            with self.assertRaises(queue.Empty):
                out.get(timeout=0.2)
            requests.post(f"{base_url}/gate/activate", timeout=5).raise_for_status()
            expected = {("activated", dp, tp) for dp in dp_ranks for tp in range(tp_size)}
            self.assertEqual({out.get(timeout=20) for _ in processes}, expected)

            # The activation remains visible to a follower arriving afterwards.
            start_group(1)
            self.assertEqual(
                {out.get(timeout=60) for _ in range(2 * tp_size)},
                {
                    (state, 1, tp)
                    for state in ("waiting", "activated")
                    for tp in range(tp_size)
                },
            )
        finally:
            release.set()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
            out.close()

    def test_one_activation_releases_all_four_independent_tp1_workers(self):
        """Ordinary DP must not leave three workers waiting on failed gate binds."""
        self._check_independent_groups(tp_size=1)

    def test_follower_leader_broadcasts_activation_to_its_own_tp_group(self):
        """A follower's nonzero TP ranks cannot reach DP0 through their WORLD."""
        self._check_independent_groups(tp_size=2)


class TestWaitUntilActivated(CustomTestCase):
    def test_single_rank_keeps_polling_until_the_flag_is_set(self):
        """A lone rank leaves the gate only after its own flag flips."""
        server = _GatedLaunchServer()
        activate_after = 2 * POLL_INTERVAL_SECONDS
        threading.Timer(
            activate_after, lambda: setattr(server, "activated", True)
        ).start()

        started_at = time.perf_counter()
        _wait_until_activated(
            world_group=_fake_world_group(rank=0, world_size=1), server=server
        )
        elapsed = time.perf_counter() - started_at

        self.assertGreaterEqual(elapsed, activate_after)

    def test_second_rank_learns_about_activation_through_the_cpu_group(self):
        """Driven through maybe_wait_for_gated_launch: the rank without the control server is released by the gloo broadcast."""
        self._check_cpu_group(enable_dp_attention=False)

    def test_dp_attention_still_uses_its_shared_world(self):
        """DP attention must retain the existing single-gate WORLD broadcast."""
        self._check_cpu_group(enable_dp_attention=True)

    def _check_cpu_group(self, *, enable_dp_attention):
        context = multiprocessing.get_context("spawn")
        out = context.Queue()
        dist_port = get_open_port()
        gate_port = get_open_port()
        activate_after = 2 * POLL_INTERVAL_SECONDS

        processes = [
            context.Process(
                target=_gated_launch_worker,
                args=(rank, dist_port, gate_port, activate_after, out, enable_dp_attention),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()

        elapsed_by_rank = {}
        for _ in processes:
            rank, elapsed = out.get(timeout=_JOIN_TIMEOUT_SECONDS)
            elapsed_by_rank[rank] = elapsed

        for process in processes:
            process.join(timeout=_JOIN_TIMEOUT_SECONDS)
        for process in processes:
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sorted(elapsed_by_rank), [0, 1])
        self.assertGreaterEqual(elapsed_by_rank[1], activate_after)


class TestMaybeWaitForGatedLaunch(CustomTestCase):
    def setUp(self):
        self.addCleanup(setattr, gated_launch, "_instance", None)
        gated_launch._instance = None

    def test_an_unset_port_leaves_the_startup_path_untouched(self):
        """Without the flag the gate never reaches the distributed environment."""
        with patch.object(gated_launch, "get_world_group") as get_world_group:
            maybe_wait_for_gated_launch(host="127.0.0.1", port=None)

        get_world_group.assert_not_called()
        self.assertIsNone(gated_launch._instance)

    def test_a_second_call_in_the_same_process_does_not_gate_again(self):
        """A draft worker re-entering the init path must not wait a second time."""
        gated_launch._instance = _GatedLaunchServer()

        with patch.object(gated_launch, "get_world_group") as get_world_group:
            maybe_wait_for_gated_launch(host="127.0.0.1", port=get_open_port())

        get_world_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
