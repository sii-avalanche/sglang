"""Ordinary DP weight controls await every replica, including failed replies."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MODULE = "sglang.srt.managers.tokenizer_control_mixin"


class TestDistributedWeightControl(CustomTestCase):
    def _call(self, *, method, failed_rank=None, pp_size=1):
        async def run():
            manager = TokenizerControlMixin()
            manager.auto_create_handle_loop = Mock()
            manager.is_pause_cond = asyncio.Condition()
            manager.is_pause = True
            manager._weight_update_staged_session = False
            sent = []
            communicator = FanOutCommunicator(send=sent.append, fan_out=4)
            setattr(manager, f"{method}_communicator", communicator)
            request = SimpleNamespace(abort_all_requests=False, flush_cache=False, weight_version=None)
            parallel = SimpleNamespace(dp_size=4, pp_size=pp_size, enable_dp_attention=False)
            with patch(f"{_MODULE}.get_parallel", return_value=parallel):
                task = asyncio.create_task(getattr(manager, method)(request))
                await asyncio.sleep(0)
                if pp_size != 1:
                    with self.assertRaisesRegex(AssertionError, "pp_size=1"):
                        await task
                    self.assertEqual(sent, [])
                    return
                self.assertEqual(sent, [request])
                for rank in range(3):
                    communicator.handle_recv(SimpleNamespace(success=rank != failed_rank, message=f"rank{rank}"))
                await asyncio.sleep(0)
                self.assertFalse(task.done(), "three replies cannot acknowledge a DP4 publication")
                communicator.handle_recv(SimpleNamespace(success=3 != failed_rank, message="rank3"))
                success, message = await task
                self.assertEqual(success, failed_rank is None)
                self.assertIn("rank3", message)
        asyncio.run(run())

    def test_every_distributed_control_waits_for_all_four_replicas(self):
        """DP4 controls were rejected before they could reach the scheduler fan-out."""
        for method in ("init_weights_update_group", "update_weights_from_distributed", "destroy_weights_update_group"):
            with self.subTest(method=method):
                self._call(method=method)

    def test_a_failed_last_replica_fails_the_publication(self):
        self._call(method="update_weights_from_distributed", failed_rank=3)

    def test_ordinary_dp_with_pipeline_parallelism_is_rejected_before_dispatch(self):
        self._call(method="init_weights_update_group", pp_size=2)


if __name__ == "__main__":
    unittest.main()
