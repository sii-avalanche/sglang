"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDerivedWeightCache(unittest.TestCase):
    def test_nested_model_cache_rejects_updates(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, device="cpu"),
            torch.nn.Sequential(torch.nn.Module()),
        )
        model[1][0]._derived_weight_cache_error = "derived scales require restart"
        original = model[0].weight.detach().clone()
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [("0.weight", torch.zeros_like(original))], load_format="direct"
                ),
                (False, "derived scales require restart"),
            )
        self.assertTrue(torch.equal(model[0].weight, original))

    def test_model_without_derived_cache_keeps_updates_enabled(self):
        model = torch.nn.Module()
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))


class TestOrdinaryDPUpdateRanks(CustomTestCase):
    def _ranks(self, *, tp_size, dp_size, dp_attention, offsets):
        ranks = []
        module = "sglang.srt.model_executor.model_runner_components.weight_updater"
        parallel = SimpleNamespace(
            tp_size=tp_size, dp_size=dp_size, pp_size=1,
            enable_dp_attention=dp_attention,
        )
        with (
            patch(f"{module}.get_parallel", return_value=parallel),
            patch("torch.distributed.is_initialized", return_value=True),
            patch(f"{module}.init_custom_process_group") as init,
        ):
            for offset in offsets:
                for dp_rank in range(1 if dp_attention else dp_size):
                    for tp_rank in range(tp_size):
                        runner = SimpleNamespace(dp_rank=dp_rank)
                        updater = WeightUpdater(
                            tp_rank=tp_rank, device="cpu", gpu_id=0, model_config=None,
                            custom_weight_loaders={}, get_model=lambda: None,
                            update_model_fields=lambda **kw: None,
                            recapture_cuda_graph=lambda: None,
                            get_model_runner=lambda: runner,
                        )
                        ok, _ = updater.init_weights_update_group(
                            master_address="127.0.0.1", master_port=12345,
                            rank_offset=offset, world_size=17, group_name="test",
                        )
                        self.assertTrue(ok)
                        ranks.append(init.call_args.kwargs["rank"])
        return ranks

    def test_two_dp4_servers_have_eight_distinct_receivers(self):
        """Ordinary DP must not give every replica the same TP-local update rank."""
        self.assertEqual(
            self._ranks(tp_size=1, dp_size=4, dp_attention=False, offsets=[1, 5]),
            list(range(1, 9)),
        )

    def test_replica_offsets_include_the_full_tp_group(self):
        self.assertEqual(
            self._ranks(tp_size=2, dp_size=4, dp_attention=False, offsets=[1, 9]),
            list(range(1, 17)),
        )

    def test_dp_attention_does_not_add_a_second_replica_dimension(self):
        self.assertEqual(
            self._ranks(tp_size=4, dp_size=4, dp_attention=True, offsets=[1, 5]),
            list(range(1, 9)),
        )


if __name__ == "__main__":
    unittest.main()
