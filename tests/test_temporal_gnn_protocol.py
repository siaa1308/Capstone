from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "gnn"))

from causal_temporal_graphsage import (  # noqa: E402
    CausalTemporalGraphSAGE, Events, epoch_sample_mask, masked_loss, train_temporal_epoch,
)
from federated_causal_temporal_graphsage import (  # noqa: E402
    aggregation_weights, fedavg, interpolated_average, local_train,
)


class SamplingProtocolTests(unittest.TestCase):
    def test_epoch_sample_is_global_and_keeps_all_positives(self) -> None:
        labels = torch.zeros(100, dtype=torch.float32)
        labels[[10, 90]] = 1
        generator = torch.Generator().manual_seed(42)
        mask = epoch_sample_mask(labels, negative_ratio=3, generator=generator)
        self.assertTrue(bool(mask[10]))
        self.assertTrue(bool(mask[90]))
        self.assertEqual(int(mask.sum()), 8)
        self.assertTrue(bool(mask[:50].any()))
        self.assertTrue(bool(mask[50:].any()))

    def test_masked_loss_skips_only_empty_sample_slice(self) -> None:
        logits = torch.tensor([0.0, 1.0])
        labels = torch.tensor([0.0, 1.0])
        self.assertIsNone(masked_loss(logits, labels, torch.tensor([False, False])))
        self.assertIsNotNone(masked_loss(logits, labels, torch.tensor([True, False])))


class FederatedProtocolTests(unittest.TestCase):
    def test_aggregation_weight_strategies(self) -> None:
        counts = [100, 25]
        self.assertEqual(aggregation_weights(counts, "samples"), [100.0, 25.0])
        self.assertEqual(aggregation_weights(counts, "sqrt_samples"), [10.0, 5.0])
        self.assertEqual(aggregation_weights(counts, "uniform"), [1.0, 1.0])

    def test_fedavg_uses_requested_weights(self) -> None:
        states = [{"weight": torch.tensor([0.0])}, {"weight": torch.tensor([10.0])}]
        result = fedavg(states, [3.0, 1.0])
        self.assertAlmostEqual(float(result["weight"].item()), 2.5)

    def test_server_interpolation_can_treat_temporal_parameters_separately(self) -> None:
        states = [
            {"memory_update.weight": torch.tensor([2.0]), "decoder.weight": torch.tensor([2.0])},
            {"memory_update.weight": torch.tensor([4.0]), "decoder.weight": torch.tensor([4.0])},
        ]
        global_state = {
            "memory_update.weight": torch.tensor([0.0]), "decoder.weight": torch.tensor([0.0]),
        }
        result = interpolated_average(states, [1.0, 1.0], global_state, 0.5, 0.25)
        self.assertAlmostEqual(float(result["memory_update.weight"].item()), 0.75)
        self.assertAlmostEqual(float(result["decoder.weight"].item()), 1.5)

    def test_two_batch_tbptt_trains_temporal_update_modules(self) -> None:
        torch.manual_seed(7)
        static = torch.randn(4, 3)
        events = Events(
            src=torch.tensor([0, 1, 2, 3]),
            dst=torch.tensor([1, 2, 3, 0]),
            edge_attr=torch.randn(4, 2),
            timestamp=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        model = CausalTemporalGraphSAGE(3, 2, 4, 0.0)
        global_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        cfg = SimpleNamespace(
            optimizer="adamw", learning_rate=1e-3, weight_decay=0.0, adam_eps=1e-8,
            momentum=0.0, local_epochs=1, negative_ratio=0, batch_size=2,
            tbptt_steps=2, algorithm="fedavg", prox_mu=0.0,
        )
        diagnostics = local_train(model, static, events, cfg, 42, global_parameters)
        self.assertGreater(diagnostics["temporal_gradient_steps"], 0)

    def test_local_two_batch_tbptt_updates_temporal_modules(self) -> None:
        torch.manual_seed(11)
        static = torch.randn(4, 3)
        events = Events(
            src=torch.tensor([0, 1, 2, 3]), dst=torch.tensor([1, 2, 3, 0]),
            edge_attr=torch.randn(4, 2), timestamp=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        model = CausalTemporalGraphSAGE(3, 2, 4, 0.0)
        before = model.memory_update.weight_hh.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_temporal_epoch(
            model, static, events, optimizer, torch.ones(4, dtype=torch.bool), 2, 2,
        )
        self.assertFalse(torch.equal(before, model.memory_update.weight_hh.detach()))


if __name__ == "__main__":
    unittest.main()
