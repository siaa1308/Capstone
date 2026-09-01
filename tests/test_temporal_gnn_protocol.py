from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "gnn"))

from causal_temporal_graphsage import epoch_sample_mask, masked_loss  # noqa: E402
from federated_causal_temporal_graphsage import aggregation_weights, fedavg  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
