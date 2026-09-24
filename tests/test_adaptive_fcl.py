from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "gnn"))
import adaptive_continual_federated as research
from causal_temporal_graphsage import CausalTemporalGraphSAGE, Events


def events(count=12):
    return Events(torch.arange(count) % 4, (torch.arange(count) + 1) % 4,
                  torch.randn(count, 2), torch.arange(count).float(),
                  torch.tensor([float(i % 4 == 0) for i in range(count)]))


class ReplayTests(unittest.TestCase):
    def test_capacity_positives_determinism_and_order(self):
        history = events(40)
        scores = np.linspace(0, 1, 40)
        first = research.risk_context_indices(history, scores, 20, 42)
        second = research.risk_context_indices(history, scores, 20, 42)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(first), 20)
        self.assertEqual(len(first.unique()), 20)
        self.assertTrue(bool(torch.all(first[1:] > first[:-1])))
        self.assertTrue(set(torch.where(history.labels == 1)[0].tolist()) <= set(first.tolist()))

    def test_zero_and_positive_overflow_budgets(self):
        history = events()
        scores = np.ones(len(history))
        self.assertEqual(len(research.risk_context_indices(history, scores, 0, 1)), 0)
        chosen = research.risk_context_indices(history, scores, 2, 1)
        self.assertTrue(bool(torch.all(history.labels[chosen] == 1)))
        self.assertEqual(len(research.risk_context_indices(history, scores, 100, 1)), len(history))

    def test_context_is_earlier_and_shares_account(self):
        history = Events(torch.tensor([0, 8, 0, 0, 0]), torch.tensor([1, 9, 2, 3, 4]),
                         torch.zeros(5, 2), torch.tensor([1., 2., 3., 3., 4.]),
                         torch.tensor([0., 0., 1., 0., 0.]))
        chosen = research.risk_context_indices(history, np.zeros(5), 2, 1, 0, 1)
        self.assertEqual(chosen.tolist(), [0, 2])

    def test_rejects_unsorted_history_and_bad_scores(self):
        history = events()
        history.timestamp[0] = 100
        with self.assertRaises(ValueError):
            research.risk_context_indices(history, np.zeros(12), 5, 1)
        with self.assertRaises(ValueError):
            research.risk_context_indices(events(), np.full(12, np.nan), 5, 1)

    def test_calendar_boundary_is_not_relative_thirty_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "training" / "bank"
            folder.mkdir(parents=True)
            pd.DataFrame({"timestamp": ["2025-06-01 00:00:24.5", "2025-07-01 00:00:01"]}).to_csv(
                folder / "edge_list.csv.gz", index=False)
            boundary = research.calendar_boundary(root, "bank")
            self.assertEqual(boundary, 30 * 86400 - 24.5)
            self.assertGreater(30 * 86400 - 23.5, boundary)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.base = {"memory_update.weight": torch.zeros(2), "decoder.weight": torch.zeros(2)}

    def test_disabled_is_fedavg_and_classifier_always_fedavg(self):
        states = [{"memory_update.weight": torch.tensor([1., 2.]), "decoder.weight": torch.tensor([2., 4.])},
                  {"memory_update.weight": torch.tensor([-3., 1.]), "decoder.weight": torch.tensor([4., 2.])}]
        average, _ = research.temporal_harmonized_average(states, self.base, 0, float("inf"))
        for key in self.base:
            self.assertTrue(torch.allclose(average[key], (states[0][key] + states[1][key]) / 2))
        harmonized, _ = research.temporal_harmonized_average(states, self.base)
        self.assertTrue(torch.equal(harmonized["decoder.weight"], torch.tensor([3., 3.])))

    def test_conflicts_reported_and_client_order_invariant(self):
        states = [{"memory_update.weight": x, "decoder.weight": torch.ones(2)}
                  for x in [torch.tensor([1., 0.]), torch.tensor([-1., 1.]), torch.tensor([2., 1.])]]
        forward, diagnostic = research.temporal_harmonized_average(states, self.base)
        reverse, _ = research.temporal_harmonized_average(states[::-1], self.base)
        self.assertGreater(diagnostic["conflicting_pairs"], 0)
        for key in self.base:
            self.assertTrue(torch.allclose(forward[key], reverse[key]))

    def test_zero_updates_are_finite_and_unchanged(self):
        result, _ = research.temporal_harmonized_average([self.base, self.base, self.base], self.base)
        for key in result:
            self.assertTrue(torch.equal(result[key], self.base[key]))

    def test_single_client_is_unchanged(self):
        state = {key: torch.ones_like(value) for key, value in self.base.items()}
        result, _ = research.temporal_harmonized_average([state], self.base)
        for key in result:
            self.assertTrue(torch.equal(result[key], state[key]))


class TrainingTests(unittest.TestCase):
    def test_fast_threshold_exactly_matches_original(self):
        from causal_temporal_graphsage import choose_threshold
        rng = np.random.default_rng(72)
        for count, positive_count, rounded in [(1, 0, False), (17, 0, True), (30, 30, True),
                                               (127, 4, True), (1024, 6, False)]:
            scores = rng.random(count)
            if rounded:
                scores = np.round(scores, 1)
            labels = np.zeros(count)
            labels[:positive_count] = 1
            self.assertEqual(research.fast_choose_threshold(labels, scores), choose_threshold(labels, scores))

    def test_uniform_global_rng_matches_frozen_client_training(self):
        from federated_causal_temporal_graphsage import continual_local_train
        torch.manual_seed(19)
        history, static = events(), torch.randn(4, 3)
        history.timestamp = torch.tensor([float(i * 500_000) for i in range(12)])
        initial = CausalTemporalGraphSAGE(3, 2, 4, .25)
        baseline, candidate = research.copy.deepcopy(initial), research.copy.deepcopy(initial)
        cfg = SimpleNamespace(local_epochs=1, batch_size=3, replay_size=4, learning_rate=.001,
                              weight_decay=.0001, optimizer="adamw", adam_eps=1e-8,
                              negative_ratio=20, tbptt_steps=2, replay="uniform", distill=0., rng_policy="global")
        rng = torch.get_rng_state()
        continual_local_train(baseline, static, history, cfg, 42)
        torch.set_rng_state(rng)
        research.train_client(candidate, static, history, 30 * 86400, cfg, 42)
        for key in baseline.state_dict():
            self.assertTrue(torch.equal(baseline.state_dict()[key], candidate.state_dict()[key]), key)

    def test_development_loader_never_requests_testing(self):
        calls = []

        def guarded(dataset, bank, *encoders, splits):
            self.assertEqual(splits, ("training", "validation"))
            calls.append(bank)
            return torch.zeros(4, 3), {"training": events(), "validation": events()}, [], []

        with patch.object(research, "fit_shared_feature_encoders", return_value=(None, None)), \
             patch.object(research, "calendar_boundary", return_value=30 * 86400), \
             patch.object(research, "build_bank_data", side_effect=guarded):
            clients = research.load_development_clients(Path("unused"))
        self.assertEqual(tuple(calls), research.BANKS)
        self.assertTrue(all("testing" not in streams for _, streams, _ in clients.values()))

    def test_retention_trains_temporal_modules(self):
        torch.manual_seed(7)
        history, static = events(), torch.randn(4, 3)
        model = CausalTemporalGraphSAGE(3, 2, 4, 0.)
        before = model.memory_update.weight_hh.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        research.distillation_epoch(model, static, history, optimizer, torch.ones(12, dtype=torch.bool),
                                    3, 2, torch.zeros(12), torch.ones(12, dtype=torch.bool), .05)
        self.assertFalse(torch.equal(before, model.memory_update.weight_hh))

    def test_future_teacher_targets_do_not_affect_training(self):
        torch.manual_seed(9)
        history, static = events(), torch.randn(4, 3)
        initial = CausalTemporalGraphSAGE(3, 2, 4, 0.)
        models = [research.copy.deepcopy(initial), research.copy.deepcopy(initial)]
        retained = torch.arange(12) < 4
        for index, model in enumerate(models):
            targets = torch.zeros(12)
            targets[~retained] = index * 10000.
            research.distillation_epoch(model, static, history, torch.optim.AdamW(model.parameters(), lr=.001),
                                        torch.ones(12, dtype=torch.bool), 3, 2, targets, retained, .05)
        for key in initial.state_dict():
            self.assertTrue(torch.equal(models[0].state_dict()[key], models[1].state_dict()[key]))


class EvaluationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import evaluate_adaptive_candidate
        cls.evaluator = evaluate_adaptive_candidate

    def test_mutated_checkpoint_rejected_before_dataset_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"changed checkpoint")
            record = {"entries": [{"checkpoint": str(checkpoint),
                                   "checkpoint_sha256": hashlib.sha256(b"original checkpoint").hexdigest()}]}
            freeze_path = root / "freeze.json"
            freeze_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                self.evaluator.evaluate(freeze_path, root / "evaluation")
            self.assertFalse((root / "evaluation").exists())

    def test_existing_evaluation_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.evaluator.evaluate(root / "missing-freeze.json", output)

    def test_test_contaminated_selection_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "selection.json").write_text(json.dumps({"winner": "candidate"}))
            run = root / "federated" / "baseline" / "seed_42"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({"test_loaded": True, "per_bank": []}))
            with self.assertRaisesRegex(ValueError, "validation-only"):
                self.evaluator.freeze(root, "candidate", root / "freeze.json")
            self.assertFalse((root / "freeze.json").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
