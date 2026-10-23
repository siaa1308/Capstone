import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "gnn"))
from causal_tail_guard import select_guarded_step, tail_ranking_risk, training_task_risks
from causal_temporal_graphsage import CausalTemporalGraphSAGE, Events


class TailGuardTests(unittest.TestCase):
    def test_ranking_and_shift_invariance(self):
        labels = np.array([1, 0, 0])
        good = tail_ranking_risk([3, 1, -1], labels, 1)
        self.assertAlmostEqual(good, np.logaddexp(0, -2))
        self.assertAlmostEqual(good, tail_ranking_risk([103, 101, 99], labels, 1))
        self.assertLess(good, tail_ranking_risk([-3, 1, -1], labels, 1))

    def test_absent_class_is_not_fabricated(self):
        self.assertIsNone(tail_ranking_risk([1, 2], [0, 0]))
        self.assertIsNone(tail_ranking_risk([], []))

    def test_all_tasks_must_pass(self):
        old, proposal = {"w": torch.tensor(0.)}, {"w": torch.tensor(1.)}
        def risks(state):
            x = float(state["w"])
            return {"bank1/june": 1 - x / 2, "bank2/july": 1 + x / 10}
        result, diagnostic = select_guarded_step(old, proposal, risks, tolerance=0.051)
        self.assertEqual(diagnostic["accepted_step"], 0.5)
        self.assertEqual(float(result["w"]), 0.5)
        self.assertEqual(float(old["w"]), 0.)

    def test_zero_step_when_every_proposal_harms_a_task(self):
        old, proposed = {"w": torch.tensor(0.)}, {"w": torch.tensor(1.)}
        result, d = select_guarded_step(old, proposed, lambda s: {"task": 1 + float(s["w"])}, 0.)
        self.assertEqual(d["accepted_step"], 0.)
        self.assertTrue(torch.equal(result["w"], old["w"]))

    def test_invalid_settings_fail(self):
        with self.assertRaises(ValueError):
            tail_ranking_risk([float("nan")], [1])
        with self.assertRaises(ValueError):
            select_guarded_step({}, {}, lambda _: {}, steps=(0.5, 1.))
        with self.assertRaises(ValueError):
            select_guarded_step({}, {}, lambda _: {})

    def test_training_only_and_no_model_mutation(self):
        class TrainingOnly(dict):
            def __getitem__(self, key):
                if key != "training":
                    raise AssertionError("Holdout accessed")
                return super().__getitem__(key)
        torch.manual_seed(1)
        model = CausalTemporalGraphSAGE(2, 2, 4, 0.25)
        events = Events(torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 1, 0]),
                        torch.randn(4, 2), torch.tensor([1., 2., 3., 4.]), torch.tensor([1., 0., 1., 0.]))
        clients = {"bank": (torch.randn(2, 2), TrainingOnly(training=events), 3.)}
        before = {k: v.clone() for k, v in model.state_dict().items()}
        first = training_task_risks(model, clients, 2)
        second = training_task_risks(model, clients, 2)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"bank/june", "bank/july"})
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
