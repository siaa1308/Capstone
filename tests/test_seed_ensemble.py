import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_seed_ensemble import mean_probabilities


class SeedEnsembleTests(unittest.TestCase):
    def test_equal_weights_and_member_order(self):
        predictions = [[0.1, 0.9], [0.4, 0.2], [0.7, 0.4]]
        np.testing.assert_allclose(mean_probabilities(predictions), [0.4, 0.5])
        np.testing.assert_allclose(mean_probabilities(predictions[::-1]), [0.4, 0.5])

    def test_identical_members_preserve_predictions(self):
        np.testing.assert_allclose(mean_probabilities([[0., 0.3, 1.]] * 3), [0., 0.3, 1.])

    def test_invalid_predictions_rejected(self):
        for invalid in ([], [[]], [[float("nan")]], [[1.1]], [[-0.1]], [0.2, 0.3]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mean_probabilities(invalid)


if __name__ == "__main__":
    unittest.main()
