import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_dashboard import cached_explanation_matches


class DashboardCacheTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "cache_key": "current-key",
            "verification": {"passed": True},
            "model": {"checkpoint_sha256": "current-checkpoint"},
            "prediction": {"probability": 0.42},
            "feature_explanation": {"target_probability": 0.4200001},
        }

    def test_matching_verified_explanation_is_reused(self):
        self.assertTrue(cached_explanation_matches(
            self.report, "current-key", "current-checkpoint"))

    def test_legacy_cache_without_target_score_is_regenerated(self):
        self.report["feature_explanation"].pop("target_probability")
        self.assertFalse(cached_explanation_matches(
            self.report, "current-key", "current-checkpoint"))

    def test_cache_with_a_different_shap_target_is_regenerated(self):
        self.report["feature_explanation"]["target_probability"] = 0.7
        self.assertFalse(cached_explanation_matches(
            self.report, "current-key", "current-checkpoint"))

    def test_wrong_cache_key_or_checkpoint_is_not_reused(self):
        self.assertFalse(cached_explanation_matches(
            self.report, "old-key", "current-checkpoint"))
        self.assertFalse(cached_explanation_matches(
            self.report, "current-key", "old-checkpoint"))


if __name__ == "__main__":
    unittest.main()
