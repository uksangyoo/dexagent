"""Check pairing and analysis failure modes using synthetic data only."""

import copy
import json
import unittest

from study import PROTOCOL, manifest, paired_bootstrap, summarize, validate_results


class StudyTests(unittest.TestCase):
    def setUp(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol["profiles"]["test"] = {
            "methods": ["a", "b"], "tasks": ["task"], "splits": ["iid"],
            "train_seeds": [0, 1, 2], "cases_per_cell": 4,
        }
        self.planned = list(manifest(protocol, "test"))
        self.rows = [{**r, "status": "completed", "experiment_id": "synthetic_test_only",
                      "success": r["method"] == "a"} for r in self.planned]

    def test_case_seeds_are_paired_across_methods_and_training_seeds(self):
        by_case = {}
        for row in self.planned:
            by_case.setdefault(row["case_id"], set()).add(row["episode_seed"])
        self.assertTrue(all(len(seeds) == 1 for seeds in by_case.values()))

    def test_known_difference_and_interval(self):
        result = paired_bootstrap(self.rows, "a", "b", draws=100)
        self.assertEqual(result["success_difference"], 1.0)
        self.assertEqual(result["ci95"], [1.0, 1.0])
        self.assertEqual(validate_results(self.rows, self.planned)["missing"], 0)

    def test_pending_or_string_success_rejected(self):
        with self.assertRaises(ValueError):
            validate_results(self.planned, self.planned)
        rows = copy.deepcopy(self.rows)
        rows[0]["success"] = "false"
        with self.assertRaises(ValueError):
            validate_results(rows, self.planned)

    def test_duplicate_and_mixed_regime_rejected(self):
        with self.assertRaises(ValueError):
            validate_results(self.rows + [self.rows[0]], self.planned)
        rows = copy.deepcopy(self.rows)
        rows[0]["observation_regime"] = "vision"
        with self.assertRaises(ValueError):
            validate_results(rows)

    def test_missing_trial_cannot_silently_change_comparison(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.rows[1:], "a", "b", draws=100)
        with self.assertRaises(ValueError):
            summarize(self.rows[2:], self.planned, ["a", "b"], 100)

    def test_seed_mismatch_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["episode_seed"] += 1
        with self.assertRaises(ValueError):
            validate_results(rows, self.planned)


if __name__ == "__main__":
    unittest.main()
