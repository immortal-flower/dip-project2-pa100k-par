import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "starter_code"))
from optimize_ensemble import (assert_aligned, blend, calculate_metrics,
                               decode_probabilities)


SPEC = {
    "num_attrs": 3,
    "attr2idx": {"class_a": 0, "class_b": 1, "flag": 2},
    "groups": [
        {"name": "class", "type": "softmax", "attrs": ["class_a", "class_b"],
         "indices": [0, 1]},
        {"name": "flag", "type": "binary", "attrs": ["flag"], "indices": [2]},
    ],
}


class MetricCombinationTest(unittest.TestCase):
    def test_softmax_is_one_hot_and_binary_uses_threshold(self):
        probabilities = np.array([[0.2, 0.8, 0.5], [0.9, 0.1, 0.49]])
        pred = decode_probabilities(probabilities, SPEC)
        np.testing.assert_array_equal(pred, [[False, True, True], [True, False, False]])
        np.testing.assert_array_equal(pred[:, :2].sum(1), [1, 1])

    def test_alignment_mismatch_is_rejected(self):
        base = {"paths": np.array(["a"]), "attributes": np.array(["x"]),
                "targets": np.array([[1]]), "probabilities": np.array([[0.9]])}
        other = {key: value.copy() for key, value in base.items()}
        other["paths"][0] = "b"
        with self.assertRaisesRegex(ValueError, "paths differs"):
            assert_aligned({"base": base, "other": other})

    def test_frozen_blend_and_metrics_are_reproducible(self):
        first = np.array([[0.7, 0.3, 0.8], [0.4, 0.6, 0.2]])
        second = np.array([[0.9, 0.1, 0.4], [0.2, 0.8, 0.6]])
        weights = {"first": 0.25, "second": 0.75}
        combined = blend({"first": first, "second": second}, weights)
        np.testing.assert_allclose(combined, 0.25 * first + 0.75 * second)
        targets = np.array([[1, 0, 1], [0, 1, 0]])
        metrics = calculate_metrics(combined, targets, SPEC)
        self.assertEqual(metrics["legacy_mean_group_AttrAcc"], 1.0)
        self.assertEqual(metrics["macro"]["f1"], 1.0)
        self.assertEqual(metrics["instance"]["f1"], 1.0)

    def test_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            blend({"a": np.ones((1, 1)), "b": np.ones((1, 1))},
                  {"a": 0.6, "b": 0.6})


if __name__ == "__main__":
    unittest.main()
