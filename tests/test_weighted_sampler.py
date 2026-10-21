"""Unit tests for the E3 sample-weight calculation."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter_code"))

from dataset import compute_sample_weights  # noqa: E402


class FakeDataset:
    def __init__(self, rows):
        self.samples = [(f"{index}.jpg", row) for index, row in enumerate(rows)]
        self.col2global = list(range(len(rows[0])))
        self.spec = {"num_attrs": len(rows[0])}

    def __len__(self):
        return len(self.samples)


class WeightedSamplerTest(unittest.TestCase):
    def test_weights_are_positive_finite_and_normalized(self):
        dataset = FakeDataset([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        weights, positive_rate, rarity = compute_sample_weights(dataset)
        self.assertEqual(weights.dtype, torch.double)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[3]), float(weights[0]))
        self.assertGreater(float(weights[4]), float(weights[0]))
        self.assertTrue(torch.allclose(positive_rate, torch.tensor([0.6, 0.2, 0.2])))
        self.assertTrue((rarity >= 1).all())

    def test_attribute_and_image_weights_respect_cap(self):
        rows = [[1, 0] for _ in range(99)] + [[0, 1]]
        dataset = FakeDataset(rows)
        weights, _, rarity = compute_sample_weights(dataset, max_weight=5)
        self.assertEqual(float(rarity[1]), 5.0)
        self.assertGreater(float(weights[-1]), float(weights[0]))

    def test_invalid_parameters_are_rejected(self):
        dataset = FakeDataset([[1], [0]])
        with self.assertRaises(ValueError):
            compute_sample_weights(dataset, power=0)
        with self.assertRaises(ValueError):
            compute_sample_weights(dataset, max_weight=0.5)


if __name__ == "__main__":
    unittest.main()
