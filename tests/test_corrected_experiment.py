import unittest

import numpy as np
import torch

from corrected_experiment import TrafficModel, mc_predict, metrics, prepare


class CorrectedExperimentTests(unittest.TestCase):
    def test_preprocessing_uses_training_only_and_causal_fill(self):
        raw = np.arange(80, dtype=np.float32)[:, None]
        raw[4, 0] = np.nan
        first = prepare(raw, history=3, horizon=2, train_ratio=.5, val_ratio=.25)
        changed = raw.copy()
        changed[40:] = 100000
        second = prepare(changed, history=3, horizon=2, train_ratio=.5, val_ratio=.25)
        np.testing.assert_array_equal(first[0][:40], second[0][:40])
        self.assertEqual(first[0][4, 0], first[0][3, 0])
        self.assertFalse(bool(first[1][0, 0]))
        train_end, val_end = first[-1]
        self.assertTrue(all(i + 3 + 2 - 1 < train_end for i in first[4]["train"]))
        self.assertTrue(all(i + 3 >= val_end for i in first[4]["test"]))

    def test_model_axes_and_uncertainty(self):
        torch.manual_seed(2)
        model = TrafficModel(hidden=8)
        x = torch.randn(2, 4, 12)
        mask = torch.ones_like(x, dtype=torch.bool)
        graph = torch.eye(4)
        mean, variance = model(x, mask, graph)
        self.assertEqual(tuple(mean.shape), (2, 4, 12))
        self.assertTrue(bool((variance > 0).all()))
        # Time ordering must affect the temporal representation.
        model.eval()
        with torch.no_grad():
            reversed_mean, _ = model(x.flip(-1), mask, graph)
        self.assertFalse(torch.allclose(mean, reversed_mean))
        _, epistemic, aleatoric = mc_predict(model, x, mask, graph, passes=8)
        self.assertFalse(model.training)
        self.assertTrue(model.dropout.training)
        self.assertTrue(bool((epistemic > 0).any()))
        self.assertTrue(bool((aleatoric > 0).all()))

    def test_intervals_combine_variances(self):
        shape = (2, 3, 12)
        zeros = np.zeros(shape)
        ones = np.ones(shape)
        result = metrics(zeros, ones, ones, ones, np.ones((1, 3, 1)), np.ones(shape, dtype=bool))
        self.assertEqual(result["mae_mph"], 1)
        self.assertEqual(result["coverage_50"], 0)
        self.assertEqual(result["coverage_80"], 1)
        self.assertAlmostEqual(result["width_95_mph"], 2 * 1.95996 * np.sqrt(2))
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, 0] = False
        changed = ones.copy()
        changed[0, 0, 0] = 1000
        masked = metrics(zeros, changed, ones, ones, np.ones((1, 3, 1)), mask)
        self.assertEqual(masked["mae_mph"], 1)


if __name__ == "__main__":
    unittest.main()
