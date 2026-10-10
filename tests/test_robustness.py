import unittest

import numpy as np

from robustness_experiment import corrupt, forward_fill_input, split_ratios


class RobustnessTests(unittest.TestCase):
    def test_stress_test_uses_source_run_split(self):
        self.assertEqual(split_ratios({"split": "70/10/20"}), (.7, .1))
        self.assertEqual(split_ratios({"val_ratio": .15, "split": "70/10/20"}), (.7, .15))

    def test_corruption_does_not_restore_originally_missing_values(self):
        values = np.ones((2, 4, 12), dtype=np.float32)
        observed = np.ones_like(values, dtype=bool)
        observed[:, 0, 0] = False
        graph = np.eye(4, dtype=np.float32)
        for kind in ("random", "persistent", "clustered", "temporal_block"):
            changed, mask, count = corrupt(values, observed, graph, kind, .25,
                                           np.random.default_rng(17))
            self.assertTrue(np.all(mask <= observed))
            self.assertFalse(mask[:, 0, 0].any())
            self.assertGreater(count, 0)
            self.assertTrue(np.all(changed[~mask & observed] == 0))

    def test_forward_fill_does_not_see_future(self):
        values = np.array([[[2., 0., 0., 7.]]], dtype=np.float32)
        mask = np.array([[[True, False, False, True]]])
        filled = forward_fill_input(values, mask)
        np.testing.assert_array_equal(filled, [[[2., 2., 2., 7.]]])

    def test_persistent_failure_uses_same_sensor_in_all_windows(self):
        values = np.ones((3, 4, 12), dtype=np.float32)
        mask = np.ones_like(values, dtype=bool)
        _, remaining, _ = corrupt(values, mask, np.eye(4), "persistent", .25,
                                  np.random.default_rng(2))
        np.testing.assert_array_equal(remaining[0], remaining[1])
        np.testing.assert_array_equal(remaining[1], remaining[2])


if __name__ == "__main__":
    unittest.main()
