import unittest

import torch

from paper_model import PaperUAGNN, enable_mc_dropout, graph_edges, transitions


class PaperModelTests(unittest.TestCase):
    def test_directed_two_hop_convention(self):
        adjacency = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        forward, reverse = transitions(adjacency)
        features = torch.tensor([[0.], [2.], [7.]])
        self.assertEqual((forward @ features).flatten().tolist(), [2., 7., 0.])
        self.assertEqual((forward @ forward @ features).flatten().tolist(), [7., 0., 0.])
        self.assertEqual((reverse @ features).flatten().tolist(), [0., 0., 2.])

    def test_history_and_sensor_axes(self):
        torch.set_num_threads(1)
        model = PaperUAGNN(hidden=16)
        graph = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        forward, reverse = transitions(graph)
        values = torch.randn(2, 12, 3)
        mask = torch.ones_like(values, dtype=torch.bool)
        context = torch.zeros(2, 12, dtype=torch.long)
        mean, variance = model(values, mask, context, context, forward, reverse, graph_edges(graph))
        self.assertEqual(mean.shape, (2, 3, 12))
        self.assertEqual(variance.shape, (2, 3, 12))
        self.assertTrue(torch.all(variance > 0))
        mean.sum().backward()
        self.assertIsNotNone(model.temporal[0].conv.weight.grad)

    def test_mc_dropout_leaves_batch_norm_frozen(self):
        model = PaperUAGNN(hidden=16)
        enable_mc_dropout(model)
        self.assertFalse(model.training)
        self.assertFalse(model.temporal[0].norm.training)
        self.assertTrue(model.temporal[0].dropout.training)


if __name__ == "__main__":
    unittest.main()
