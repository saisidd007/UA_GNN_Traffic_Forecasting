import unittest

import torch

from paper_model import PaperUAGNN, enable_mc_dropout, graph_edges, transitions
from paper_experiment import manuscript_loss


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

    def test_loss_warmup_replaces_mse_with_nll(self):
        mean = torch.tensor([0., 1.])
        variance = torch.tensor([4., 4.])
        target = torch.tensor([2., 0.])
        observed = torch.tensor([True, False])
        self.assertAlmostEqual(manuscript_loss(mean, variance, target, observed, 0).item(), 4.)
        expected_nll = .5 * (torch.log(variance[0]) + 1.)
        self.assertAlmostEqual(manuscript_loss(mean, variance, target, observed, 10).item(),
                               expected_nll.item())
        midpoint = manuscript_loss(mean, variance, target, observed, 5)
        self.assertAlmostEqual(midpoint.item(), (4. + expected_nll.item()) / 2)
        self.assertAlmostEqual(manuscript_loss(mean, variance, target, observed, 10,
                                               no_aleatoric=True).item(), 4.)

    def test_no_dropout_model_is_deterministic_in_mc_mode(self):
        torch.set_num_threads(1)
        model = PaperUAGNN(hidden=16, dropout=0.0)
        graph = torch.tensor([[0., 1.], [1., 0.]])
        forward, reverse = transitions(graph)
        values = torch.randn(2, 12, 2)
        mask = torch.ones_like(values, dtype=torch.bool)
        context = torch.zeros(2, 12, dtype=torch.long)
        enable_mc_dropout(model)
        with torch.no_grad():
            first = model(values, mask, context, context, forward, reverse,
                          graph_edges(graph))[0]
            second = model(values, mask, context, context, forward, reverse,
                           graph_edges(graph))[0]
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
