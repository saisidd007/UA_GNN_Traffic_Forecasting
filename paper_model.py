"""A literal, auditable interpretation of the manuscript's UA-GNN block equations.

This is a new implementation. The manuscript does not specify every engineering
choice (attention residuals, exact output-head layout, masking schedule),
so this module must not be presented as a reproduction of its reported scores.
"""

import torch
from torch import nn
from torch.nn import functional as F


def transitions(adjacency):
    """A[i,j] is an edge i->j; output at i aggregates features at j."""
    adjacency = adjacency.float().clone()
    adjacency.fill_diagonal_(0)
    forward = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1)
    reverse = adjacency.T.contiguous()
    reverse = reverse / reverse.sum(-1, keepdim=True).clamp_min(1)
    return forward, reverse


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(hidden, hidden, 2, dilation=dilation)
        self.norm = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, nodes, hidden = x.shape
        series = x.permute(0, 2, 3, 1).reshape(batch * nodes, hidden, time)
        encoded = self.conv(F.pad(series, (self.dilation, 0)))
        encoded = self.dropout(F.relu(self.norm(encoded)))
        return x + encoded.reshape(batch, nodes, hidden, time).permute(0, 3, 1, 2)


class Diffusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.project = nn.Linear(hidden * 5, hidden)

    def forward(self, x, forward, reverse):
        terms = [x]
        for matrix in (forward, reverse):
            first = torch.einsum("ij,btjc->btic", matrix, x)
            second = torch.einsum("ij,btjc->btic", matrix, first)
            terms.extend((first, second))
        return F.relu(self.project(torch.cat(terms, -1)))


class GraphAttention(nn.Module):
    def __init__(self, hidden, heads=4):
        super().__init__()
        assert hidden % heads == 0
        self.heads, self.width = heads, hidden // heads
        self.project = nn.Linear(hidden, hidden, bias=False)
        self.source = nn.Parameter(torch.empty(heads, self.width))
        self.target = nn.Parameter(torch.empty(heads, self.width))
        self.output = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.source)
        nn.init.xavier_uniform_(self.target)

    def forward(self, x, edges):
        # Softmax over incoming messages for each destination node; only
        # graph edges and self connections participate in attention.
        from torch_geometric.utils import softmax
        batch, time, nodes, hidden = x.shape
        source, target = edges
        z = self.project(x).reshape(batch * time, nodes, self.heads, self.width)
        scores = F.leaky_relu((z[:, source] * self.source).sum(-1) +
                              (z[:, target] * self.target).sum(-1), .2)
        weights = softmax(scores, target, num_nodes=nodes, dim=1)
        messages = (weights[..., None] * z[:, source]).reshape(batch * time, -1, hidden)
        result = torch.zeros_like(z).reshape(batch * time, nodes, hidden)
        result.index_add_(1, target, messages)
        return F.relu(self.output(result.reshape(batch, time, nodes, hidden)))


class PaperUAGNN(nn.Module):
    def __init__(self, horizon=12, hidden=64, dropout=.3, use_diffusion=True):
        super().__init__()
        self.tod = nn.Embedding(288, 8)
        self.dow = nn.Embedding(7, 8)
        self.input = nn.Linear(18, hidden)
        self.temporal = nn.ModuleList(TemporalBlock(hidden, d, dropout) for d in (1, 2, 4, 8))
        self.diffusion = nn.ModuleList(Diffusion(hidden) for _ in range(4))
        self.attention = nn.ModuleList(GraphAttention(hidden) for _ in range(4))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.mean = nn.Linear(hidden * 2, horizon)
        self.variance = nn.Linear(hidden * 2, horizon)
        self.use_diffusion = use_diffusion

    def forward(self, values, observed, tod, dow, forward, reverse, edges):
        # values and observed: [batch, history, sensors]; context: [batch, history].
        batch, time, nodes = values.shape
        context = torch.cat((self.tod(tod), self.dow(dow)), -1)
        context = context[:, :, None].expand(-1, -1, nodes, -1)
        x = torch.cat((values[..., None], observed.float()[..., None], context), -1)
        x = self.input(x)
        for temporal, diffusion, attention in zip(self.temporal, self.diffusion, self.attention):
            x = temporal(x)
            if self.use_diffusion:
                x = diffusion(x, forward, reverse)
            x = attention(x, edges)
        series = x.permute(0, 2, 1, 3).reshape(batch * nodes, time, -1)
        encoded, _ = self.lstm(series)
        # Last forward state and first backward state cover the complete input history.
        summary = torch.cat((encoded[:, -1, :self.lstm.hidden_size],
                             encoded[:, 0, self.lstm.hidden_size:]), -1)
        summary = self.dropout(summary).reshape(batch, nodes, -1)
        return self.mean(summary), F.softplus(self.variance(summary)) + 1e-6


def graph_edges(adjacency):
    """Return source j, destination i for A[i,j] > 0, with self connections."""
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(1)
    target, source = (adjacency > 0).nonzero(as_tuple=True)
    return source, target


def enable_mc_dropout(model):
    """Keep normalization frozen while sampling dropout masks at inference."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
