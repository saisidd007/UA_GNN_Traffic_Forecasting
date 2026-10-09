"""
PEMS-BAY Newtain Traffic Flow GNN with Temporal Context Embeddings
This variant preserves the original TSSP pipeline while adding learnable
embeddings for time-of-day and day-of-week before the input projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, LayerNorm
from typing import Optional, Tuple
import math


class TemporalSelfSupervisedPrediction(nn.Module):
    """
    TSSP: Temporal-Spatial State Propagation Block
    Replaces Bi-LSTM with a lightweight temporal state updater and spatial propagation block.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.1):
        super(TemporalSelfSupervisedPrediction, self).__init__()
        self.hidden_channels = hidden_channels

        self.temporal_update = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.spatial_propagation = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        state_update = self.temporal_update(x)

        num_nodes = x.shape[0]
        device = x.device
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        src, dst = edge_index.long()
        adj[dst, src] = 1.0
        deg = adj.sum(dim=1, keepdim=True)
        adj = adj / (deg + 1e-6)

        propagated = torch.matmul(adj, state_update)
        propagated = self.spatial_propagation(propagated)

        output = residual + propagated
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output


class MultiHeadSpatialAttention(nn.Module):
    """Multi-head spatial attention for graph nodes"""

    def __init__(self, hidden_channels: int, num_heads: int = 8):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0

        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.output(attended)


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with residual connections"""

    def __init__(self, hidden_channels: int, kernel_size: int = 3):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = x.transpose(1, 2)
        return F.relu(x + residual)


class NewtainTrafficGNN(nn.Module):
    """
    PEMS-BAY Traffic Flow GNN variant with learnable temporal context embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 12,
        num_gnn_layers: int = 4,
        num_temporal_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.15,
        sequence_length: int = 12,
        time_bins: int = 24,
        day_bins: int = 7,
        time_embedding_dim: int = 8,
        day_embedding_dim: int = 8
    ):
        super(NewtainTrafficGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout
        self.sequence_length = sequence_length

        self.time_embedding = nn.Embedding(time_bins, time_embedding_dim)
        self.day_embedding = nn.Embedding(day_bins, day_embedding_dim)

        self.input_proj = nn.Linear(in_channels + time_embedding_dim + day_embedding_dim, hidden_channels)
        self.missing_data_embedding = nn.Parameter(torch.randn(hidden_channels))

        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=8, padding=8)
        ])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(4)])
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i % 2 == 0:
                self.gnn_layers.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.gnn_layers.append(GATConv(hidden_channels, hidden_channels // num_attention_heads,
                                               heads=num_attention_heads, concat=True, dropout=dropout))
            self.gnn_norms.append(LayerNorm(hidden_channels))

        self.spatial_attention = MultiHeadSpatialAttention(hidden_channels, num_attention_heads)

        self.temporal_layers = nn.ModuleList([TemporalConvBlock(hidden_channels) for _ in range(4)])

        self.tssp = TemporalSelfSupervisedPrediction(hidden_channels=hidden_channels)

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        self.mc_dropout = nn.Dropout(dropout)

    def handle_missing_data(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if missing_mask is None:
            missing_mask = torch.isnan(x) | (torch.abs(x) < 1e-6)
        x_filled = x.clone()
        if missing_mask.any():
            missing_value = torch.zeros_like(x)
            x_filled = torch.where(missing_mask, missing_value, x)
        return x_filled

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_of_day_idx: Optional[torch.Tensor] = None,
        day_of_week_idx: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.shape[0] < x.shape[1]:
            x = x.t()

        x = self.handle_missing_data(x, missing_mask)

        if time_of_day_idx is None:
            time_of_day_idx = torch.tensor(0, dtype=torch.long, device=x.device)
        if day_of_week_idx is None:
            day_of_week_idx = torch.tensor(0, dtype=torch.long, device=x.device)

        if isinstance(time_of_day_idx, torch.Tensor):
            time_of_day_idx = time_of_day_idx.view(-1)[0]
        if isinstance(day_of_week_idx, torch.Tensor):
            day_of_week_idx = day_of_week_idx.view(-1)[0]

        time_context = self.time_embedding(time_of_day_idx).unsqueeze(0).expand(x.size(0), -1)
        day_context = self.day_embedding(day_of_week_idx).unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, time_context, day_context], dim=-1)

        x = self.input_proj(x)

        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            x = gnn_layer(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if i % 2 == 1:
                x = x + residual

        x_spatial = x.unsqueeze(0)
        x_spatial = self.spatial_attention(x_spatial)
        x_spatial = x_spatial.squeeze(0)

        x_temporal = x.unsqueeze(0)
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)
        x_temporal = x_temporal.squeeze(0)

        tssp_out = self.tssp(x_temporal, edge_index)
        x_temporal = tssp_out

        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        x = self.mc_dropout(x)
        predictions = self.prediction_head(x)

        if return_uncertainty:
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var

        return predictions, None, None


class TrafficFlowLoss(nn.Module):
    """Custom loss function for traffic flow prediction with uncertainty"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.05):
        super(TrafficFlowLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aleatoric_var: Optional[torch.Tensor] = None,
        epistemic_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets)
        total_loss = self.alpha * mse_loss
        if aleatoric_var is not None:
            nll_loss = 0.5 * torch.mean(torch.log(aleatoric_var) + (predictions - targets) ** 2 / aleatoric_var)
            total_loss += self.beta * nll_loss
        if epistemic_var is not None:
            epistemic_reg = torch.mean(epistemic_var)
            total_loss += self.gamma * epistemic_reg
        return total_loss


def create_newtain_model(
    in_channels: int = 12,
    hidden_channels: int = 128,
    out_channels: int = 12,
    num_gnn_layers: int = 4,
    num_temporal_layers: int = 3,
    num_attention_heads: int = 8,
    dropout: float = 0.15,
    sequence_length: int = 12,
    time_bins: int = 24,
    day_bins: int = 7,
    time_embedding_dim: int = 8,
    day_embedding_dim: int = 8
) -> NewtainTrafficGNN:
    return NewtainTrafficGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_gnn_layers=num_gnn_layers,
        num_temporal_layers=num_temporal_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        sequence_length=sequence_length,
        time_bins=time_bins,
        day_bins=day_bins,
        time_embedding_dim=time_embedding_dim,
        day_embedding_dim=day_embedding_dim
    )
