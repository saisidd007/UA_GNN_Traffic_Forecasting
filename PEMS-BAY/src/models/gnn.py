import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from typing import Optional

class TrafficGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Spatio-Temporal GNN model with uncertainty estimation
        
        Args:
            in_channels (int): Number of input time steps
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of prediction time steps
            num_layers (int): Number of GNN layers
            dropout (float): Dropout rate for uncertainty estimation
        """
        super(TrafficGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Initial projection layer
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GNN layers for spatial dependencies
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        
        # GRU layer for temporal dependencies
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # MC Dropout for uncertainty
        self.dropout = nn.Dropout(dropout)
        
        # Temporal attention
        self.attention = TemporalAttention(hidden_channels)
        
        # Uncertainty estimation
        self.aleatoric_var = nn.Linear(hidden_channels, out_channels)  # For aleatoric uncertainty
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_uncertainty: bool = True):
        """
        Forward pass with uncertainty estimation
        
        Args:
            x (torch.Tensor): Input features [num_nodes, in_channels]
            edge_index (torch.Tensor): Graph connectivity [2, num_edges]
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            tuple: (predictions, aleatoric_uncertainty) if return_uncertainty=True
                  predictions only if return_uncertainty=False
        """
        batch_size = 1
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            
        # Initial projection
        x = self.input_proj(x)
        
        # Apply GCN for spatial dependencies
        x = self.convs[0](x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        
        # Reshape for GRU
        x = x.view(batch_size, -1, x.size(-1))  # [batch, nodes, features]
        
        # Apply GRU for temporal dependencies
        x, _ = self.gru(x)
        
        # Apply MC Dropout for uncertainty estimation
        x = self.dropout(x)
        
        # Predictions and uncertainty
        mean = self.output_proj(x)
        aleatoric_var = F.softplus(self.aleatoric_var(x))  # Ensure positive variance
        
        # Reshape back if needed
        if batch_size == 1:
            mean = mean.squeeze(0)
            aleatoric_var = aleatoric_var.squeeze(0)
        
        if return_uncertainty:
            return mean, aleatoric_var
        return mean

class TemporalAttention(nn.Module):
    def __init__(self, hidden_channels: int):
        """
        Temporal attention mechanism
        
        Args:
            hidden_channels (int): Number of hidden features
        """
        super(TemporalAttention, self).__init__()
        
        self.att = nn.Parameter(torch.randn(hidden_channels))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention
        
        Args:
            x (torch.Tensor): Input features [num_nodes, hidden_channels]
            
        Returns:
            torch.Tensor: Attended features [num_nodes, hidden_channels]
        """
        # Compute attention weights
        weights = torch.matmul(x, self.att)
        weights = self.softmax(weights)
        
        # Apply attention
        return x * weights.unsqueeze(-1)