"""
Enhanced Traffic Flow GNN Model with Advanced Architecture
Improved model with multiple attention mechanisms, residual connections, and better uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, LayerNorm
from typing import Optional, Tuple
import math


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
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.output(attended)


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with residual connections"""
    
    def __init__(self, hidden_channels: int, kernel_size: int = 3):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, nodes, features]
        residual = x
        x = x.transpose(1, 2)  # [batch, features, nodes]
        
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        
        x = x.transpose(1, 2)  # [batch, nodes, features]
        return F.relu(x + residual)


class EnhancedTrafficGNN(nn.Module):
    """
    Enhanced Traffic Flow GNN with advanced architecture
    Features:
    - Multi-layer GCN and GAT combination
    - Multi-head spatial attention
    - Temporal convolution blocks
    - Residual connections
    - Advanced uncertainty estimation
    - Missing data handling
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 12,
        num_gnn_layers: int = 4,
        num_temporal_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.15
    ):
        super(EnhancedTrafficGNN, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout
        
        # Input projection with missing data handling
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.missing_data_embedding = nn.Parameter(torch.randn(hidden_channels))
        
        # Multi-layer GNN (combination of GCN and GAT)
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        for i in range(num_gnn_layers):
            if i % 2 == 0:  # Alternate between GCN and GAT
                self.gnn_layers.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.gnn_layers.append(GATConv(hidden_channels, hidden_channels // num_attention_heads, 
                                             heads=num_attention_heads, concat=True, dropout=dropout))
            self.gnn_norms.append(LayerNorm(hidden_channels))
        
        # Spatial attention
        self.spatial_attention = MultiHeadSpatialAttention(hidden_channels, num_attention_heads)
        
        # Temporal processing layers
        self.temporal_layers = nn.ModuleList()
        for _ in range(num_temporal_layers):
            self.temporal_layers.append(TemporalConvBlock(hidden_channels))
        
        # LSTM for long-term temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.lstm_proj = nn.Linear(hidden_channels * 2, hidden_channels)  # Project bidirectional output
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Output heads
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Uncertainty estimation heads
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
        
        # Dropout for MC sampling
        self.mc_dropout = nn.Dropout(dropout)
        
    def handle_missing_data(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Handle missing data by replacing with learned embeddings"""
        if missing_mask is None:
            # Detect missing data (assuming NaN or very small values indicate missing data)
            missing_mask = torch.isnan(x) | (torch.abs(x) < 1e-6)
        
        # Replace missing values with learned embeddings
        x_filled = x.clone()
        if missing_mask.any():
            # Create missing embedding that matches input dimensions
            # x shape: [num_nodes, in_channels], missing_embedding shape: [hidden_channels]
            # We need to project the missing embedding to match input dimensions
            missing_value = torch.zeros_like(x)  # Use zeros for missing values initially
            x_filled = torch.where(missing_mask, missing_value, x)
        
        return x_filled
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        missing_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with enhanced architecture
        
        Args:
            x: Input features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            missing_mask: Boolean mask for missing data [num_nodes, in_channels]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            predictions: Traffic predictions [num_nodes, out_channels]
            aleatoric_uncertainty: Data uncertainty (if return_uncertainty=True)
            epistemic_uncertainty: Model uncertainty (if return_uncertainty=True)
        """
        batch_size = 1
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.squeeze(0) if batch_size == 1 else x.view(-1, x.shape[-1])
        
        num_nodes = x.shape[0]
        
        # Handle missing data
        x = self.handle_missing_data(x, missing_mask)
        
        # Input projection
        x = self.input_proj(x)
        spatial_features = x.clone()
        
        # Multi-layer GNN processing
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            x = gnn_layer(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual connection every 2 layers
            if i % 2 == 1:
                x = x + residual
        
        # Spatial attention
        x_spatial = x.unsqueeze(0)  # Add batch dimension for attention
        x_spatial = self.spatial_attention(x_spatial)
        x_spatial = x_spatial.squeeze(0)
        
        # Temporal processing
        x_temporal = x.unsqueeze(0)  # [1, nodes, features]
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)
        
        # LSTM for long-term dependencies
        lstm_out, _ = self.lstm(x_temporal)
        lstm_out = self.lstm_proj(lstm_out)
        x_temporal = lstm_out.squeeze(0)
        
        # Feature fusion
        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        
        # Apply MC Dropout for uncertainty
        x = self.mc_dropout(x)
        
        # Predictions
        predictions = self.prediction_head(x)
        
        if return_uncertainty:
            # Aleatoric uncertainty (data uncertainty)
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            
            # Epistemic uncertainty (model uncertainty) - estimated through MC dropout
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            
            return predictions, aleatoric_var, epistemic_var
        
        return predictions, None, None


class TrafficFlowLoss(nn.Module):
    """
    Custom loss function for traffic flow prediction with uncertainty
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.05):
        super(TrafficFlowLoss, self).__init__()
        self.alpha = alpha  # Weight for prediction loss
        self.beta = beta    # Weight for aleatoric uncertainty loss
        self.gamma = gamma  # Weight for epistemic uncertainty loss
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        aleatoric_var: Optional[torch.Tensor] = None,
        epistemic_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss with uncertainty
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            aleatoric_var: Aleatoric uncertainty variance
            epistemic_var: Epistemic uncertainty variance
            
        Returns:
            Combined loss value
        """
        # Main prediction loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)
        
        total_loss = self.alpha * mse_loss
        
        if aleatoric_var is not None:
            # Negative log-likelihood for aleatoric uncertainty
            nll_loss = 0.5 * torch.mean(
                torch.log(aleatoric_var) + (predictions - targets) ** 2 / aleatoric_var
            )
            total_loss += self.beta * nll_loss
        
        if epistemic_var is not None:
            # Regularization for epistemic uncertainty
            epistemic_reg = torch.mean(epistemic_var)
            total_loss += self.gamma * epistemic_reg
        
        return total_loss


def create_enhanced_model(
    in_channels: int = 12,
    hidden_channels: int = 128,
    out_channels: int = 12,
    num_gnn_layers: int = 4,
    num_temporal_layers: int = 3,
    num_attention_heads: int = 8,
) -> EnhancedTrafficGNN:
    """
    Factory function to create enhanced traffic GNN model
    
    Args:
        in_channels: Number of input time steps
        hidden_channels: Hidden layer size
        out_channels: Number of prediction time steps
        num_gnn_layers: Number of GNN layers
        num_temporal_layers: Number of temporal layers
        
    Returns:
        Enhanced traffic GNN model
    """
    return EnhancedTrafficGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_gnn_layers=num_gnn_layers,
        num_temporal_layers=num_temporal_layers,
        num_attention_heads=num_attention_heads,
        dropout=0.15
    )


class DiffusionConv(nn.Module):
    """Lightweight diffusion convolution (row-normalized) using edge_index."""

    def __init__(self, in_channels: int, out_channels: int):
        super(DiffusionConv, self).__init__()
        self.theta0 = nn.Linear(in_channels, out_channels, bias=False)
        self.theta1 = nn.Linear(in_channels, out_channels, bias=False)
        self.theta2 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, F], edge_index: [2, E]
        N = x.shape[0]
        device = x.device

        # Build dense adjacency (safe for N~few hundreds)
        A = torch.zeros((N, N), device=device)
        src, dst = edge_index
        A[dst.long(), src.long()] = 1.0

        # Row-normalize A (D^{-1} A)
        deg = A.sum(dim=1)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        A_norm = deg_inv.unsqueeze(1) * A

        left = torch.matmul(A_norm, x)
        right = torch.matmul(A_norm.t(), x)

        out = self.theta0(x) + self.theta1(left) + self.theta2(right)
        return out


class DilatedTemporalConvBlock(nn.Module):
    """WaveNet-style dilated 1D conv block preserving sequence length."""

    def __init__(self, hidden_channels: int, kernel_size: int = 2, dilation: int = 1, dropout: float = 0.1):
        super(DilatedTemporalConvBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features]
        residual = x
        x = x.transpose(1, 2)  # [batch, features, nodes]
        x = F.relu(self.norm(self.conv(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        # Ensure the conv output has the same nodes dimension as the residual.
        # Convolutions with dilation and even kernel sizes can produce
        # off-by-one length changes depending on padding; crop or pad
        # the conv output to match the residual before adding.
        if x.size(1) != residual.size(1):
            if x.size(1) > residual.size(1):
                # crop extra positions
                x = x[:, :residual.size(1), :].contiguous()
            else:
                # pad with zeros to match residual length (pad on the right)
                pad_len = residual.size(1) - x.size(1)
                pad_tensor = torch.zeros((x.size(0), pad_len, x.size(2)), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad_tensor], dim=1)

        return F.relu(x + residual)


class ImprovedTrafficGNN(EnhancedTrafficGNN):
    """Improved model combining diffusion conv + WaveNet-like temporal blocks.

    This subclass reuses many components from EnhancedTrafficGNN but replaces
    the spatial GCNs with diffusion conv and the temporal blocks with dilated
    convolutional stacks for longer receptive field.
    """

    def __init__(self, *args, num_dilations: int = 3, dilation_base: int = 2, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace gnn layers with diffusion conv alternating with GAT
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for i in range(kwargs.get('num_gnn_layers', 4)):
            if i % 2 == 0:
                self.gnn_layers.append(DiffusionConv(self.hidden_channels, self.hidden_channels))
            else:
                self.gnn_layers.append(GATConv(self.hidden_channels, self.hidden_channels // kwargs.get('num_attention_heads', 4),
                                               heads=kwargs.get('num_attention_heads', 4), concat=True, dropout=self.dropout_rate))
            self.gnn_norms.append(LayerNorm(self.hidden_channels))

        # Replace temporal layers with dilated stack
        self.temporal_layers = nn.ModuleList()
        for i in range(kwargs.get('num_temporal_layers', 3)):
            dilation = dilation_base ** (i % num_dilations)
            self.temporal_layers.append(DilatedTemporalConvBlock(self.hidden_channels, kernel_size=2, dilation=dilation, dropout=self.dropout_rate))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, return_uncertainty: bool = True):
        # Reuse the parent's forward but adapt calls to diffusion conv when present
        # Handle missing data and input projection as parent
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)

        # Spatial (diffusion) + attention
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            if isinstance(gnn_layer, DiffusionConv):
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if i % 2 == 1:
                x = x + residual

        x_spatial = x.unsqueeze(0)
        x_spatial = self.spatial_attention(x_spatial).squeeze(0)

        x_temporal = x.unsqueeze(0)
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)

        lstm_out, _ = self.lstm(x_temporal)
        lstm_out = self.lstm_proj(lstm_out)
        x_temporal = lstm_out.squeeze(0)

        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        x = self.mc_dropout(x)
        predictions = self.prediction_head(x)

        if return_uncertainty:
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var

        return predictions, None, None


def create_improved_model(
    in_channels: int = 12,
    hidden_channels: int = 64,
    out_channels: int = 12,
    num_gnn_layers: int = 4,
    num_temporal_layers: int = 3,
    num_attention_heads: int = 4,
) -> ImprovedTrafficGNN:
    return ImprovedTrafficGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_gnn_layers=num_gnn_layers,
        num_temporal_layers=num_temporal_layers,
        num_attention_heads=num_attention_heads,
        dropout=0.15
    )