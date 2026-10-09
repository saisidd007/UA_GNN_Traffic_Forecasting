"""
Real Ablation Study: Evaluate model variants with components disabled.
Measures actual MAE/RMSE on test set for:
- Full Model (baseline)
- w/o Diffusion Conv (GCN only)
- w/o BiLSTM (Temporal Conv only)
- w/o Graph Attention
- w/o MC Dropout (Deterministic)

Usage:
    python ablation_real.py

Output:
    - Prints ablation table with MAE/RMSE for each variant
    - Saves results/ablation_real_results.json
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Setup paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import ImprovedTrafficGNN, DiffusionConv, GATConv

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load dataset
print("Loading dataset...")
dataset = create_enhanced_dataset(
    root_dir=str(DATA),
    sequence_length=12,
    prediction_length=12
)
test_data = dataset.get_test_data()
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
print(f"Test batches: {len(test_loader)}")

# Checkpoint path
checkpoint_path = RESULTS / 'enhanced_best_model.pt'
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

def load_checkpoint(model, path):
    """Load checkpoint state dict into model"""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set, return MAE and RMSE"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # Forward pass
            preds, _, _ = model(batch.x, batch.edge_index, batch.missing_mask, return_uncertainty=False)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)  # [num_samples, num_nodes, horizon]
    targets = np.concatenate(all_targets, axis=0)
    
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    return mae, rmse

# ============================================================================
# Define Model Variants with Components Disabled
# ============================================================================

class AblatedGNN_NoDiffusionConv(ImprovedTrafficGNN):
    """Disable Diffusion Conv: use only GAT layers"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        # Use only GAT layers, skip diffusion conv
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            # Only apply GAT (skip DiffusionConv)
            if not isinstance(gnn_layer, DiffusionConv):
                x = gnn_layer(x, edge_index)
                x = norm(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
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
            aleatoric_var = torch.nn.functional.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = torch.nn.functional.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

class AblatedGNN_NoBiLSTM(ImprovedTrafficGNN):
    """Disable BiLSTM: use only temporal conv"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        # Multi-layer GNN with diffusion
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            if isinstance(gnn_layer, DiffusionConv):
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
            x = norm(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
            if i % 2 == 1:
                x = x + residual
        
        x_spatial = x.unsqueeze(0)
        x_spatial = self.spatial_attention(x_spatial).squeeze(0)
        
        # Temporal processing WITHOUT LSTM
        x_temporal = x.unsqueeze(0)
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)
        x_temporal = x_temporal.squeeze(0)
        
        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        x = self.mc_dropout(x)
        predictions = self.prediction_head(x)
        
        if return_uncertainty:
            aleatoric_var = torch.nn.functional.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = torch.nn.functional.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

class AblatedGNN_NoAttention(ImprovedTrafficGNN):
    """Disable spatial attention"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            if isinstance(gnn_layer, DiffusionConv):
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
            x = norm(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
            if i % 2 == 1:
                x = x + residual
        
        # SKIP spatial attention
        x_spatial = x
        
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
            aleatoric_var = torch.nn.functional.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = torch.nn.functional.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

class AblatedGNN_NoDeterministic(ImprovedTrafficGNN):
    """Disable MC Dropout (deterministic)"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            if isinstance(gnn_layer, DiffusionConv):
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
            x = norm(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
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
        # SKIP MC Dropout
        
        predictions = self.prediction_head(x)
        
        if return_uncertainty:
            aleatoric_var = torch.nn.functional.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = torch.nn.functional.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

# ============================================================================
# Run Ablation Study
# ============================================================================

variants = [
    ("Full Model (all components)", ImprovedTrafficGNN, True),
    ("w/o Diffusion Conv (GCN only)", AblatedGNN_NoDiffusionConv, True),
    ("w/o BiLSTM (Temporal Conv only)", AblatedGNN_NoBiLSTM, True),
    ("w/o Graph Attention", AblatedGNN_NoAttention, True),
    ("w/o MC Dropout (Deterministic)", AblatedGNN_NoDeterministic, True),
]

results = []

for variant_name, model_class, use_checkpoint in variants:
    print(f"\nEvaluating: {variant_name}")
    print("-" * 60)
    
    # Create model
    model = model_class(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,  # Checkpoint was trained with 4 heads
        dropout=0.15
    ).to(device)
    
    # Load checkpoint
    if use_checkpoint:
        model = load_checkpoint(model, checkpoint_path)
    
    # Evaluate
    mae, rmse = evaluate_model(model, test_loader, device)
    
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    results.append({
        "variant": variant_name,
        "mae": float(mae),
        "rmse": float(rmse)
    })

# ============================================================================
# Compute Improvements & Print Table
# ============================================================================

baseline_mae = results[0]["mae"]
baseline_rmse = results[0]["rmse"]

print("\n" + "=" * 80)
print("ABLATION STUDY RESULTS")
print("=" * 80)
print(f"\n{'Model Variant':<40} {'MAE':<12} {'RMSE':<12} {'Improvement':<12}")
print("-" * 80)

latex_rows = []

for i, result in enumerate(results):
    variant = result["variant"]
    mae = result["mae"]
    rmse = result["rmse"]
    
    if i == 0:
        improvement = "Baseline"
        latex_improvement = "Baseline"
    else:
        improvement_pct = ((mae - baseline_mae) / baseline_mae) * 100
        improvement = f"{improvement_pct:+.2f}%"
        latex_improvement = f"{improvement_pct:+.2f}\\%"
    
    print(f"{variant:<40} {mae:<12.4f} {rmse:<12.4f} {improvement:<12}")
    
    # Prepare LaTeX row
    latex_rows.append(f"{variant} & {mae:.4f} & {rmse:.4f} & {latex_improvement} \\\\")

# Print LaTeX table
print("\n" + "=" * 80)
print("LaTeX Table:")
print("=" * 80)
print(r"\begin{table}[!h]")
print(r"\centering")
print(r"\small")
print(r"\caption{Ablation Study: Component Contribution to Overall Performance on PEMS-BAY Dataset}")
print(r"\label{tab:ablation}")
print(r"\resizebox{\columnwidth}{!}{")
print(r"\begin{tabular}{|l|c|c|c|}")
print(r"\hline")
print(r"\textbf{Model Variant} & \textbf{MAE} & \textbf{RMSE} & \textbf{Improvement} \\")
print(r"\hline")
for row in latex_rows:
    print(row)
print(r"\hline")
print(r"\end{tabular}}")
print(r"\end{table}")

# Save results
results_dict = {
    "baseline": {
        "mae": baseline_mae,
        "rmse": baseline_rmse
    },
    "variants": results
}

with open(RESULTS / 'ablation_real_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nResults saved to: {RESULTS / 'ablation_real_results.json'}")
print("\nAblation study complete!")
