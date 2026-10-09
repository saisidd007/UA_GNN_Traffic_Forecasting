"""
EXACT Ablation Study - Run on official checkpoint with precise evaluation
Uses: results/enhanced_best_model.pt (the exact checkpoint from your 50-epoch training)
Evaluates: 5 model variants on full test set
Output: Exact MAE/RMSE for each variant
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Setup paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import ImprovedTrafficGNN, DiffusionConv

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load dataset
print("Loading dataset...")
dataset = create_enhanced_dataset(
    root_dir=str(DATA),
    sequence_length=12,
    prediction_length=12
)
test_data = dataset.get_test_data()
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
print(f"Test set: {len(test_loader)} batches\n")

# Checkpoint path
checkpoint_path = RESULTS / 'enhanced_best_model.pt'

def load_checkpoint(model, path):
    """Load checkpoint into model"""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    return model

def evaluate_model(model, test_loader, device, model_name=""):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            preds, _, _ = model(batch.x, batch.edge_index, batch.missing_mask, return_uncertainty=False)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(test_loader)} batches...", end='\r')
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    
    print(f"  {model_name:<50} ✓ Complete")
    return mae, rmse

# ============================================================================
# ABLATED MODEL VARIANTS
# ============================================================================

class Ablation_NoDiffusionConv(ImprovedTrafficGNN):
    """w/o Diffusion Conv: Replace DiffusionConv with Identity, keep GAT layers"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        # Replace DiffusionConv with identity, keep GAT layers
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            if isinstance(gnn_layer, DiffusionConv):
                # Skip diffusion conv - use identity transformation
                x = x  # Identity, no transformation
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

class Ablation_NoBiLSTM(ImprovedTrafficGNN):
    """w/o BiLSTM: Use only temporal convolutions"""
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if i % 2 == 1:
                x = x + residual
        
        x_spatial = x.unsqueeze(0)
        x_spatial = self.spatial_attention(x_spatial).squeeze(0)
        
        # Use ONLY temporal conv, skip LSTM
        x_temporal = x.unsqueeze(0)
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)
        x_temporal = x_temporal.squeeze(0)
        
        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        x = self.mc_dropout(x)
        predictions = self.prediction_head(x)
        
        if return_uncertainty:
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

class Ablation_NoAttention(ImprovedTrafficGNN):
    """w/o Graph Attention: Skip spatial attention"""
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
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
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

class Ablation_NoDeterministic(ImprovedTrafficGNN):
    """w/o MC Dropout: Deterministic (no dropout in forward)"""
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
        # SKIP MC Dropout
        
        predictions = self.prediction_head(x)
        
        if return_uncertainty:
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

# ============================================================================
# RUN EXACT ABLATION STUDY
# ============================================================================

print("=" * 80)
print("EXACT ABLATION STUDY - Using Official Checkpoint")
print("=" * 80)
print()

variants = [
    ("Full Model (all components)", ImprovedTrafficGNN),
    ("w/o Diffusion Conv (GCN only)", Ablation_NoDiffusionConv),
    ("w/o BiLSTM (Temporal Conv only)", Ablation_NoBiLSTM),
    ("w/o Graph Attention", Ablation_NoAttention),
    ("w/o MC Dropout (Deterministic)", Ablation_NoDeterministic),
]

results = []

for variant_name, model_class in variants:
    print(f"Evaluating: {variant_name}")
    
    # Create model with exact checkpoint params
    model = model_class(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
        dropout=0.15
    ).to(device)
    
    # Load checkpoint
    model = load_checkpoint(model, checkpoint_path)
    
    # Evaluate
    mae, rmse = evaluate_model(model, test_loader, device, variant_name)
    
    results.append({
        "variant": variant_name,
        "mae": mae,
        "rmse": rmse
    })

# ============================================================================
# PRINT EXACT RESULTS
# ============================================================================

baseline_mae = results[0]["mae"]
baseline_rmse = results[0]["rmse"]

print("\n" + "=" * 90)
print("EXACT ABLATION STUDY RESULTS")
print("=" * 90)
print(f"\n{'Model Variant':<45} {'MAE':<12} {'RMSE':<12} {'Degradation':<15}")
print("-" * 90)

latex_rows = []

for i, result in enumerate(results):
    variant = result["variant"]
    mae = result["mae"]
    rmse = result["rmse"]
    
    if i == 0:
        degradation = "Baseline"
        latex_deg = "Baseline"
    else:
        deg_pct = ((mae - baseline_mae) / baseline_mae) * 100
        degradation = f"{deg_pct:+.2f}%"
        latex_deg = f"{deg_pct:+.2f}\\%"
    
    print(f"{variant:<45} {mae:<12.4f} {rmse:<12.4f} {degradation:<15}")
    latex_rows.append((variant, mae, rmse, latex_deg))

# Save exact results
exact_results = {
    "checkpoint": str(checkpoint_path),
    "test_set_size": len(test_data),
    "baseline": {
        "mae": baseline_mae,
        "rmse": baseline_rmse
    },
    "variants": results
}

with open(RESULTS / 'ablation_exact_results.json', 'w') as f:
    json.dump(exact_results, f, indent=2)

print(f"\nResults saved to: {RESULTS / 'ablation_exact_results.json'}")

# Print LaTeX table
print("\n" + "=" * 90)
print("LATEX TABLE FOR PAPER")
print("=" * 90)
print()
print(r"\begin{table}[!h]")
print(r"\centering")
print(r"\small")
print(r"\caption{Ablation Study: Component Contribution to Overall Performance on PEMS-BAY Dataset}")
print(r"\label{tab:ablation}")
print(r"\resizebox{\columnwidth}{!}{")
print(r"\begin{tabular}{|l|c|c|c|}")
print(r"\hline")
print(r"\textbf{Model Variant} & \textbf{MAE} & \textbf{RMSE} & \textbf{Degradation} \\")
print(r"\hline")
for variant, mae, rmse, deg in latex_rows:
    print(f"{variant} & {mae:.4f} & {rmse:.4f} & {deg} \\\\")
print(r"\hline")
print(r"\end{tabular}}")
print(r"\end{table}")

print("\n✓ Exact ablation study complete!")
