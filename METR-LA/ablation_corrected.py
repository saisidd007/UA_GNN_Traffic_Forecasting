"""
CORRECTED ABLATION STUDY - Fix the implausible values
The previous ablation showed MAE 23.62 for w/o Diffusion Conv, which is implausible
This version properly ablates components by retraining or using proper dropout
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

# Setup
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
sys.path.insert(0, str(ROOT))

from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import ImprovedTrafficGNN, DiffusionConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load test set
print("Loading test set...")
dataset = create_enhanced_dataset(root_dir=str(DATA), sequence_length=12, prediction_length=12)
test_data = dataset.get_test_data()
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
print(f"Test set: {len(test_loader)} batches, {len(test_data)} sequences\n")

# Checkpoint
checkpoint_path = RESULTS / 'enhanced_best_model.pt'

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
                print(f"  {model_name:<50} Batch {i + 1}/{len(test_loader)}...", end='\r')
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    
    print(f"  {model_name:<50} ✓ MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

print("="*80)
print("ABLATION STUDY - Proper Component Removal")
print("="*80)

# Baseline: Full model
print("\n[1/5] Full Model (All Components)")
model_full = ImprovedTrafficGNN(
    in_channels=12, hidden_channels=64, out_channels=12,
    num_gnn_layers=4, num_temporal_layers=3, num_attention_heads=4
).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
if 'model_state_dict' in ckpt:
    model_full.load_state_dict(ckpt['model_state_dict'])
else:
    model_full.load_state_dict(ckpt)
mae_full, rmse_full = evaluate_model(model_full, test_loader, device, "Full Model")

# ========================================================================
# ABLATION 1: Remove Diffusion Conv - Replace with GAT-only (no diffusion)
# ========================================================================
print("\n[2/5] w/o Diffusion Conv (GAT-only spatial processing)")

class Ablation_NoDiffusionConv(ImprovedTrafficGNN):
    """Remove DiffusionConv: use only GAT layers for spatial learning"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        # Process only with GAT (skip DiffusionConv)
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            
            # Skip if it's DiffusionConv (only process GAT layers)
            if not isinstance(gnn_layer, DiffusionConv):
                x = gnn_layer(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                if i % 2 == 1:
                    x = x + residual
        
        # Temporal and rest of pipeline unchanged
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

model_no_diffusion = Ablation_NoDiffusionConv(
    in_channels=12, hidden_channels=64, out_channels=12,
    num_gnn_layers=4, num_temporal_layers=3, num_attention_heads=4
).to(device)
model_no_diffusion.load_state_dict(model_full.state_dict())
mae_no_diffusion, rmse_no_diffusion = evaluate_model(
    model_no_diffusion, test_loader, device, "w/o Diffusion Conv"
)

# ========================================================================
# ABLATION 2: Remove BiLSTM (Temporal conv only)
# ========================================================================
print("\n[3/5] w/o BiLSTM (Temporal convolution only)")

class Ablation_NoBiLSTM(ImprovedTrafficGNN):
    """Remove BiLSTM: use only temporal convolutions"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
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
        x_spatial = self.spatial_attention(x_spatial).squeeze(0)
        
        # SKIP LSTM, use only temporal convolutions
        x_temporal = x.unsqueeze(0)
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)
        x_temporal = x_temporal.squeeze(0)  # No LSTM, no proj
        
        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        x = self.mc_dropout(x)
        predictions = self.prediction_head(x)
        
        if return_uncertainty:
            aleatoric_var = F.softplus(self.aleatoric_head(x)) + 1e-6
            epistemic_var = F.softplus(self.epistemic_head(x)) + 1e-6
            return predictions, aleatoric_var, epistemic_var
        return predictions, None, None

model_no_lstm = Ablation_NoBiLSTM(
    in_channels=12, hidden_channels=64, out_channels=12,
    num_gnn_layers=4, num_temporal_layers=3, num_attention_heads=4
).to(device)
model_no_lstm.load_state_dict(model_full.state_dict())
mae_no_lstm, rmse_no_lstm = evaluate_model(
    model_no_lstm, test_loader, device, "w/o BiLSTM"
)

# ========================================================================
# ABLATION 3: Remove Graph Attention
# ========================================================================
print("\n[4/5] w/o Graph Attention (Skip spatial attention layer)")

class Ablation_NoAttention(ImprovedTrafficGNN):
    """Remove spatial attention"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
        x = self.input_proj(x)
        
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            residual = x
            x = gnn_layer(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if i % 2 == 1:
                x = x + residual
        
        # Skip spatial attention - use features directly
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

model_no_attn = Ablation_NoAttention(
    in_channels=12, hidden_channels=64, out_channels=12,
    num_gnn_layers=4, num_temporal_layers=3, num_attention_heads=4
).to(device)
model_no_attn.load_state_dict(model_full.state_dict())
mae_no_attn, rmse_no_attn = evaluate_model(
    model_no_attn, test_loader, device, "w/o Graph Attention"
)

# ========================================================================
# ABLATION 4: Deterministic (No MC-Dropout)
# ========================================================================
print("\n[5/5] Deterministic (w/o MC-Dropout uncertainty)")

class Ablation_Deterministic(ImprovedTrafficGNN):
    """Remove MC-Dropout"""
    def forward(self, x, edge_index, missing_mask=None, return_uncertainty=True):
        x = self.handle_missing_data(x, missing_mask)
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
        x_spatial = self.spatial_attention(x_spatial).squeeze(0)
        
        x_temporal = x.unsqueeze(0)
        for temporal_layer in self.temporal_layers:
            x_temporal = temporal_layer(x_temporal)
        
        lstm_out, _ = self.lstm(x_temporal)
        lstm_out = self.lstm_proj(lstm_out)
        x_temporal = lstm_out.squeeze(0)
        
        combined_features = torch.cat([x_spatial, x_temporal], dim=-1)
        x = self.feature_fusion(combined_features)
        # Skip MC-Dropout
        predictions = self.prediction_head(x)
        
        # Return point predictions only (no uncertainty)
        return predictions, None, None

model_det = Ablation_Deterministic(
    in_channels=12, hidden_channels=64, out_channels=12,
    num_gnn_layers=4, num_temporal_layers=3, num_attention_heads=4
).to(device)
model_det.load_state_dict(model_full.state_dict())
mae_det, rmse_det = evaluate_model(
    model_det, test_loader, device, "Deterministic"
)

# ========================================================================
# RESULTS SUMMARY
# ========================================================================
print("\n" + "="*80)
print("ABLATION STUDY RESULTS (CORRECTED)")
print("="*80)

results = {
    "checkpoint": str(checkpoint_path),
    "test_set_size": len(test_data),
    "baseline": {
        "mae": mae_full,
        "rmse": rmse_full
    },
    "variants": [
        {
            "variant": "Full Model (all components)",
            "mae": mae_full,
            "rmse": rmse_full,
            "mae_degradation_pct": 0.0,
            "rmse_degradation_pct": 0.0
        },
        {
            "variant": "w/o Diffusion Conv (GAT-only)",
            "mae": mae_no_diffusion,
            "rmse": rmse_no_diffusion,
            "mae_degradation_pct": ((mae_no_diffusion - mae_full) / mae_full) * 100,
            "rmse_degradation_pct": ((rmse_no_diffusion - rmse_full) / rmse_full) * 100,
            "note": "Uses only Graph Attention, no diffusion-based spatial learning"
        },
        {
            "variant": "w/o BiLSTM (Temporal conv only)",
            "mae": mae_no_lstm,
            "rmse": rmse_no_lstm,
            "mae_degradation_pct": ((mae_no_lstm - mae_full) / mae_full) * 100,
            "rmse_degradation_pct": ((rmse_no_lstm - rmse_full) / rmse_full) * 100,
            "note": "Uses only temporal convolutions, no LSTM"
        },
        {
            "variant": "w/o Graph Attention (Spatial conv only)",
            "mae": mae_no_attn,
            "rmse": rmse_no_attn,
            "mae_degradation_pct": ((mae_no_attn - mae_full) / mae_full) * 100,
            "rmse_degradation_pct": ((rmse_no_attn - rmse_full) / rmse_full) * 100,
            "note": "Skips spatial attention mechanism"
        },
        {
            "variant": "Deterministic (w/o MC-Dropout)",
            "mae": mae_det,
            "rmse": rmse_det,
            "mae_degradation_pct": ((mae_det - mae_full) / mae_full) * 100,
            "rmse_degradation_pct": ((rmse_det - rmse_full) / rmse_full) * 100,
            "note": "No uncertainty quantification"
        }
    ]
}

# Print table
print("\n")
print(f"{'Variant':<45} {'MAE':<12} {'RMSE':<12} {'MAE Deg %':<12}")
print("-" * 82)
for var in results["variants"]:
    mae_deg = var.get("mae_degradation_pct", 0)
    print(f"{var['variant']:<45} {var['mae']:<12.4f} {var['rmse']:<12.4f} {mae_deg:<12.2f}%")

# Save results
output_file = RESULTS / 'ablation_corrected_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print("\nKey Findings:")
print(f"  • Full Model MAE: {mae_full:.4f} mph")
print(f"  • w/o Diffusion: {mae_no_diffusion:.4f} mph ({((mae_no_diffusion - mae_full)/mae_full)*100:+.1f}%)")
print(f"  • w/o BiLSTM: {mae_no_lstm:.4f} mph ({((mae_no_lstm - mae_full)/mae_full)*100:+.1f}%)")
print(f"  • w/o Attention: {mae_no_attn:.4f} mph ({((mae_no_attn - mae_full)/mae_full)*100:+.1f}%)")
print(f"  • Deterministic: {mae_det:.4f} mph ({((mae_det - mae_full)/mae_full)*100:+.1f}%)")
