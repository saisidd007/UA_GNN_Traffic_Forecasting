#!/usr/bin/env python
"""
PEMS-BAY 50-Epoch Training Script with TSSP Model
Replaces Bi-LSTM with Temporal Self-Supervised Prediction (TSSP)
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn as nn
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import glob
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.newtain_gnn import NewtainTrafficGNN, TrafficFlowLoss, create_newtain_model


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class MetricsCalculator:
    """Calculate comprehensive traffic prediction metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate MAE, RMSE, MAPE, R², correlation
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary of metrics
        """
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-6))) * 100
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-6))
        
        # Correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0
        }


def prompt_yes_no(prompt: str, default: str = 'n') -> bool:
    answer = input(f"{prompt} [{'Y/n' if default.lower() == 'y' else 'y/N'}]: ").strip().lower()
    if answer == '':
        answer = default.lower()
    return answer in ('y', 'yes')


def save_test_predictions(predictions: np.ndarray, targets: np.ndarray, results_dir: str):
    preds_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    pd.DataFrame({'prediction': preds_flat, 'ground_truth': targets_flat}).to_csv(
        os.path.join(results_dir, 'test_predictions_vs_actual.csv'), index=False)
    pd.DataFrame({'prediction': preds_flat}).to_csv(
        os.path.join(results_dir, 'test_predictions.csv'), index=False)
    pd.DataFrame({'ground_truth': targets_flat}).to_csv(
        os.path.join(results_dir, 'test_ground_truths.csv'), index=False)


def plot_training_curves(history: Dict[str, list], results_dir: str):
    epochs = history.get('epoch', [])
    if not epochs:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get('train_loss', []), marker='o', label='Train Loss')
    plt.plot(epochs, history.get('val_loss', []), marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close()

    val_metrics = history.get('val_metrics', [])
    val_mae = [m.get('mae', np.nan) for m in val_metrics]
    if val_mae:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_mae, marker='o', label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Validation MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'mae_curve.png'), bbox_inches='tight')
        plt.close()


def plot_predictions_vs_actual(predictions: np.ndarray, targets: np.ndarray, results_dir: str):
    preds_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    plt.figure(figsize=(6, 6))
    sample_count = min(preds_flat.shape[0], 1000)
    plt.scatter(targets_flat[:sample_count], preds_flat[:sample_count], alpha=0.4, s=8)
    low = min(np.nanmin(targets_flat), np.nanmin(preds_flat))
    high = max(np.nanmax(targets_flat), np.nanmax(preds_flat))
    plt.plot([low, high], [low, high], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'pred_vs_actual.png'), bbox_inches='tight')
    plt.close()


class PEMSBayDataset(Dataset):
    """Simple PyTorch dataset wrapper for PEMS-BAY sequences."""

    def __init__(self, sequences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, time_idx, day_idx = self.sequences[idx]
        return x, y, time_idx, day_idx


def load_pems_bay_data(data_dir: str = '../data', sequence_length: int = 12, prediction_length: int = 12):
    """Load PEMS-BAY dataset and create PyG Data objects"""
    
    print(f"[LOAD] Loading PEMS-BAY data from {data_dir}")
    
    # Load data
    data_path = os.path.join(data_dir, 'PEMS-BAY.csv')
    meta_path = os.path.join(data_dir, 'PEMS-BAY-META.csv')
    
    # Read CSV
    df = pd.read_csv(data_path, index_col=0)
    print(f"   Data shape: {df.shape}")
    
    # Read metadata
    meta_df = pd.read_csv(meta_path)
    num_sensors = len(meta_df)
    print(f"   Sensors: {num_sensors}")

    # Extract temporal context from timestamps
    timestamps = pd.to_datetime(df.index)
    time_of_day = timestamps.hour.astype(int)
    day_of_week = timestamps.dayofweek.astype(int)
    
    # Load adjacency matrix
    import pickle
    adj_path = os.path.join(data_dir, 'adj_mx_bay.pkl')
    adj_mx = None
    
    if os.path.exists(adj_path):
        try:
            with open(adj_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Try to convert to numpy array - handle various formats
            if isinstance(data, np.ndarray):
                adj_mx = data
            elif hasattr(data, 'toarray'):  # Sparse matrix
                adj_mx = data.toarray()
            elif isinstance(data, (list, tuple)):
                try:
                    adj_mx = np.array(data, dtype=np.float32)
                except:
                    print(f"   Could not convert adjacency matrix; using default fully-connected")
                    adj_mx = None
            
            if adj_mx is not None:
                print(f"   Adjacency matrix shape: {adj_mx.shape}")
        except Exception as e:
            print(f"   Error loading adjacency matrix: {e}")
            adj_mx = None
    else:
        print(f"   Adjacency matrix not found at {adj_path}")
        adj_mx = None
    
    # If no adjacency matrix, create fully connected graph
    if adj_mx is None:
        print(f"   Creating fully-connected adjacency matrix ({num_sensors}x{num_sensors})")
        adj_mx = np.ones((num_sensors, num_sensors), dtype=np.float32)
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    
    # Create sequences
    data_tensor = torch.FloatTensor(data_scaled)
    sequences = []
    
    for i in range(len(data_tensor) - sequence_length - prediction_length + 1):
        x = data_tensor[i:i+sequence_length]
        y = data_tensor[i+sequence_length:i+sequence_length+prediction_length]
        time_idx = torch.tensor(int(time_of_day[i + sequence_length]), dtype=torch.long)
        day_idx = torch.tensor(int(day_of_week[i + sequence_length]), dtype=torch.long)
        sequences.append((x, y, time_idx, day_idx))
    
    print(f"   Total sequences: {len(sequences)}")
    
    # Split into train/val/test
    total = len(sequences)
    train_size = int(0.6 * total)
    val_size = int(0.2 * total)
    
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size+val_size]
    test_sequences = sequences[train_size+val_size:]
    
    print(f"   Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    # Convert to edge index from adjacency matrix
    if adj_mx is not None:
        edge_list = []
        for i in range(adj_mx.shape[0]):
            for j in range(adj_mx.shape[1]):
                if adj_mx[i, j] > 0:
                    edge_list.append([i, j])
        
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
        else:
            # Fallback to fully connected if no edges found
            edge_index = torch.ones(num_sensors, num_sensors).nonzero().t()
    else:
        # Create fully connected graph if adjacency matrix not available
        edge_index = torch.ones(num_sensors, num_sensors).nonzero().t()
    
    print(f"   Edge index shape: {edge_index.shape}")
    
    return (train_sequences, val_sequences, test_sequences), edge_index, num_sensors, scaler


def train_epoch(
    model: NewtainTrafficGNN,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: TrafficFlowLoss,
    edge_index: torch.Tensor,
    scaler: GradScaler,
    num_mc_samples: int = 5
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    edge_index_device = edge_index.to(device)

    for batch_x, batch_y, batch_time, batch_day in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_time = batch_time.to(device, non_blocking=True)
        batch_day = batch_day.to(device, non_blocking=True)

        optimizer.zero_grad()
        batch_loss = 0.0

        for sample_idx in range(batch_x.shape[0]):
            x = batch_x[sample_idx]  # [seq_len, num_sensors]
            y = batch_y[sample_idx]  # [pred_len, num_sensors]
            time_idx = batch_time[sample_idx]
            day_idx = batch_day[sample_idx]

            with autocast(enabled=(device.type == 'cuda')):
                mc_predictions = []
                for _ in range(num_mc_samples):
                    pred, aleatoric, epistemic = model(
                        x,
                        edge_index_device,
                        time_of_day_idx=time_idx,
                        day_of_week_idx=day_idx,
                        return_uncertainty=True
                    )
                    mc_predictions.append(pred)

                pred_mean = torch.stack(mc_predictions).mean(dim=0)
                y_transposed = y.t()  # [num_sensors, pred_len]
                loss = loss_fn(pred_mean, y_transposed, aleatoric, epistemic)

            scaler.scale(loss).backward()
            batch_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_loss = batch_loss / batch_x.shape[0]
        total_loss += batch_loss
        batch_count += 1

        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}, Loss: {batch_loss:.4f}")

    return total_loss / max(batch_count, 1)


def validate(
    model: NewtainTrafficGNN,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: TrafficFlowLoss,
    edge_index: torch.Tensor,
    metrics_calc: MetricsCalculator,
    num_samples: int = 10
) -> Tuple[float, Dict]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    batch_count = 0
    edge_index_device = edge_index.to(device)

    with torch.no_grad():
        for batch_x, batch_y, batch_time, batch_day in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_time = batch_time.to(device, non_blocking=True)
            batch_day = batch_day.to(device, non_blocking=True)

            for sample_idx in range(batch_x.shape[0]):
                x = batch_x[sample_idx]
                y = batch_y[sample_idx]
                time_idx = batch_time[sample_idx]
                day_idx = batch_day[sample_idx]

                # Single forward pass in eval mode (no MC sampling for speed)
                with autocast(enabled=(device.type == 'cuda')):
                    pred, aleatoric, epistemic = model(
                        x,
                        edge_index_device,
                        time_of_day_idx=time_idx,
                        day_of_week_idx=day_idx,
                        return_uncertainty=True
                    )

                y_transposed = y.t()
                loss = loss_fn(pred, y_transposed, aleatoric, epistemic)
                total_loss += loss.item()

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_transposed.cpu().numpy())
                batch_count += 1

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    val_loss = total_loss / max(batch_count, 1)
    metrics = metrics_calc.calculate_metrics(all_preds, all_targets)

    return val_loss, metrics


def test_model(
    model: NewtainTrafficGNN,
    test_loader: DataLoader,
    device: torch.device,
    edge_index: torch.Tensor,
    metrics_calc: MetricsCalculator,
    num_samples: int = 30
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Test the model"""
    model.eval()
    # Enable MC Dropout during inference by putting Dropout layers in train mode
    def enable_mc_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    model.apply(enable_mc_dropout)
    all_preds = []
    all_targets = []
    all_aleatoric = []
    all_epistemic_from_preds = []
    sample_count = 0
    edge_index_device = edge_index.to(device)

    print("[TEST] Running inference on test set...")
    with torch.no_grad():
        for batch_x, batch_y, batch_time, batch_day in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_time = batch_time.to(device, non_blocking=True)
            batch_day = batch_day.to(device, non_blocking=True)

            for sample_idx in range(batch_x.shape[0]):
                x = batch_x[sample_idx]
                y = batch_y[sample_idx]
                time_idx = batch_time[sample_idx]
                day_idx = batch_day[sample_idx]

                mc_preds = []
                mc_aleatoric = []
                for _ in range(num_samples):
                    with autocast(enabled=(device.type == 'cuda')):
                        pred, aleatoric, _ = model(
                            x,
                            edge_index_device,
                            time_of_day_idx=time_idx,
                            day_of_week_idx=day_idx,
                            return_uncertainty=True
                        )
                    mc_preds.append(pred)
                    mc_aleatoric.append(aleatoric)

                preds_stack = torch.stack(mc_preds)  # shape: (num_samples, ...prediction shape...)
                pred_mean = preds_stack.mean(dim=0)
                # Epistemic uncertainty: variance across MC predictive samples
                epistemic_var = preds_stack.var(dim=0, unbiased=False)

                # Aleatoric: model-predicted aleatoric mean across MC samples
                aleatoric_mean = torch.stack(mc_aleatoric).mean(dim=0)

                all_preds.append(pred_mean.cpu().numpy())
                all_targets.append(y.t().cpu().numpy())
                all_aleatoric.append(aleatoric_mean.cpu().numpy())
                all_epistemic_from_preds.append(epistemic_var.cpu().numpy())
                sample_count += 1

                if sample_count % 50 == 0:
                    print(f"  Processed {sample_count} test samples")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_aleatoric = np.concatenate(all_aleatoric)
    all_epistemic_from_preds = np.concatenate(all_epistemic_from_preds)

    test_metrics = metrics_calc.calculate_metrics(all_preds, all_targets)
    test_metrics['mean_aleatoric_uncertainty'] = float(np.mean(all_aleatoric))
    test_metrics['mean_epistemic_uncertainty'] = float(np.mean(all_epistemic_from_preds))
    test_metrics['mean_total_uncertainty'] = float(test_metrics['mean_aleatoric_uncertainty'] + test_metrics['mean_epistemic_uncertainty'])

    return test_metrics, all_preds, all_targets

    # Compute mean uncertainties
    mean_aleatoric = float(np.mean(all_aleatoric))
    mean_epistemic = float(np.mean(all_epistemic_from_preds))
    mean_total = float(mean_aleatoric + mean_epistemic)

    test_metrics['mean_aleatoric_uncertainty'] = mean_aleatoric
    test_metrics['mean_epistemic_uncertainty'] = mean_epistemic
    test_metrics['mean_total_uncertainty'] = mean_total

    return test_metrics


def main():
    """Main training function"""
    
    print("=" * 80)
    print("PEMS-BAY 50-EPOCH TRAINING WITH TSSP MODEL")
    print("=" * 80)
    
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 50
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 12
    PREDICTION_LENGTH = 12
    HIDDEN_CHANNELS = 128
    DROPOUT = 0.15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}, Prediction Length: {PREDICTION_LENGTH}")
    print()
    
    # Load data
    (train_data, val_data, test_data), edge_index, num_sensors, scaler = load_pems_bay_data(
        data_dir=DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        prediction_length=PREDICTION_LENGTH
    )
    
    # Create dataloaders using standard PyTorch DataLoader
    train_dataset = PEMSBayDataset(train_data)
    val_dataset = PEMSBayDataset(val_data)
    test_dataset = PEMSBayDataset(test_data)

    pin_memory = DEVICE.type == 'cuda'
    num_workers = 4 if DEVICE.type == 'cuda' else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    # Create model
    print(f"[MODEL] Creating Newtain-GNN with {num_sensors} sensors")
    model = create_newtain_model(
        in_channels=SEQUENCE_LENGTH,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=PREDICTION_LENGTH,
        num_gnn_layers=4,
        num_temporal_layers=4,
        num_attention_heads=4,
        dropout=DROPOUT,
        sequence_length=SEQUENCE_LENGTH,
        time_bins=24,
        day_bins=7,
        time_embedding_dim=8,
        day_embedding_dim=8
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    # Loss function
    loss_fn = TrafficFlowLoss(alpha=1.0, beta=0.1, gamma=0.05)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    early_stop_counter = 0
    early_stopping = EarlyStopping(patience=10)
    
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Record start time to enforce 24-hour cap (reset on resume)
    start_time = time.time()

    # Resume support: detect per-epoch result files and load latest checkpoint
    epoch_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "epoch_*_results_*.json")))
    checkpoint_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "checkpoint_epoch_*.pt")))
    resume_checkpoint_path = None
    if epoch_files:
        try:
            # parse available epoch files to find last completed epoch
            completed_epochs = []
            for ef in epoch_files:
                try:
                    with open(ef, 'r') as fh:
                        ej = json.load(fh)
                    e_num = int(ej.get('epoch', 0))
                    if e_num > 0:
                        completed_epochs.append((e_num, ef, os.path.getmtime(ef)))
                except Exception:
                    continue

            if completed_epochs:
                latest_epoch_file = max(completed_epochs, key=lambda x: x[2])
                last_epoch = latest_epoch_file[0]
                start_epoch = last_epoch + 1
                max_epoch = max(e[0] for e in completed_epochs)
                if max_epoch != last_epoch:
                    print(f"[RESUME] Ignoring older stale epoch files up to {max_epoch}; resuming from latest run epoch {last_epoch}.")

                # Load latest epoch checkpoint if present
                completed_checkpoints = []
                for cf in checkpoint_files:
                    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", cf)
                    if not match:
                        continue
                    try:
                        cp_epoch = int(match.group(1))
                        completed_checkpoints.append((cp_epoch, cf, os.path.getmtime(cf)))
                    except ValueError:
                        continue

                if completed_checkpoints:
                    latest_checkpoint = max(completed_checkpoints, key=lambda x: (x[0], x[2]))
                    checkpoint_epoch, checkpoint_path, _ = latest_checkpoint
                    if checkpoint_epoch == last_epoch:
                        resume_checkpoint_path = checkpoint_path
                        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
                        best_val_mae = checkpoint.get('best_val_mae', best_val_mae)
                        early_stop_counter = checkpoint.get('early_stop_counter', 0)
                        history = checkpoint.get('history', history)
                        print(f"[RESUME] Loaded checkpoint {checkpoint_path}; resuming at epoch {start_epoch}")
                    else:
                        print(f"[RESUME] Found checkpoint for epoch {checkpoint_epoch} but latest completed epoch is {last_epoch}; skipping checkpoint load.")
                else:
                    print("[WARNING] No epoch checkpoint found; the script will ask before restarting from epoch 0.")
                    if not prompt_yes_no("No checkpoint available. Restart from epoch 0 anyway?", default='n'):
                        print("Exiting without starting training.")
                        sys.exit(1)
                    start_epoch = 1

                # Rebuild history from epoch files in order by epoch index
                history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}
                for e_num, ef, _ in sorted(completed_epochs, key=lambda x: x[0]):
                    try:
                        with open(ef, 'r') as fh:
                            ej = json.load(fh)
                        history['epoch'].append(int(ej.get('epoch', 0)))
                        history['train_loss'].append(float(ej.get('train_loss', float('nan'))))
                        history['val_loss'].append(float(ej.get('val_loss', float('nan'))))
                        history['val_metrics'].append(ej.get('val_metrics', {}))
                    except Exception:
                        continue

                if history['val_loss']:
                    try:
                        best_val_loss = min([v for v in history['val_loss'] if not np.isnan(v)])
                    except Exception:
                        best_val_loss = float('inf')
            else:
                start_epoch = 1
        except Exception:
            start_epoch = 1
    else:
        start_epoch = 1

    print("[TRAIN] Starting 50-epoch training with TSSP model...")
    print("-" * 80)
    
    TRAIN_DURATION_LIMIT_HOURS = float(os.getenv('TRAIN_DURATION_LIMIT_HOURS', '24'))
    SKIP_TRAIN = os.getenv('SKIP_TRAIN', '0') == '1'
    if SKIP_TRAIN:
        print("[INFO] SKIP_TRAIN=1 — skipping training loop and proceeding directly to evaluation using existing checkpoint.")
    else:
        for epoch in range(start_epoch, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
        
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE, loss_fn, edge_index, scaler, num_mc_samples=3)
            print(f"  Train Loss: {train_loss:.6f}")
            
            # Validate (reduced MC samples for speed)
            val_loss, val_metrics = validate(model, val_loader, DEVICE, loss_fn, edge_index, MetricsCalculator(), num_samples=1)
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val MAE: {val_metrics['mae']:.6f}, RMSE: {val_metrics['rmse']:.6f}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  LR: {current_lr:.8f}")
            
            # Save history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_loss)

            # Update early stopping counter using validation MAE
            if val_metrics['mae'] < best_val_mae - 1e-8:
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stopping.patience:
                print(f"[EARLY STOPPING] no improvement for {early_stop_counter} epochs; stopping training.")
                break

            # Save best model and checkpoint data based on validation MAE
            is_best = False
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = float(val_metrics['mae'])
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
                print(f"  ✓ Best model saved (Val MAE: {best_val_mae:.6f})")
                is_best = True

            epoch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            epoch_results = {
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_metrics': val_metrics,
                'is_best': is_best,
                'timestamp': epoch_timestamp
            }
            epoch_path = os.path.join(RESULTS_DIR, f'epoch_{epoch}_results_{epoch_timestamp}.json')
            with open(epoch_path, 'w') as ef:
                json.dump(epoch_results, ef, indent=2)

            # Save epoch checkpoint for resume support
            checkpoint_path = os.path.join(RESULTS_DIR, 'latest_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae,
                'early_stop_counter': early_stop_counter,
                'history': history,
            }, checkpoint_path)

            # Save epoch-specific checkpoint copy for more robust resume support
            checkpoint_epoch_path = os.path.join(RESULTS_DIR, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae,
                'early_stop_counter': early_stop_counter,
                'history': history,
            }, checkpoint_epoch_path)

            # Append per-epoch CSV row
            csv_path = os.path.join(RESULTS_DIR, 'pems_bay_epoch_logs.csv')
            csv_row = {
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_mae': float(val_metrics.get('mae', np.nan)),
                'val_rmse': float(val_metrics.get('rmse', np.nan)),
                'val_mape': float(val_metrics.get('mape', np.nan)),
                'val_r2': float(val_metrics.get('r2_score', np.nan)),
                'val_corr': float(val_metrics.get('correlation', np.nan)),
                'is_best': int(is_best),
                'timestamp': epoch_timestamp
            }
            # Write header if file doesn't exist
            if not os.path.exists(csv_path):
                pd.DataFrame([csv_row]).to_csv(csv_path, index=False)
            else:
                pd.DataFrame([csv_row]).to_csv(csv_path, mode='a', header=False, index=False)

            # Optional training duration cap; set TRAIN_DURATION_LIMIT_HOURS=-1 to disable
            elapsed = time.time() - start_time
            if TRAIN_DURATION_LIMIT_HOURS >= 0 and elapsed > TRAIN_DURATION_LIMIT_HOURS * 3600:
                print(f"\n[TIMEOUT] {TRAIN_DURATION_LIMIT_HOURS} hours elapsed. Stopping training at epoch {epoch}.")
                break
    
    print("\n" + "=" * 80)
    print("[TEST] Evaluating on test set...")
    
    # Load best model
    best_model_path = os.path.join(RESULTS_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print(f"[WARNING] best_model.pth not found in {RESULTS_DIR}; loading latest_checkpoint.pth instead.")
        latest_checkpoint_path = os.path.join(RESULTS_DIR, 'latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint_path):
            checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError("No model checkpoint found for final evaluation.")
    
    # Test
    test_metrics, test_preds, test_targets = test_model(model, test_loader, DEVICE, edge_index, MetricsCalculator())
    
    print(f"\nTest Results:")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  R²: {test_metrics['r2_score']:.6f}")
    print(f"  Correlation: {test_metrics['correlation']:.6f}")
    print(f"  Mean Aleatoric Uncertainty: {test_metrics['mean_aleatoric_uncertainty']:.6f}")
    print(f"  Mean Epistemic Uncertainty: {test_metrics['mean_epistemic_uncertainty']:.6f}")
    print(f"  Mean Total Uncertainty: {test_metrics['mean_total_uncertainty']:.6f}")

    # Save test predictions and plots
    save_test_predictions(test_preds, test_targets, RESULTS_DIR)
    plot_training_curves(history, RESULTS_DIR)
    plot_predictions_vs_actual(test_preds, test_targets, RESULTS_DIR)
    print(f"\nSaved test predictions and plots to {RESULTS_DIR}")
    
    print("\n" + "=" * 80)
    print("FINAL TEST METRICS SUMMARY")
    print("=" * 80)
    print(f"Prediction Accuracy:")
    print(f"  Mean Absolute Error (MAE):        {test_metrics['mae']:.6f}")
    print(f"  Root Mean Square Error (RMSE):   {test_metrics['rmse']:.6f}")
    print(f"  Mean Absolute Percentage Error:  {test_metrics['mape']:.2f}%")
    print(f"  R² Score:                        {test_metrics['r2_score']:.6f}")
    print(f"  Pearson Correlation:             {test_metrics['correlation']:.6f}")
    print(f"\nUncertainty Quantification:")
    print(f"  Mean Aleatoric Uncertainty:      {test_metrics['mean_aleatoric_uncertainty']:.6f}")
    print(f"  Mean Epistemic Uncertainty:      {test_metrics['mean_epistemic_uncertainty']:.6f}")
    print(f"  Mean Total Uncertainty:          {test_metrics['mean_total_uncertainty']:.6f}")
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'model': 'TSSP-GNN',
        'dataset': 'PEMS-BAY',
        'epochs': EPOCHS,
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_length': PREDICTION_LENGTH,
        'hidden_channels': HIDDEN_CHANNELS,
        'num_sensors': num_sensors,
        'total_parameters': total_params,
        'device': str(DEVICE),
        'mean_aleatoric_uncertainty': test_metrics['mean_aleatoric_uncertainty'],
        'mean_epistemic_uncertainty': test_metrics['mean_epistemic_uncertainty'],
        'mean_total_uncertainty': test_metrics['mean_total_uncertainty'],
        'training_history': history,
        'test_metrics': test_metrics,
        'timestamp': timestamp
    }
    
    results_path = os.path.join(RESULTS_DIR, f'tssp_training_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        # Convert numpy/tensor values to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    # Save metrics comparison
    metrics_df = pd.DataFrame({
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    })
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'pems_bay_50epoch_metrics.csv'), index=False)
    
    print("=" * 80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    main()
