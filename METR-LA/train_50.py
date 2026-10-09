#!/usr/bin/env python
"""
Enhanced Training Script with Sensor Failure Rate Tracking
Shows sensor failure % for every epoch at 0%, 5%, and 10% dropout rates
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from src.models.enhanced_gnn import EnhancedTrafficGNN, TrafficFlowLoss, create_improved_model
from src.utils.enhanced_dataset import EnhancedPEMSBayDataset, create_enhanced_dataset


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
    """Calculate comprehensive metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate MAE, RMSE, MAPE, R²"""
        # Reshape to 2D if needed
        if predictions.ndim > 2:
            predictions = predictions.reshape(-1)
        if targets.ndim > 2:
            targets = targets.reshape(-1)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2_score),
            'correlation': float(correlation)
        }


class SensorFailureTracker:
    """Track sensor failures at different dropout rates"""
    
    def __init__(self, num_sensors: int = 325):
        self.num_sensors = num_sensors
        self.failure_rates = {0: [], 5: [], 10: []}
    
    def calculate_sensor_failure_at_dropout(self, missing_mask: np.ndarray, dropout_rate: float = 0) -> float:
        """
        Calculate sensor failure percentage with simulated dropout
        
        Args:
            missing_mask: Original missing data mask (1 = present, 0 = missing)
            dropout_rate: Simulated sensor dropout rate (0-100)
            
        Returns:
            Failure percentage (0-100)
        """
        # Flatten to handle batch dimension
        mask_flat = missing_mask.flatten()
        
        # Original missing sensors
        original_missing = np.sum(mask_flat == 0)
        
        # Simulate additional dropout
        if dropout_rate > 0:
            valid_sensors = np.where(mask_flat == 1)[0]
            num_to_dropout = int(len(valid_sensors) * dropout_rate / 100)
            if num_to_dropout > 0:
                dropout_indices = np.random.choice(valid_sensors, size=num_to_dropout, replace=False)
                mask_with_dropout = mask_flat.copy()
                mask_with_dropout[dropout_indices] = 0
                failed = np.sum(mask_with_dropout == 0)
            else:
                failed = original_missing
        else:
            failed = original_missing
        
        failure_percentage = 100.0 * failed / len(mask_flat)
        return failure_percentage


class EnhancedTrainer:
    """Enhanced training with sensor failure tracking"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, output_dir='results', learning_rate=1e-3, weight_decay=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.metrics_calc = MetricsCalculator()
        self.failure_tracker = SensorFailureTracker()
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(patience=15)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} - Training')
        
        K = 2  # MC samples for training
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # MC forward passes
            mc_preds = []
            mc_aleas = []
            for k in range(K):
                self.model.train()
                preds_k, alea_k, _ = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.missing_mask,
                    return_uncertainty=True
                )
                mc_preds.append(preds_k)
                mc_aleas.append(alea_k)
            
            mc_preds = torch.stack(mc_preds, dim=0)
            mc_aleas = torch.stack(mc_aleas, dim=0)
            
            pred_mean = mc_preds.mean(dim=0)
            epi_var = mc_preds.var(dim=0)
            alea_mean = mc_aleas.mean(dim=0)
            pred_var = epi_var + alea_mean
            
            # Loss with uncertainty awareness
            mse = F.mse_loss(pred_mean, batch.y)
            nll = 0.5 * torch.mean(torch.log(pred_var + 1e-8) + (pred_mean - batch.y) ** 2 / (pred_var + 1e-8))
            loss = mse + 0.1 * nll
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(pred_mean.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_missing_masks = []
        
        K = 5  # MC samples for validation
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation')
        for batch in pbar:
            batch = batch.to(self.device)
            all_missing_masks.append(batch.missing_mask.detach().cpu().numpy())
            
            mc_preds = []
            mc_aleas = []
            for k in range(K):
                self.model.train()  # Keep dropout enabled
                with torch.no_grad():
                    preds_k, alea_k, _ = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.missing_mask,
                        return_uncertainty=True
                    )
                mc_preds.append(preds_k)
                mc_aleas.append(alea_k)
            
            mc_preds = torch.stack(mc_preds, dim=0)
            mc_aleas = torch.stack(mc_aleas, dim=0)
            
            mean_pred = mc_preds.mean(dim=0)
            epi_var = mc_preds.var(dim=0)
            alea_var = mc_aleas.mean(dim=0)
            total_var = epi_var + alea_var
            
            mse = F.mse_loss(mean_pred, batch.y)
            nll = 0.5 * torch.mean(torch.log(total_var + 1e-8) + (mean_pred - batch.y) ** 2 / (total_var + 1e-8))
            loss = mse + 0.1 * nll
            total_loss += loss.item()
            
            all_predictions.append(mean_pred.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        # Calculate sensor failure rates at different dropout levels
        if all_missing_masks:
            missing_concat = np.concatenate(all_missing_masks, axis=0)
            for dropout_rate in [0, 5, 10]:
                failure_rate = self.failure_tracker.calculate_sensor_failure_at_dropout(missing_concat, dropout_rate)
                metrics[f'sensor_failure_{dropout_rate}pct'] = float(failure_rate)
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def train(self, num_epochs: int = 50) -> Dict:
        """Main training loop"""
        print(f"\n[TRAIN] Starting enhanced training for {num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"\n[RESULTS] Epoch {epoch+1}:")
            print(f"  Loss      -> Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            print(f"  MAE       -> Train: {train_metrics['mae']:.4f} | Val: {val_metrics['mae']:.4f}")
            print(f"  RMSE      -> Train: {train_metrics['rmse']:.4f} | Val: {val_metrics['rmse']:.4f}")
            print(f"  R²        -> Train: {train_metrics['r2_score']:.4f} | Val: {val_metrics['r2_score']:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Print sensor failure rates
            print(f"\n[SENSOR FAILURES]:")
            for dropout in [0, 5, 10]:
                key = f'sensor_failure_{dropout}pct'
                if key in val_metrics:
                    failure_pct = val_metrics[key]
                    print(f"  Dropout {dropout:2d}% -> Sensor Failure: {failure_pct:6.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics
                }, os.path.join(self.output_dir, 'enhanced_best_model.pt'))
                print(f"\n[SAVED] New best model! (Val Loss: {val_loss:.6f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics
                }, os.path.join(self.output_dir, f'enhanced_checkpoint_epoch_{epoch+1}.pt'))
                print(f"[CHECKPOINT] Saved epoch {epoch+1}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n[STOP] Early stopping at epoch {epoch+1}")
                print(f"   Best epoch: {best_epoch+1} (Loss: {best_val_loss:.6f})")
                break
        
        print(f"\n{'='*80}")
        print(f"[COMPLETE] Training finished!")
        print(f"   Best epoch: {best_epoch+1}")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"{'='*80}\n")
        
        return self.history


def main(num_epochs: int = 50, output_dir: str = 'results'):
    """Main training function"""
    print("[TRAFFIC FLOW] Enhanced GNN Training with Sensor Failure Tracking")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("[DATA] Loading enhanced dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data/metr-la',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust',
        dataset_name='METR-LA'
    )
    
    print(f"[STATS] Dataset loaded:")
    print(f"   Sequences: {len(dataset.sequences)}")
    print(f"   Sensors: {dataset.raw_data.shape[1]}")
    print(f"   Train/Val/Test: {len(dataset.train_indices)}/{len(dataset.val_indices)}/{len(dataset.test_indices)}\n")
    
    # Create data loaders
    BATCH_SIZE = 8
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()
    test_data = dataset.get_test_data()
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("[MODEL] Creating enhanced GNN...")
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    # Train
    print("[START] Beginning training...\n")
    history = trainer.train(num_epochs=num_epochs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'enhanced_training_results_{timestamp}.json')
    
    results = {
        'history': history,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[SAVED] Results: {results_file}")
    
    return trainer, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced training with sensor failure tracking')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for checkpoints and logs')
    args = parser.parse_args()
    
    trainer, history = main(num_epochs=args.epochs, output_dir=args.output_dir)
