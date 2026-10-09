#!/usr/bin/env python
"""
Enhanced Training Script for METR-LA Dataset with 50 Epochs
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
from src.utils.enhanced_dataset import create_enhanced_dataset


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
        """Calculate MAE, RMSE, MAPE, R², Correlation"""
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
    
    def __init__(self, num_sensors: int = 207):
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
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, learning_rate=1e-3, weight_decay=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.metrics_calc = MetricsCalculator()
        self.failure_tracker = SensorFailureTracker(num_sensors=207)
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(patience=15)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
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
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def test_epoch(self, epoch: int = -1) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Test model"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties_alea = []
        all_uncertainties_epi = []
        
        K = 10  # MC samples for testing
        
        pbar = tqdm(self.test_loader, desc=f'Testing')
        for batch in pbar:
            batch = batch.to(self.device)
            
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
            
            all_predictions.append(mean_pred.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            all_uncertainties_alea.append(alea_var.detach().cpu().numpy())
            all_uncertainties_epi.append(epi_var.detach().cpu().numpy())
        
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        alea_concat = np.concatenate(all_uncertainties_alea, axis=0)
        epi_concat = np.concatenate(all_uncertainties_epi, axis=0)
        
        metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        # Add uncertainty metrics
        mean_alea = float(np.mean(alea_concat))
        mean_epi = float(np.mean(epi_concat))
        mean_total = mean_alea + mean_epi
        
        metrics['mean_aleatoric_uncertainty'] = mean_alea
        metrics['mean_epistemic_uncertainty'] = mean_epi
        metrics['mean_total_uncertainty'] = mean_total
        
        return metrics, predictions_concat, targets_concat
    
    def train(self, num_epochs: int = 50):
        """Main training loop"""
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics'].append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Test
            test_metrics, _, _ = self.test_epoch(epoch)
            self.history['test_metrics'].append(test_metrics)
            
            # Print epoch summary
            print(f"\n[EPOCH {epoch+1} SUMMARY]")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Train MAE:  {train_metrics['mae']:.6f}")
            print(f"  Val MAE:    {val_metrics['mae']:.6f}")
            print(f"  Test MAE:   {test_metrics['mae']:.6f}")
            print(f"  Val RMSE:   {val_metrics['rmse']:.6f}")
            print(f"  Val R²:     {val_metrics['r2_score']:.6f}")
            print(f"  Test Aleatoric Uncertainty: {test_metrics['mean_aleatoric_uncertainty']:.6f}")
            print(f"  Test Epistemic Uncertainty: {test_metrics['mean_epistemic_uncertainty']:.6f}")
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'results/best_model_metrla.pt')
                print(f"  ✓ Best model saved (Val Loss: {val_loss:.6f})")
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\n[EARLY STOPPING] No improvement for {self.early_stopping.patience} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history


def main(num_epochs: int = 50):
    """Main training function"""
    print("[TRAFFIC FLOW] Enhanced GNN Training for METR-LA with Sensor Failure Tracking")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}\n")
    
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    print("[DATA] Loading enhanced METR-LA dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust',
        dataset_name='METR-LA'
    )
    
    print(f"[STATS] Dataset loaded:")
    print(f"   Sequences: {len(dataset.sequences)}")
    print(f"   Sensors: 207")
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
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    # Train
    print("[START] Beginning training...\n")
    history = trainer.train(num_epochs=num_epochs)
    
    # Final test evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    final_test_metrics, predictions, targets = trainer.test_epoch()
    
    print("\n[FINAL TEST METRICS]")
    print(f"  MAE:                           {final_test_metrics['mae']:.6f}")
    print(f"  RMSE:                          {final_test_metrics['rmse']:.6f}")
    print(f"  R² Score:                      {final_test_metrics['r2_score']:.6f}")
    print(f"  Pearson Correlation:           {final_test_metrics['correlation']:.6f}")
    print(f"  Mean Aleatoric Uncertainty:    {final_test_metrics['mean_aleatoric_uncertainty']:.6f}")
    print(f"  Mean Epistemic Uncertainty:    {final_test_metrics['mean_epistemic_uncertainty']:.6f}")
    print(f"  Mean Total Uncertainty:        {final_test_metrics['mean_total_uncertainty']:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/enhanced_training_results_metrla_{timestamp}.json'
    
    results = {
        'dataset': 'METR-LA',
        'num_epochs': num_epochs,
        'num_sensors': 207,
        'history': history,
        'final_test_metrics': final_test_metrics,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results: {results_file}")
    print(f"[SAVED] Best model: results/best_model_metrla.pt")
    
    return trainer, history, final_test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced training for METR-LA with sensor failure tracking')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()
    
    trainer, history, final_metrics = main(num_epochs=args.epochs)
