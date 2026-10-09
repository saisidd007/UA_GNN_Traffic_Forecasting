#!/usr/bin/env python
"""
Enhanced Training Script with Sensor Failure Rate Tracking
Shows sensor failure % for every epoch at 0%, 5%, and 10% dropout rates
"""

import os
import gc
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


class RunningMetricsCalculator:
    """Calculate comprehensive metrics incrementally with O(1) memory"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.running_abs_err = 0.0
        self.running_sq_err = 0.0
        self.running_ape = 0.0
        self.running_count = 0
        self.running_y_sum = 0.0
        self.running_y_sq_sum = 0.0
        self.running_pred_sum = 0.0
        self.running_pred_sq_sum = 0.0
        self.running_prod_sum = 0.0
    
    def update(self, pred: np.ndarray, y: np.ndarray):
        pred = pred.reshape(-1).astype(np.float64)
        y = y.reshape(-1).astype(np.float64)
        
        self.running_abs_err += float(np.sum(np.abs(pred - y)))
        self.running_sq_err += float(np.sum((pred - y) ** 2))
        self.running_ape += float(np.sum(np.abs((y - pred) / (np.abs(y) + 1e-8))))
        self.running_count += pred.size
        self.running_y_sum += float(np.sum(y))
        self.running_y_sq_sum += float(np.sum(y ** 2))
        self.running_pred_sum += float(np.sum(pred))
        self.running_pred_sq_sum += float(np.sum(pred ** 2))
        self.running_prod_sum += float(np.sum(pred * y))
    
    def compute(self) -> Dict[str, float]:
        if self.running_count == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2_score': 0.0, 'correlation': 0.0}
        
        mae = self.running_abs_err / self.running_count
        rmse = np.sqrt(self.running_sq_err / self.running_count)
        mape = (self.running_ape / self.running_count) * 100.0
        
        ss_res = self.running_sq_err
        mean_y = self.running_y_sum / self.running_count
        ss_tot = self.running_y_sq_sum - self.running_count * (mean_y ** 2)
        r2_score = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        cov = self.running_prod_sum - (self.running_pred_sum * self.running_y_sum) / self.running_count
        var_pred = max(self.running_pred_sq_sum - (self.running_pred_sum ** 2) / self.running_count, 0.0)
        var_y = max(ss_tot, 0.0)
        denom = np.sqrt(var_pred * var_y)
        correlation = float(cov / denom) if denom > 1e-8 else 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2_score),
            'correlation': float(correlation)
        }


class SensorFailureTracker:
    """Track sensor failures at different dropout rates with streaming O(1) memory"""
    
    def __init__(self, num_sensors: int = 325):
        self.num_sensors = num_sensors
        self.reset()
    
    def reset(self):
        self.total_elements = 0
        self.original_missing = 0
        self.valid_sensors = 0
    
    def update(self, missing_mask_tensor: torch.Tensor):
        total = missing_mask_tensor.numel()
        missing = int((missing_mask_tensor == 0).sum().item())
        valid = total - missing
        self.total_elements += total
        self.original_missing += missing
        self.valid_sensors += valid
    
    def compute_failure_rates(self) -> Dict[str, float]:
        if self.total_elements == 0:
            return {'sensor_failure_0pct': 0.0, 'sensor_failure_5pct': 0.0, 'sensor_failure_10pct': 0.0}
        
        rates = {}
        for dropout_rate in [0, 5, 10]:
            if dropout_rate > 0:
                num_dropped = int(self.valid_sensors * dropout_rate / 100)
                failed = self.original_missing + num_dropped
            else:
                failed = self.original_missing
            failure_pct = 100.0 * failed / self.total_elements
            rates[f'sensor_failure_{dropout_rate}pct'] = float(failure_pct)
        return rates


class EnhancedTrainer:
    """Enhanced training with sensor failure tracking and dual normalized/denormalized metric tracking"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        output_dir='results/pems-bay/final_results_pems',
        learning_rate=1e-3,
        weight_decay=1e-5,
        scaler=None,
        num_sensors=325
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.scaler = scaler
        self.num_sensors = num_sensors
        
        # Metrics calculators for normalized scale
        self.train_metrics_calc = RunningMetricsCalculator()
        self.val_metrics_calc = RunningMetricsCalculator()
        
        # Metrics calculators for denormalized scale (original physical units, mph)
        self.train_metrics_calc_denorm = RunningMetricsCalculator()
        self.val_metrics_calc_denorm = RunningMetricsCalculator()
        
        self.failure_tracker = SensorFailureTracker(num_sensors=num_sensors)
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(patience=15)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'train_metrics_denorm': [],
            'val_metrics_denorm': [],
            'learning_rates': []
        }
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original physical scale (mph)"""
        if self.scaler is None or not hasattr(self.scaler, 'scale_') or not hasattr(self.scaler, 'center_'):
            return data
        center = self.scaler.center_.reshape(1, self.num_sensors, 1)
        scale = self.scaler.scale_.reshape(1, self.num_sensors, 1)
        reshaped = data.reshape(-1, self.num_sensors, data.shape[-1])
        denorm = reshaped * scale + center
        return denorm.reshape(data.shape)
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        self.train_metrics_calc.reset()
        self.train_metrics_calc_denorm.reset()
        
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
            
            pred_mean_np = pred_mean.detach().cpu().numpy()
            y_np = batch.y.detach().cpu().numpy()
            
            # Streaming metrics (normalized)
            self.train_metrics_calc.update(pred_mean_np, y_np)
            
            # Streaming metrics (denormalized)
            pred_denorm = self.denormalize(pred_mean_np)
            y_denorm = self.denormalize(y_np)
            self.train_metrics_calc_denorm.update(pred_denorm, y_denorm)
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        metrics = self.train_metrics_calc.compute()
        metrics_denorm = self.train_metrics_calc_denorm.compute()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics, metrics_denorm
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        self.val_metrics_calc.reset()
        self.val_metrics_calc_denorm.reset()
        self.failure_tracker.reset()
        
        K = 5  # MC samples for validation
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation')
        for batch in pbar:
            batch = batch.to(self.device)
            self.failure_tracker.update(batch.missing_mask)
            
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
            
            mean_pred_np = mean_pred.detach().cpu().numpy()
            y_np = batch.y.detach().cpu().numpy()
            
            # Normalized
            self.val_metrics_calc.update(mean_pred_np, y_np)
            
            # Denormalized
            mean_pred_denorm = self.denormalize(mean_pred_np)
            y_denorm = self.denormalize(y_np)
            self.val_metrics_calc_denorm.update(mean_pred_denorm, y_denorm)
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        metrics = self.val_metrics_calc.compute()
        failure_rates = self.failure_tracker.compute_failure_rates()
        metrics.update(failure_rates)
        
        metrics_denorm = self.val_metrics_calc_denorm.compute()
        metrics_denorm.update(failure_rates)
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics, metrics_denorm
    
    def test_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate best model on test set in both normalized and denormalized scales"""
        self.model.eval()
        test_calc_norm = RunningMetricsCalculator()
        test_calc_denorm = RunningMetricsCalculator()
        
        running_alea = 0.0
        running_epi = 0.0
        running_alea_denorm = 0.0
        running_epi_denorm = 0.0
        total_elements = 0
        
        scale_sq = (self.scaler.scale_ ** 2).reshape(1, self.num_sensors, 1) if (self.scaler is not None and hasattr(self.scaler, 'scale_')) else 1.0
        
        K = 10  # MC samples for testing
        
        pbar = tqdm(self.test_loader, desc='Testing Evaluation')
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
            
            mean_pred_np = mean_pred.detach().cpu().numpy()
            y_np = batch.y.detach().cpu().numpy()
            alea_np = alea_var.detach().cpu().numpy()
            epi_np = epi_var.detach().cpu().numpy()
            
            # Normalized
            test_calc_norm.update(mean_pred_np, y_np)
            
            # Denormalized
            pred_denorm = self.denormalize(mean_pred_np)
            y_denorm = self.denormalize(y_np)
            test_calc_denorm.update(pred_denorm, y_denorm)
            
            # Uncertainty accumulation
            running_alea += float(np.sum(alea_np))
            running_epi += float(np.sum(epi_np))
            
            if self.scaler is not None and hasattr(self.scaler, 'scale_'):
                alea_denorm = alea_np.reshape(-1, self.num_sensors, alea_np.shape[-1]) * scale_sq
                epi_denorm = epi_np.reshape(-1, self.num_sensors, epi_np.shape[-1]) * scale_sq
                running_alea_denorm += float(np.sum(alea_denorm))
                running_epi_denorm += float(np.sum(epi_denorm))
            else:
                running_alea_denorm += float(np.sum(alea_np))
                running_epi_denorm += float(np.sum(epi_np))
                
            total_elements += mean_pred_np.size
        
        test_metrics_norm = test_calc_norm.compute()
        test_metrics_norm['mean_aleatoric_uncertainty'] = running_alea / total_elements if total_elements > 0 else 0.0
        test_metrics_norm['mean_epistemic_uncertainty'] = running_epi / total_elements if total_elements > 0 else 0.0
        test_metrics_norm['mean_total_uncertainty'] = test_metrics_norm['mean_aleatoric_uncertainty'] + test_metrics_norm['mean_epistemic_uncertainty']
        
        test_metrics_denorm = test_calc_denorm.compute()
        test_metrics_denorm['mean_aleatoric_uncertainty'] = running_alea_denorm / total_elements if total_elements > 0 else 0.0
        test_metrics_denorm['mean_epistemic_uncertainty'] = running_epi_denorm / total_elements if total_elements > 0 else 0.0
        test_metrics_denorm['mean_total_uncertainty'] = test_metrics_denorm['mean_aleatoric_uncertainty'] + test_metrics_denorm['mean_epistemic_uncertainty']
        
        return test_metrics_norm, test_metrics_denorm
    
    def train(self, num_epochs: int = 50) -> Dict:
        """Main training loop"""
        print(f"\n[TRAIN] Starting enhanced training for {num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Training
            train_loss, train_metrics, train_metrics_denorm = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics, val_metrics_denorm = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['train_metrics_denorm'].append(train_metrics_denorm)
            self.history['val_metrics_denorm'].append(val_metrics_denorm)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"\n[RESULTS] Epoch {epoch+1}:")
            print(f"  Loss                 -> Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            print(f"  Normalized MAE       -> Train: {train_metrics['mae']:.4f} | Val: {val_metrics['mae']:.4f}")
            print(f"  Normalized RMSE      -> Train: {train_metrics['rmse']:.4f} | Val: {val_metrics['rmse']:.4f}")
            print(f"  Normalized R²        -> Train: {train_metrics['r2_score']:.4f} | Val: {val_metrics['r2_score']:.4f}")
            print(f"  Denormalized MAE     -> Train: {train_metrics_denorm['mae']:.4f} | Val: {val_metrics_denorm['mae']:.4f} mph")
            print(f"  Denormalized RMSE    -> Train: {train_metrics_denorm['rmse']:.4f} | Val: {val_metrics_denorm['rmse']:.4f} mph")
            print(f"  Denormalized R²      -> Train: {train_metrics_denorm['r2_score']:.4f} | Val: {val_metrics_denorm['r2_score']:.4f}")
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
                best_model_state = self.model.state_dict().copy()
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'metrics_denorm': val_metrics_denorm
                }, os.path.join(self.output_dir, 'enhanced_best_model.pt'))
                print(f"\n[SAVED] New best model! (Val Loss: {val_loss:.6f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'metrics_denorm': val_metrics_denorm
                }, os.path.join(self.output_dir, f'enhanced_checkpoint_epoch_{epoch+1}.pt'))
                print(f"[CHECKPOINT] Saved epoch {epoch+1}")
            
            # Memory cleanup between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
        
        # Load best model for final evaluation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history


def main(num_epochs: int = 50, output_dir: str = 'results/pems-bay/final_results_pems'):
    """Main training function"""
    print("[TRAFFIC FLOW] Enhanced GNN Training with Sensor Failure Tracking")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("[DATA] Loading enhanced dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust',
        dataset_name='PEMS-BAY'
    )
    
    num_sensors = dataset.raw_data.shape[1]
    scaler = getattr(dataset.preprocessor, 'scalers', {}).get('main', None)
    
    print(f"[STATS] Dataset loaded:")
    print(f"   Sequences: {len(dataset.sequences)}")
    print(f"   Sensors: {num_sensors}")
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
        weight_decay=1e-5,
        scaler=scaler,
        num_sensors=num_sensors
    )
    
    # Train
    print("[START] Beginning training...\n")
    history = trainer.train(num_epochs=num_epochs)
    
    # Final test evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    final_test_metrics, final_test_metrics_denorm = trainer.test_epoch()
    
    print("\n[FINAL TEST METRICS - NORMALIZED]")
    print(f"  MAE:                           {final_test_metrics['mae']:.6f}")
    print(f"  RMSE:                          {final_test_metrics['rmse']:.6f}")
    print(f"  MAPE:                          {final_test_metrics['mape']:.2f}%")
    print(f"  R² Score:                      {final_test_metrics['r2_score']:.6f}")
    print(f"  Pearson Correlation:           {final_test_metrics['correlation']:.6f}")
    print(f"  Mean Aleatoric Uncertainty:    {final_test_metrics['mean_aleatoric_uncertainty']:.6f}")
    print(f"  Mean Epistemic Uncertainty:    {final_test_metrics['mean_epistemic_uncertainty']:.6f}")
    print(f"  Mean Total Uncertainty:        {final_test_metrics['mean_total_uncertainty']:.6f}")
    
    print("\n[FINAL TEST METRICS - DENORMALIZED (mph)]")
    print(f"  MAE:                           {final_test_metrics_denorm['mae']:.4f} mph")
    print(f"  RMSE:                          {final_test_metrics_denorm['rmse']:.4f} mph")
    print(f"  MAPE:                          {final_test_metrics_denorm['mape']:.2f}%")
    print(f"  R² Score:                      {final_test_metrics_denorm['r2_score']:.4f}")
    print(f"  Pearson Correlation:           {final_test_metrics_denorm['correlation']:.4f}")
    print(f"  Mean Aleatoric Uncertainty:    {final_test_metrics_denorm['mean_aleatoric_uncertainty']:.4f}")
    print(f"  Mean Epistemic Uncertainty:    {final_test_metrics_denorm['mean_epistemic_uncertainty']:.4f}")
    print(f"  Mean Total Uncertainty:        {final_test_metrics_denorm['mean_total_uncertainty']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'enhanced_training_results_{timestamp}.json')
    
    results = {
        'dataset': 'PEMS-BAY',
        'num_epochs': num_epochs,
        'num_sensors': num_sensors,
        'history': history,
        'final_test_metrics': final_test_metrics,
        'final_test_metrics_denorm': final_test_metrics_denorm,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results: {results_file}")
    
    return trainer, history, final_test_metrics, final_test_metrics_denorm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced training with sensor failure tracking')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='results/pems-bay/final_results_pems', help='Output directory for checkpoints and logs')
    args = parser.parse_args()
    
    trainer, history, test_norm, test_denorm = main(num_epochs=args.epochs, output_dir=args.output_dir)
