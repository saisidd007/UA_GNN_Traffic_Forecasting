"""
Enhanced Training Script for Traffic Flow GNN
Features:
- 50 epochs training
- Advanced model architecture
- Missing data handling
- Comprehensive evaluation
- Early stopping and learning rate scheduling
- Model checkpointing
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
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate various evaluation metrics"""
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Basic metrics
        mae = np.mean(np.abs(pred_flat - target_flat))
        mse = np.mean((pred_flat - target_flat) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (avoid division by zero)
        mask = target_flat != 0
        mape = np.mean(np.abs((pred_flat[mask] - target_flat[mask]) / target_flat[mask])) * 100
        
        # R² score
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Correlation coefficient
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'correlation': float(correlation)
        }


class EnhancedTrainer:
    """Enhanced trainer with advanced features"""
    
    def __init__(
        self,
        model: EnhancedTrafficGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = TrafficFlowLoss(alpha=1.0, beta=0.1, gamma=0.05)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=15, min_delta=1e-4)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Training history
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

        # Use small K for MC during training to estimate epistemic uncertainty (keep low for CPU runs)
        K = 2

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # MC forward passes with dropout enabled
            mc_preds = []
            mc_aleas = []
            for k in range(K):
                # ensure dropout active
                self.model.train()
                preds_k, alea_k, _ = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.missing_mask,
                    return_uncertainty=True
                )
                mc_preds.append(preds_k)
                mc_aleas.append(alea_k)

            mc_preds = torch.stack(mc_preds, dim=0)  # [K, B, N, H]
            mc_aleas = torch.stack(mc_aleas, dim=0)

            pred_mean = mc_preds.mean(dim=0)
            epi_var = mc_preds.var(dim=0)
            alea_mean = mc_aleas.mean(dim=0)
            pred_var = epi_var + alea_mean

            # Loss: MSE + 0.1*NLL (aleatoric aware)
            mse = F.mse_loss(pred_mean, batch.y)
            nll = 0.5 * torch.mean(torch.log(pred_var) + (pred_mean - batch.y) ** 2 / pred_var)
            loss = mse + 0.1 * nll

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Store predictions and targets for metrics
            all_predictions.append(pred_mean.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics
    
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        # Use MC dropout at validation to estimate epistemic uncertainty
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_aleatoric_vars = []
        all_epistemic_vars = []
        all_missing_masks = []

        # Fewer MC samples for validation to reduce memory/time during verification
        K = 5  # MC samples for validation

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation')
        for batch in pbar:
            batch = batch.to(self.device)
            all_missing_masks.append(batch.missing_mask.detach().cpu().numpy())

            # Collect MC samples with dropout enabled
            mc_preds = []
            mc_aleas = []
            for k in range(K):
                self.model.train()  # enable dropout
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

            # Compute validation loss (MSE + NLL)
            mse = F.mse_loss(mean_pred, batch.y)
            nll = 0.5 * torch.mean(torch.log(total_var) + (mean_pred - batch.y) ** 2 / total_var)
            loss = mse + 0.1 * nll
            total_loss += loss.item()

            # Store
            all_predictions.append(mean_pred.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            all_aleatoric_vars.append(alea_var.detach().cpu().numpy())
            all_epistemic_vars.append(epi_var.detach().cpu().numpy())

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        # Add uncertainty metrics
        aleatoric_concat = np.concatenate(all_aleatoric_vars, axis=0)
        epistemic_concat = np.concatenate(all_epistemic_vars, axis=0)
        
        metrics['mean_aleatoric_uncertainty'] = float(np.mean(aleatoric_concat))
        metrics['mean_epistemic_uncertainty'] = float(np.mean(epistemic_concat))
        metrics['total_uncertainty'] = float(np.mean(aleatoric_concat + epistemic_concat))
        
        # Calculate sensor failure percentage (from missing masks)
        if all_missing_masks:
            missing_concat = np.concatenate(all_missing_masks, axis=0)
            # missing_mask == 0 means missing/failed, sum across all but first dimension
            sensor_failure_rate = 100.0 * (1.0 - np.mean(missing_concat))
            metrics['sensor_failure_percentage'] = float(sensor_failure_rate)
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def train(self, num_epochs: int = 50) -> Dict:
        """Main training loop"""
        print(f"[TRAIN] Starting enhanced training for {num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
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
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"  Train MAE: {train_metrics['mae']:.4f} | Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"  Train R²: {train_metrics['r2_score']:.4f} | Val R²: {val_metrics['r2_score']:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Print sensor failure percentage if available
            if 'sensor_failure_percentage' in val_metrics:
                print(f"  Sensor Failure Rate: {val_metrics['sensor_failure_percentage']:.2f}%")
            
            if 'total_uncertainty' in val_metrics:
                print(f"\n[METRICS] Confidence Metrics (95% Confidence Level):")
                print(f"  Mean Aleatoric Uncertainty: {val_metrics['mean_aleatoric_uncertainty']:.4f}")
                print(f"  Mean Epistemic Uncertainty: {val_metrics['mean_epistemic_uncertainty']:.4f}")
                print(f"  Total Uncertainty: {val_metrics['total_uncertainty']:.4f}")
                conf_interval = 1.96 * np.sqrt(val_metrics['total_uncertainty'])
                print(f"  Average Confidence Interval: ±{conf_interval:.4f}")
                if 'confidence_coverage' in val_metrics:
                    coverage = val_metrics['confidence_coverage'] * 100
                    print(f"  Confidence Coverage: {coverage:.1f}% of true values within interval")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.save_checkpoint('results/enhanced_best_model.pt', epoch, is_best=True)
                print(f"  [SAVED] New best model saved! (Val Loss: {val_loss:.6f})")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'results/enhanced_checkpoint_epoch_{epoch+1}.pt', epoch)
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\n[STOP] Early stopping triggered at epoch {epoch+1}")
                print(f"   Best epoch: {best_epoch+1} (Val Loss: {best_val_loss:.6f})")
                break
        
        print(f"\n[DONE] Training completed!")
        print(f"   Best epoch: {best_epoch+1}")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            'history': self.history,
            'is_best': is_best
        }
        
        torch.save(checkpoint, path)
    
    def test_model(self) -> Dict[str, float]:
        """Test the model on test set"""
        print("\n[TEST] Testing model on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_aleatoric_vars = []
        all_epistemic_vars = []
        
        # Use MC dropout for test-time uncertainty estimation (reduced for verification)
        K = 10
        for batch in tqdm(self.test_loader, desc='Testing'):
            batch = batch.to(self.device)

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

            mean_pred = mc_preds.mean(dim=0)
            epi_var = mc_preds.var(dim=0)
            alea_var = mc_aleas.mean(dim=0)

            all_predictions.append(mean_pred.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            all_aleatoric_vars.append(alea_var.detach().cpu().numpy())
            all_epistemic_vars.append(epi_var.detach().cpu().numpy())
        
        # Calculate test metrics
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        test_metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        # Add uncertainty metrics
        aleatoric_concat = np.concatenate(all_aleatoric_vars, axis=0)
        epistemic_concat = np.concatenate(all_epistemic_vars, axis=0)
        
        test_metrics['mean_aleatoric_uncertainty'] = float(np.mean(aleatoric_concat))
        test_metrics['mean_epistemic_uncertainty'] = float(np.mean(epistemic_concat))
        test_metrics['total_uncertainty'] = float(np.mean(aleatoric_concat + epistemic_concat))
        
        print("\n📊 Test Results:")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  R² Score: {test_metrics['r2_score']:.4f}")
        print(f"  Correlation: {test_metrics['correlation']:.4f}")
        print(f"  Total Uncertainty: {test_metrics['total_uncertainty']:.4f}")
        
        return test_metrics


def plot_training_history(history: Dict, save_path: str = 'results/enhanced_training_history.png'):
    """Plot training history"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Traffic GNN Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    train_mae = [m['mae'] for m in history['train_metrics']]
    val_mae = [m['mae'] for m in history['val_metrics']]
    axes[0, 1].plot(epochs, train_mae, 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(epochs, val_mae, 'r-', label='Val MAE', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² plot
    train_r2 = [m['r2_score'] for m in history['train_metrics']]
    val_r2 = [m['r2_score'] for m in history['val_metrics']]
    axes[0, 2].plot(epochs, train_r2, 'b-', label='Train R²', linewidth=2)
    axes[0, 2].plot(epochs, val_r2, 'r-', label='Val R²', linewidth=2)
    axes[0, 2].set_title('R² Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty plot (if available)
    if 'total_uncertainty' in history['val_metrics'][0]:
        val_uncertainty = [m['total_uncertainty'] for m in history['val_metrics']]
        axes[1, 1].plot(epochs, val_uncertainty, 'purple', linewidth=2)
        axes[1, 1].set_title('Validation Uncertainty')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Total Uncertainty')
        axes[1, 1].grid(True, alpha=0.3)
    
    # MAPE plot
    train_mape = [m['mape'] for m in history['train_metrics']]
    val_mape = [m['mape'] for m in history['val_metrics']]
    axes[1, 2].plot(epochs, train_mape, 'b-', label='Train MAPE', linewidth=2)
    axes[1, 2].plot(epochs, val_mape, 'r-', label='Val MAPE', linewidth=2)
    axes[1, 2].set_title('Mean Absolute Percentage Error')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MAPE (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[PLOT] Training history plot saved to {save_path}")


def main(num_epochs: int = 50):
    """Main training function"""
    print("[TRAFFIC FLOW] Enhanced GNN Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load enhanced dataset
    print("\n[DATA] Loading enhanced dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust'
    )
    
    # Print dataset statistics
    stats = dataset.get_data_statistics()
    print(f"\n[STATS] Dataset Statistics:")
    print(f"   Sensors: {stats['num_sensors']}")
    print(f"   Time steps: {stats['num_timesteps']}")
    print(f"   Sequences: {stats['num_sequences']}")
    print(f"   Train/Val/Test: {stats['train_sequences']}/{stats['val_sequences']}/{stats['test_sequences']}")
    print(f"   Missing data: {stats['missing_data_stats']['total_missing_percentage']:.2f}%")
    
    # Create data loaders
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()
    test_data = dataset.get_test_data()
    
    # Reduce batch size to avoid CPU memory pressure on machines without large RAM
    BATCH_SIZE = 8
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create enhanced model
    print("\n[MODEL] Creating enhanced model...")
    # Use a slightly smaller model for quick verification to reduce memory and runtime
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    # Train model
    print("\n[START] Starting training...")
    history = trainer.train(num_epochs=num_epochs)
    
    # Test model
    test_metrics = trainer.test_model()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training history
    results = {
        'training_history': history,
        'test_metrics': test_metrics,
        'dataset_stats': stats,
        'model_config': {
            'in_channels': 12,
            'hidden_channels': 128,
            'out_channels': 12,
            'num_gnn_layers': 4,
            'num_temporal_layers': 3
        },
        'training_config': {
            'num_epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5
        },
        'timestamp': timestamp
    }
    
    with open(f'results/enhanced_training_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\n[SUCCESS] Training completed successfully!")
    print(f"   Results saved with timestamp: {timestamp}")
    print(f"   Best model: results/enhanced_best_model.pt")
    print(f"   Training history: results/enhanced_training_results_{timestamp}.json")
    
    return trainer, history, test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced training runner')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    args = parser.parse_args()

    trainer, history, test_metrics = main(num_epochs=args.epochs)