import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
from src.models.gnn import TrafficGNN
from src.utils.dataset import PEMSBayDataset
from typing import Tuple, Dict
import json
from datetime import datetime


def negative_log_likelihood_loss(pred_mean, pred_var, target):
    """Calculate negative log-likelihood loss for uncertainty"""
    return 0.5 * torch.mean(torch.log(pred_var) + (pred_mean - target)**2 / pred_var)

def simulate_sensor_failures(x: torch.Tensor, failure_rate: float = 0.2) -> torch.Tensor:
    """Simulate random sensor failures by masking values"""
    mask = torch.rand_like(x) > failure_rate
    return x * mask

def train_epoch(
    model: TrafficGNN,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_mc_samples: int = 5
) -> float:
    """Train for one epoch with uncertainty estimation"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Simulate sensor failures during training
        batch.x = simulate_sensor_failures(batch.x)
        
        # Forward pass with multiple MC samples
        mc_predictions = []
        mc_uncertainties = []
        for _ in range(num_mc_samples):
            pred, uncertainty = model(batch.x, batch.edge_index)
            mc_predictions.append(pred)
            mc_uncertainties.append(uncertainty)
            
        # Calculate mean and variance of predictions
        pred_stack = torch.stack(mc_predictions)
        uncertainty_stack = torch.stack(mc_uncertainties)
        pred_mean = torch.mean(pred_stack, dim=0)
        # Combine model uncertainty (variance across samples) with aleatoric uncertainty
        pred_var = torch.var(pred_stack, dim=0) + torch.mean(uncertainty_stack, dim=0)
        
        # Combined loss with uncertainty
        nll_loss = negative_log_likelihood_loss(pred_mean, pred_var, batch.y)
        mse_loss = F.mse_loss(pred_mean, batch.y)
        loss = mse_loss + 0.1 * nll_loss  # Weight for uncertainty loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(
    model: TrafficGNN,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10
) -> Tuple[float, Dict[str, float]]:
    """Validate the model with uncertainty estimation"""
    model.train()  # Keep in train mode for MC Dropout
    total_loss = 0
    predictions_list = []
    uncertainties_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(device)
            mc_preds = []
            mc_uncertainties = []
            
            # Monte Carlo sampling
            for _ in range(num_samples):
                pred, uncertainty = model(batch.x, batch.edge_index)
                mc_preds.append(pred)
                mc_uncertainties.append(uncertainty)
            
            # Stack MC samples
            mc_preds = torch.stack(mc_preds)  # [num_samples, batch, nodes, steps]
            mc_uncertainties = torch.stack(mc_uncertainties)
            
            # Calculate mean prediction and total uncertainty
            mean_pred = mc_preds.mean(dim=0)
            epistemic_uncertainty = mc_preds.var(dim=0)
            aleatoric_uncertainty = mc_uncertainties.mean(dim=0)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # Calculate loss
            loss = F.mse_loss(mean_pred, batch.y)
            total_loss += loss.item()
            
            # Store results
            predictions_list.append(mean_pred.cpu().numpy())
            uncertainties_list.append(total_uncertainty.cpu().numpy())
            targets_list.append(batch.y.cpu().numpy())
    
    # Concatenate all results
    predictions = np.concatenate(predictions_list, axis=0)
    uncertainties = np.concatenate(uncertainties_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # Avoid division by zero in MAPE
    mape_mask = targets != 0
    mape = np.mean(np.abs((predictions[mape_mask] - targets[mape_mask]) / targets[mape_mask])) * 100
    
    # Calculate confidence intervals (95%)
    confidence_intervals = 1.96 * np.sqrt(uncertainties)
    
    metrics = {
        'mae': np.mean(np.abs(predictions - targets)),
        'rmse': np.sqrt(np.mean((predictions - targets) ** 2)),
        'mape': mape,
        'mean_uncertainty': np.mean(uncertainties),
        'mean_confidence_interval': np.mean(confidence_intervals),
        'confidence_coverage': np.mean(np.abs(predictions - targets) <= confidence_intervals)
    }
    
    return total_loss / len(val_loader), metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = PEMSBayDataset(root_dir='data')
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Use larger batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Create model
    model = TrafficGNN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=12,  # Predict next 12 time steps
        num_layers=3
    ).to(device)
    
    # Training setup
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10  # Quick trial run
    best_val_loss = float('inf')
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, device)
        
        # Save results
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_metrics'].append(metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Metrics: {metrics}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/best_model.pt')
        
        # Save checkpoint only at the end
        if epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, f'results/final_checkpoint.pt')
    
    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/training_results_{timestamp}.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()