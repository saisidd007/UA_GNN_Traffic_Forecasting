import torch
from torch_geometric.loader import DataLoader
from src.models.gnn import TrafficGNN
from src.utils.dataset import PEMSBayDataset
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

def test_model(
    model: TrafficGNN,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Test the model and return metrics
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x, batch.edge_index)
            
            predictions.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Calculate metrics
    metrics = {
        'mae': np.mean(np.abs(predictions - targets)),
        'rmse': np.sqrt(np.mean((predictions - targets) ** 2)),
        'mape': np.mean(np.abs((predictions - targets) / targets)) * 100
    }
    
    return metrics, predictions, targets

def plot_predictions(predictions: np.ndarray, targets: np.ndarray, save_path: str):
    """
    Create visualization plots
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Prediction vs Target scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    plt.savefig(f'{save_path}/scatter_plot.png')
    plt.close()
    
    # 2. Error distribution
    errors = predictions.flatten() - targets.flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.savefig(f'{save_path}/error_distribution.png')
    plt.close()
    
    # 3. Time series plot for a random sensor
    sensor_idx = np.random.randint(predictions.shape[1])
    plt.figure(figsize=(15, 6))
    plt.plot(targets[:100, sensor_idx], label='True')
    plt.plot(predictions[:100, sensor_idx], label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Traffic Speed')
    plt.title(f'Predictions vs True Values (Sensor {sensor_idx})')
    plt.legend()
    plt.savefig(f'{save_path}/time_series.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = PEMSBayDataset(root_dir='data')
    
    # Use the test split
    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [int(0.7 * len(dataset)), int(0.15 * len(dataset)), len(dataset) - int(0.85 * len(dataset))]
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load the best model
    model = TrafficGNN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=12,
        num_layers=3
    ).to(device)
    
    model.load_state_dict(torch.load('results/best_model.pt'))
    
    # Test the model
    metrics, predictions, targets = test_model(model, test_loader, device)
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'results/evaluation_{timestamp}'
    os.makedirs(save_path, exist_ok=True)
    
    # Save metrics
    with open(f'{save_path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Create and save plots
    plot_predictions(predictions, targets, save_path)
    
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()