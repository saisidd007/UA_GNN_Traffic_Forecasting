import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset


def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)


def evaluate_per_horizon(checkpoint_path='results/pems-bay/final_results_pems/enhanced_best_model.pt' if os.path.exists('results/pems-bay/final_results_pems/enhanced_best_model.pt') else 'results/enhanced_best_model.pt',
                         batch_size=8, mc_samples=50, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    data_dir = 'data/pems-bay' if os.path.exists('data/pems-bay') else 'data'
    dataset = create_enhanced_dataset(
        root_dir=data_dir,
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust',
        dataset_name='PEMS-BAY'
    )
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate model with the same architecture used during training (hidden=64, 4 attention heads)
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    )
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    print(f'Loading checkpoint: {checkpoint_path}')
    load_checkpoint(checkpoint_path, model, device)

    model.eval()

    # We'll collect predictions and targets per step
    all_preds = []  # list of arrays [N_nodes, pred_len]
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # MC sampling
            mc_preds = []
            for k in range(mc_samples):
                model.train()  # enable dropout
                preds_k, alea_k, epi_k = model(batch.x, batch.edge_index, batch.missing_mask, return_uncertainty=True)
                mc_preds.append(preds_k.detach().cpu().numpy())

            mc_preds = np.stack(mc_preds, axis=0)  # [K, N, pred_len]
            mean_pred = mc_preds.mean(axis=0)  # [N, pred_len]

            all_preds.append(mean_pred)
            all_targets.append(batch.y.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # [Total_nodes, pred_len]
    targets = np.concatenate(all_targets, axis=0)

    assert preds.shape == targets.shape, f"Shape mismatch preds {preds.shape} targets {targets.shape}"

    pred_len = preds.shape[1]
    mae_per_step = np.mean(np.abs(preds - targets), axis=0)
    rmse_per_step = np.sqrt(np.mean((preds - targets) ** 2, axis=0))

    # Map steps to minutes: each step is 5 minutes (sequence_length=12 -> 60 min)
    step_minutes = [(i + 1) * 5 for i in range(pred_len)]

    print('\nPer-step metrics (step -> minutes):')
    for i in range(pred_len):
        print(f' Step {i+1} ({step_minutes[i]} min): MAE={mae_per_step[i]:.6f}, RMSE={rmse_per_step[i]:.6f}')

    # Common horizons (15,30,45,60) correspond to steps 3,6,9,12 (0-based indices 2,5,8,11)
    horizon_steps = [2, 5, 8, 11]
    print('\nCommon horizon metrics:')
    for s in horizon_steps:
        if s < pred_len:
            print(f' Horizon {step_minutes[s]} min - MAE: {mae_per_step[s]:.6f} | RMSE: {rmse_per_step[s]:.6f}')

    # Also print aggregated mean across horizons used in many papers (3,6,9,12)
    selected_mae = mae_per_step[horizon_steps]
    selected_rmse = rmse_per_step[horizon_steps]
    print('\nAggregated (mean over 15/30/45/60 min):')
    print(f" MAE: {selected_mae.mean():.6f} | RMSE: {selected_rmse.mean():.6f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='results/enhanced_best_model.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--mc', type=int, default=50)
    args = parser.parse_args()

    evaluate_per_horizon(checkpoint_path=args.ckpt, batch_size=args.batch_size, mc_samples=args.mc)
