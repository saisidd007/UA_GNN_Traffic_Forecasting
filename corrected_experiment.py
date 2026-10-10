"""Leakage-free, time-axis-corrected traffic forecasting experiment.

The original publication pipeline and its artifacts are deliberately retained.
This script writes each new run to a fresh output directory.
"""

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent


def load_data(name):
    folder = ROOT / name / "data" / ("metr-la" if name == "METR-LA" else "pems-bay")
    path = folder / f"{name}.csv"
    table = pd.read_csv(path)
    timestamps = table.iloc[:, 0].astype(str).tolist()
    sensor_ids = table.columns[1:].astype(str).tolist()
    raw = table.iloc[:, 1:].to_numpy(dtype=np.float32)
    if name == "PEMS-BAY":
        with (folder / "adj_mx_bay.pkl").open("rb") as stream:
            saved = pickle.load(stream, encoding="latin1")
        if isinstance(saved, list) and saved[0] != sensor_ids:
            raise ValueError("PEMS-BAY adjacency sensor order differs from CSV")
        adjacency = np.asarray(saved[2] if isinstance(saved, list) else saved)
    else:
        meta = pd.read_csv(folder / "METR-LA-META.csv")
        if meta["sensor_id"].astype(str).tolist() != sensor_ids:
            raise ValueError("METR-LA metadata sensor order differs from CSV")
        official_graph = folder / "adj_mx.pkl"
        if official_graph.exists():
            with official_graph.open("rb") as stream:
                saved = pickle.load(stream, encoding="latin1")
            if saved[0] != sensor_ids:
                raise ValueError("METR-LA adjacency sensor order differs from CSV")
            adjacency = np.asarray(saved[2], dtype=np.float32)
            return raw, timestamps, adjacency
        # Explicit fallback for checkouts lacking the benchmark adjacency.
        coords = np.deg2rad(meta[["latitude", "longitude"]].to_numpy())
        lat, lon = coords[:, 0], coords[:, 1]
        delta_lat = lat[:, None] - lat[None, :]
        delta_lon = lon[:, None] - lon[None, :]
        distance = 2 * np.arcsin(np.sqrt(np.sin(delta_lat / 2) ** 2 +
                            np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2) ** 2))
        adjacency = np.zeros((len(coords), len(coords)), dtype=np.float32)
        nearest = np.argsort(distance, axis=1)[:, 1:9]
        adjacency[np.arange(len(coords))[:, None], nearest] = 1
    return raw, timestamps, adjacency


def prepare(raw, history=12, horizon=12, train_ratio=.7, val_ratio=.15):
    """Split by target time; fit imputation and scaling on training history only."""
    n = len(raw)
    train_end, val_end = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    # These benchmark files contain all-sensor zero rows and almost no positive
    # readings below 1 mph. Treat zero as a missing-data sentinel for these files.
    observed = np.isfinite(raw) & (raw != 0)
    filled = np.empty_like(raw)
    previous = np.zeros(raw.shape[1], dtype=np.float32)
    for t in range(n):
        previous = np.where(observed[t], raw[t], previous)
        filled[t] = previous
    center = np.median(filled[:train_end], axis=0)
    q25, q75 = np.percentile(filled[:train_end], [25, 75], axis=0)
    scale = np.where(q75 > q25, q75 - q25, 1.0)
    scaled = ((filled - center) / scale).astype(np.float32)
    # A window belongs to the period containing all of its forecast targets.
    starts = np.arange(n - history - horizon + 1)
    target_first = starts + history
    target_last = target_first + horizon - 1
    splits = {
        "train": starts[target_last < train_end],
        "val": starts[(target_first >= train_end) & (target_last < val_end)],
        "test": starts[target_first >= val_end],
    }
    return scaled, observed, center, scale, splits, (train_end, val_end)


class TrafficModel(nn.Module):
    """Graph aggregation per time step, then temporal convolution and LSTM."""

    def __init__(self, history=12, horizon=12, hidden=32, dropout=.15):
        super().__init__()
        self.input = nn.Linear(2, hidden)
        self.graph = nn.Linear(hidden * 2, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mean = nn.Linear(hidden, horizon)
        self.variance = nn.Linear(hidden, horizon)

    def forward(self, values, observed, adjacency):
        # values/observed: [batch, sensors, history]
        features = torch.stack((values, observed.float()), dim=-1)
        x = F.relu(self.input(features))  # [batch, sensors, history, channels]
        neighbors = torch.einsum("ij,bjtc->bitc", adjacency, x)
        x = F.relu(self.graph(torch.cat((x, neighbors), dim=-1)))
        batch, sensors, history, channels = x.shape
        x = x.reshape(batch * sensors, history, channels)
        x = F.relu(self.temporal(x.transpose(1, 2))).transpose(1, 2)
        x, _ = self.lstm(x)  # LSTM sequence length is history
        x = self.dropout(x[:, -1]).reshape(batch, sensors, channels)
        return self.mean(x), F.softplus(self.variance(x)) + 1e-6


def mc_predict(model, x, mask, adjacency, passes=30):
    model.eval()
    model.dropout.train()  # Other layers, including any future batch norm, stay in eval mode.
    with torch.no_grad():
        samples = [model(x, mask, adjacency) for _ in range(passes)]
    means = torch.stack([sample[0] for sample in samples])
    variances = torch.stack([sample[1] for sample in samples])
    return means.mean(0), means.var(0, unbiased=False), variances.mean(0)


def metrics(pred, target, epistemic, aleatoric, scale, target_observed):
    physical_pred = pred * scale + 0  # centered offset cancels in errors
    physical_target = target * scale
    error = (physical_pred - physical_target)[target_observed]
    total_std = (np.sqrt(np.maximum(epistemic + aleatoric, 0)) * scale)[target_observed]
    result = {
        "mae_mph": float(np.abs(error).mean()),
        "rmse_mph": float(np.sqrt(np.square(error).mean())),
        "epistemic_variance_normalized": float(epistemic[target_observed].mean()),
        "aleatoric_variance_normalized": float(aleatoric[target_observed].mean()),
    }
    for level, z in ((50, .67449), (80, 1.28155), (90, 1.64485), (95, 1.95996)):
        result[f"coverage_{level}"] = float((np.abs(error) <= z * total_std).mean())
        result[f"width_{level}_mph"] = float((2 * z * total_std).mean())
    full_error = np.abs(physical_pred - physical_target)
    result["mae_by_horizon_mph"] = [float(full_error[:, :, step][target_observed[:, :, step]].mean())
                                     for step in range(full_error.shape[-1])]
    result["observed_test_targets"] = int(target_observed.sum())
    return result


def persistence_baseline(name, train_ratio=.7, val_ratio=.15, test_windows=None):
    """Repeat the last input reading across horizons on the same test windows."""
    raw, _, _ = load_data(name)
    values, observed, _, scale, splits, _ = prepare(raw, train_ratio=train_ratio, val_ratio=val_ratio)
    starts = splits["test"]
    if test_windows and test_windows < len(starts):
        starts = starts[np.linspace(0, len(starts)-1, test_windows, dtype=int)]
    last = values[starts + 11]
    target = np.stack([values[starts + 12 + step] for step in range(12)], axis=1)
    mask = np.stack([observed[starts + 12 + step] for step in range(12)], axis=1)
    error = (last[:, None, :] - target) * scale[None, None, :]
    valid = error[mask]
    return {"mae_mph": float(np.abs(valid).mean()),
            "rmse_mph": float(np.sqrt(np.square(valid).mean())),
            "mae_by_horizon_mph": [float(np.abs(error[:, step, :][mask[:, step, :]]).mean())
                                   for step in range(12)],
            "observed_test_targets": int(mask.sum()), "test_windows": int(len(starts))}


def run(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw, timestamps, adjacency = load_data(args.dataset)
    values, observed, center, scale, splits, boundaries = prepare(
        raw, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = torch.tensor((adjacency > .1).astype(np.float32), device=device)
    graph.fill_diagonal_(1)
    graph /= graph.sum(1, keepdim=True).clamp_min(1)
    if args.no_graph:
        graph = torch.eye(graph.shape[0], device=device)
    model = TrafficModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batches(split, limit, shuffle):
        indices = splits[split].copy()
        if shuffle:
            np.random.shuffle(indices)
        if limit is not None and limit < len(indices):
            indices = indices[:limit] if shuffle else indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]
        for part in np.array_split(indices, math.ceil(len(indices) / args.batch_size)):
            x = np.stack([values[i:i+12].T for i in part])
            mask = np.stack([observed[i:i+12].T for i in part])
            y = np.stack([values[i+12:i+24].T for i in part])
            y_mask = np.stack([observed[i+12:i+24].T for i in part])
            yield (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device),
                   torch.from_numpy(y).to(device), torch.from_numpy(y_mask).to(device))

    log = []
    best = float("inf")
    best_epoch = 0
    start_epoch = 0
    if args.resume:
        state = torch.load(output / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        log, best, best_epoch = state["log"], state["best"], state["best_epoch"]
        start_epoch = state["epoch"]
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_random_state"])
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []
        for x, mask, y, y_mask in batches("train", args.train_windows, True):
            if not y_mask.any():
                continue
            optimizer.zero_grad()
            mean, variance = model(x, mask, graph)
            squared = (mean-y).square()
            loss = squared[y_mask].mean() + .1 * (.5 * (variance.log() + squared/variance))[y_mask].mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        validation_error, validation_count = 0.0, 0
        with torch.no_grad():
            for x, mask, y, y_mask in batches("val", args.val_windows, False):
                if not y_mask.any():
                    continue
                mean, _ = model(x, mask, graph)
                validation_error += (mean[y_mask] - y[y_mask]).square().sum().item()
                validation_count += int(y_mask.sum().item())
        if not validation_count:
            raise ValueError("Validation split contains no observed targets")
        score = validation_error / validation_count
        log.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_mse": score})
        print(json.dumps(log[-1]), flush=True)
        if score < best:
            best = score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output / "best.pt")
        latest_tmp = output / "latest.tmp"
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "log": log,
                    "best": best, "best_epoch": best_epoch,
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   latest_tmp)
        latest_tmp.replace(output / "latest.pt")
        (output / "training.json").write_text(json.dumps(log, indent=2))
        if epoch + 1 - best_epoch >= args.patience:
            print(f"early stopping after {epoch + 1} epochs; best epoch {best_epoch}", flush=True)
            break
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    predictions, targets, epis, aleas, target_masks = [], [], [], [], []
    for x, mask, y, y_mask in batches("test", args.test_windows, False):
        mean, epi, alea = mc_predict(model, x, mask, graph, args.mc_passes)
        predictions.append(mean.cpu().numpy())
        targets.append(y.cpu().numpy())
        epis.append(epi.cpu().numpy())
        aleas.append(alea.cpu().numpy())
        target_masks.append(y_mask.cpu().numpy())
    pred, target, epi, alea = [np.concatenate(items) for items in (predictions, targets, epis, aleas)]
    target_observed = np.concatenate(target_masks)
    result = metrics(pred, target, epi, alea, scale[None, :, None], target_observed)
    result.update({"dataset": args.dataset, "device": str(device), "seed": args.seed,
                   "epochs_completed": len(log), "max_epochs": args.epochs, "early_stopping_patience": args.patience,
                   "mc_passes": args.mc_passes, "best_epoch": best_epoch,
                   "batch_size": args.batch_size, "learning_rate": args.lr,
                   "torch_threads": args.threads,
                   "windows": {key: int(min(len(item), getattr(args, key + "_windows") or len(item))) for key,item in splits.items()},
                   "split_timestamps": [timestamps[i] for i in boundaries], "graph_edges": int((graph > 0).sum()),
                   "graph_note": ("METR-LA official DCRNN adjacency, threshold >0.1" if (ROOT / "METR-LA/data/metr-la/adj_mx.pkl").exists() else "METR-LA geographic 8-nearest-neighbor proxy") if args.dataset == "METR-LA" else "PEMS-BAY supplied adjacency, threshold >0.1",
                   "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
                   "no_graph": args.no_graph})
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    (output / "training.json").write_text(json.dumps(log, indent=2))
    np.savez_compressed(output / "predictions.npz", mean=pred, target=target, epistemic=epi,
                        aleatoric=alea, target_observed=target_observed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-windows", type=int)
    parser.add_argument("--val-windows", type=int)
    parser.add_argument("--test-windows", type=int)
    parser.add_argument("--train-ratio", type=float, default=.7)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--no-graph", action="store_true", help="matched graph-aggregation ablation; this model has no diffusion")
    parser.add_argument("--resume", action="store_true", help="resume from the output directory's latest.pt")
    run(parser.parse_args())
