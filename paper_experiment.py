"""CPU/GPU rerun of an explicitly specified interpretation of the paper model."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from corrected_experiment import load_data, metrics, persistence_baseline, prepare
from paper_model import PaperUAGNN, enable_mc_dropout, graph_edges, transitions


def manuscript_loss(mean, variance, target, observed, epoch, no_aleatoric=False):
    """Eq. (total loss): linearly replace MSE with Gaussian NLL in 10 epochs."""
    squared = (mean - target).square()
    if no_aleatoric:
        return squared[observed].mean()
    weight_mse = max(0.0, 1.0 - epoch / 10.0)
    nll = 0.5 * (variance.log() + squared / variance)
    return (weight_mse * squared + (1.0 - weight_mse) * nll)[observed].mean()


def run(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw, timestamps, adjacency = load_data(args.dataset)
    values, observed, center, scale, splits, boundaries = prepare(raw, val_ratio=.1)
    times = pd.to_datetime(timestamps)
    tod = (times.hour * 12 + times.minute // 5).to_numpy(dtype=np.int64)
    dow = times.dayofweek.to_numpy(dtype=np.int64)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = torch.tensor(np.where(adjacency > .1, adjacency, 0), dtype=torch.float32, device=device)
    forward, reverse = transitions(graph)
    edges = graph_edges(graph)
    model = PaperUAGNN(hidden=args.hidden, dropout=0.0 if args.no_dropout else .3,
                       use_diffusion=not args.no_diffusion).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10)
    parameter_count = sum(p.numel() for name, p in model.named_parameters()
                          if (not args.no_diffusion or not name.startswith("diffusion."))
                          and (not args.no_aleatoric or not name.startswith("variance.")))

    def batches(split, limit, shuffle):
        indices = splits[split].copy()
        if shuffle:
            np.random.shuffle(indices)
        if limit and limit < len(indices):
            indices = indices[:limit] if shuffle else indices[np.linspace(0, len(indices)-1, limit, dtype=int)]
        for chunk in np.array_split(indices, math.ceil(len(indices)/args.batch_size)):
            x = np.stack([values[i:i+12] for i in chunk])
            mask = np.stack([observed[i:i+12] for i in chunk])
            y = np.stack([values[i+12:i+24].T for i in chunk])
            y_mask = np.stack([observed[i+12:i+24].T for i in chunk])
            p = np.stack([tod[i:i+12] for i in chunk])
            d = np.stack([dow[i:i+12] for i in chunk])
            yield (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device),
                   torch.from_numpy(y).to(device), torch.from_numpy(y_mask).to(device),
                   torch.from_numpy(p).to(device), torch.from_numpy(d).to(device))

    best, best_epoch, log, start_epoch = float("inf"), 0, [], 0
    if args.resume:
        state = torch.load(output / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, best_epoch, log, start_epoch = (state[key] for key in
                                            ("best", "best_epoch", "log", "epoch"))
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_random_state"])
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for x, mask, y, target_mask, p, d in batches("train", args.train_windows, True):
            if not target_mask.any():
                continue
            # Randomly mask whole sensors for the full historical window.
            dropped = torch.rand(x.shape[0], 1, x.shape[2], device=device) < .1
            augmented_mask = mask & ~dropped
            augmented = x.masked_fill(dropped, 0)
            optimizer.zero_grad()
            mean, variance = model(augmented, augmented_mask, p, d, forward, reverse, edges)
            loss = manuscript_loss(mean, variance, y, target_mask, epoch, args.no_aleatoric)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError("Training sample contains no observed targets")
        model.eval()
        error_sum, count = 0., 0
        with torch.no_grad():
            for x, mask, y, target_mask, p, d in batches("val", args.val_windows, False):
                mean, _ = model(x, mask, p, d, forward, reverse, edges)
                error_sum += (mean-y).square()[target_mask].sum().item()
                count += int(target_mask.sum())
        if not count:
            raise ValueError("Validation split contains no observed targets")
        score = error_sum/count
        scheduler.step(score)
        row = {"epoch": epoch+1, "train_loss": float(np.mean(losses)), "val_mse": score,
               "seconds": round(time.monotonic()-started, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best, best_epoch = score, epoch+1
            torch.save(model.state_dict(), output / "best.pt")
        latest_tmp = output / "latest.tmp"
        torch.save({"epoch": epoch+1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "best_epoch": best_epoch, "log": log,
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   latest_tmp)
        latest_tmp.replace(output / "latest.pt")
        (output / "training.json").write_text(json.dumps(log, indent=2))
        if epoch+1-best_epoch >= args.patience:
            break

    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    model.eval()
    predictions, targets, epis, aleas, masks = [], [], [], [], []
    for x, mask, y, target_mask, p, d in batches("test", args.test_windows, False):
        enable_mc_dropout(model)
        with torch.no_grad():
            # A model trained with zero dropout is deterministic in eval mode.
            # Repeating the same forward pass cannot add epistemic variance.
            effective_passes = 1 if args.no_dropout else args.mc_passes
            samples = [model(x, mask, p, d, forward, reverse, edges)
                       for _ in range(effective_passes)]
        means = torch.stack([sample[0] for sample in samples])
        variances = torch.stack([sample[1] for sample in samples])
        predictions.append(means.mean(0).cpu().numpy())
        epis.append(means.var(0, unbiased=False).cpu().numpy())
        aleas.append((torch.zeros_like(variances[0]) if args.no_aleatoric else
                      variances.mean(0)).cpu().numpy())
        targets.append(y.cpu().numpy())
        masks.append(target_mask.cpu().numpy())
    pred, target, epi, alea, target_mask = [np.concatenate(items) for items in
                                           (predictions, targets, epis, aleas, masks)]
    result = metrics(pred, target, epi, alea, scale[None, :, None], target_mask)
    result.update(dataset=args.dataset, seed=args.seed, architecture="paper_interpretation_v1",
                  parameters=parameter_count, hidden=args.hidden, no_diffusion=args.no_diffusion,
                  no_dropout=args.no_dropout, no_aleatoric=args.no_aleatoric,
                  device=str(device), torch_threads=args.threads,
                  epochs_completed=len(log), max_epochs=args.epochs,
                  early_stopping_patience=args.patience, best_epoch=best_epoch,
                  batch_size=args.batch_size, learning_rate=.001,
                  train_sensor_mask_rate=.1, loss_nll_ramp_epochs=10,
                  training_loss=("mse_only_no_aleatoric_v2" if args.no_aleatoric else
                                 "linear_mse_to_nll_v2"),
                  scheduler_factor=.5,
                  graph_threshold=.1,
                  train_windows_per_epoch=args.train_windows,
                  val_windows=int(min(len(splits["val"]), args.val_windows or len(splits["val"]))),
                  test_windows=int(min(len(splits["test"]), args.test_windows or len(splits["test"]))),
                  mc_passes=args.mc_passes,
                  effective_mc_passes=1 if args.no_dropout else args.mc_passes,
                  train_ratio=.7, val_ratio=.1,
                  split="70/10/20", split_timestamps=[timestamps[i] for i in boundaries],
                  graph_edges_without_self_loops=int((graph>0).sum()-int((graph.diag()>0).sum())))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    baseline = persistence_baseline(args.dataset, train_ratio=.7, val_ratio=.1,
                                    test_windows=args.test_windows)
    (output / "persistence_baseline.json").write_text(json.dumps(baseline, indent=2))
    np.savez_compressed(output / "predictions.npz", mean=pred, target=target,
                        epistemic=epi, aleatoric=alea, target_observed=target_mask)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-windows", type=int, default=8192)
    parser.add_argument("--val-windows", type=int)
    parser.add_argument("--test-windows", type=int)
    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument("--no-diffusion", action="store_true")
    parser.add_argument("--no-dropout", action="store_true")
    parser.add_argument("--no-aleatoric", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
