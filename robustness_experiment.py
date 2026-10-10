"""Matched test-time sensor-failure evaluation for corrected TrafficModel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from corrected_experiment import TrafficModel, load_data, prepare
from paper_model import PaperUAGNN, enable_mc_dropout, graph_edges, transitions


def corrupt(x, observed, adjacency, kind, rate, rng):
    """Artificial failure is separate from original missingness; targets never change."""
    batch, sensors, history = x.shape
    dropped = np.zeros_like(observed, dtype=bool)
    count = max(1, round(rate*sensors))
    if kind == "random":
        dropped = rng.random(x.shape) < rate
    elif kind == "persistent":
        # One failed set of physical sensors remains offline for every window.
        dropped[:, rng.choice(sensors, count, replace=False), :] = True
    elif kind == "clustered":
        connected = (adjacency > .1) | (adjacency.T > .1)
        seeds = np.flatnonzero(connected.sum(1) > 0)
        for b in range(batch):
            seed = rng.choice(seeds)
            # Graph proximity from outgoing adjacency; repeat BFS until size met.
            group, frontier = {int(seed)}, [int(seed)]
            while len(group) < count and frontier:
                node = frontier.pop(0)
                for neighbor in np.flatnonzero(connected[node]):
                    if int(neighbor) not in group:
                        group.add(int(neighbor))
                        frontier.append(int(neighbor))
                    if len(group) >= count:
                        break
            if len(group) < count:
                remaining = np.array(sorted(set(range(sensors))-group))
                group.update(rng.choice(remaining, count-len(group), replace=False).tolist())
            dropped[b, list(group), :] = True
    elif kind == "temporal_block":
        length = max(1, round(rate*history))
        for b in range(batch):
            start = rng.integers(history-length+1)
            dropped[b, :, start:start+length] = True
    else:
        raise ValueError(kind)
    # Replace failed readings with train-fitted normalized zero and set mask false.
    return np.where(dropped, 0, x), observed & ~dropped, int((dropped & observed).sum())


def forward_fill_input(values, observed):
    filled = values.copy()
    previous = np.zeros(values.shape[:2], dtype=values.dtype)
    for step in range(values.shape[2]):
        previous = np.where(observed[:, :, step], values[:, :, step], previous)
        filled[:, :, step] = previous
    return filled


def split_ratios(config):
    """Use the source run's target-period boundaries for failure tests."""
    return (config.get("train_ratio", .7),
            config.get("val_ratio", .1 if config.get("split") == "70/10/20" else .15))


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.inference_seed)
    folder = Path(args.run)
    config = json.loads((folder / "metrics.json").read_text())
    dataset = config["dataset"]
    raw, timestamps, adjacency = load_data(dataset)
    train_ratio, val_ratio = split_ratios(config)
    values, observed, _, scale, splits, _ = prepare(raw, train_ratio=train_ratio,
                                                     val_ratio=val_ratio)
    count = min(args.windows, len(splits["test"]))
    indices = splits["test"][np.linspace(0, len(splits["test"])-1, count, dtype=int)]
    x = np.stack([values[i:i+12].T for i in indices])
    input_mask = np.stack([observed[i:i+12].T for i in indices])
    y = np.stack([values[i+12:i+24].T for i in indices])
    target_mask = np.stack([observed[i+12:i+24].T for i in indices])
    paper = config.get("architecture") == "paper_interpretation_v1"
    graph = torch.tensor(np.where(adjacency > .1, adjacency, 0), dtype=torch.float32)
    if paper:
        forward, reverse = transitions(graph)
        edges = graph_edges(graph)
        times = pd.to_datetime(timestamps)
        slots = (times.hour*12+times.minute//5).to_numpy(dtype=np.int64)
        weekdays = times.dayofweek.to_numpy(dtype=np.int64)
        p = torch.from_numpy(np.stack([slots[i:i+12] for i in indices]))
        d = torch.from_numpy(np.stack([weekdays[i:i+12] for i in indices]))
        model = PaperUAGNN(hidden=config["hidden"], use_diffusion=not config["no_diffusion"])
    else:
        graph = (graph > 0).float()
        graph.fill_diagonal_(1)
        graph /= graph.sum(1, keepdim=True).clamp_min(1)
        if config.get("no_graph"):
            graph = torch.eye(graph.shape[0])
        model = TrafficModel()
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    result = []

    def evaluate(xx, mm):
        # Pair model dropout draws across clean and corrupted inputs.
        torch.manual_seed(args.inference_seed)
        errors, sigmas, epi_values, alea_values = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(xx), args.batch_size):
                xx_batch = torch.from_numpy(xx[start:start+args.batch_size])
                mm_batch = torch.from_numpy(mm[start:start+args.batch_size])
                def predict():
                    if paper:
                        return model(xx_batch.transpose(1, 2), mm_batch.transpose(1, 2),
                                     p[start:start+args.batch_size], d[start:start+args.batch_size],
                                     forward, reverse, edges)
                    return model(xx_batch, mm_batch, graph)
                if args.mc_passes > 1:
                    enable_mc_dropout(model)
                    samples = [predict() for _ in range(args.mc_passes)]
                    means = torch.stack([item[0] for item in samples])
                    mean = means.mean(0)
                    epi = means.var(0, unbiased=False)
                    alea = torch.stack([item[1] for item in samples]).mean(0)
                else:
                    model.eval()
                    mean, alea = predict()
                    epi = torch.zeros_like(alea)
                selected = target_mask[start:start+args.batch_size]
                err = (mean.numpy()-y[start:start+args.batch_size])*scale[None, :, None]
                sigma = np.sqrt((epi+alea).numpy())*scale[None, :, None]
                errors.append(err[selected])
                sigmas.append(sigma[selected])
                epi_values.append((epi.numpy()*scale[None, :, None]**2)[selected])
                alea_values.append((alea.numpy()*scale[None, :, None]**2)[selected])
        errors = np.concatenate(errors)
        sigmas = np.concatenate(sigmas)
        return {"mae_mph": float(np.abs(errors).mean()),
                "rmse_mph": float(np.sqrt(np.square(errors).mean())),
                "coverage_95": float((np.abs(errors) <= 1.95996398454*sigmas).mean()),
                "width_95_mph": float((2*1.95996398454*sigmas).mean()),
                "epistemic_variance_mph2": float(np.concatenate(epi_values).mean()),
                "aleatoric_variance_mph2": float(np.concatenate(alea_values).mean())}

    def persistence(xx):
        error = (xx[:, :, -1, None]-y)*scale[None, :, None]
        valid = error[target_mask]
        return float(np.abs(valid).mean()), float(np.sqrt(np.square(valid).mean()))

    clean = evaluate(x, input_mask)
    clean_mae = clean["mae_mph"]
    baseline_mae, baseline_rmse = persistence(x)
    result.append(dict(kind="clean", rate=0, corruption_seed=None, **clean,
                       persistence_mae_mph=baseline_mae,
                       persistence_rmse_mph=baseline_rmse, newly_hidden_observations=0))
    if args.clean_only:
        output = {"dataset": dataset, "source_run": str(folder),
                  "sampled_test_windows": len(indices), "mc_passes": args.mc_passes,
                  "inference_seed": args.inference_seed,
                  "observed_targets": int(target_mask.sum()), **clean,
                  "persistence_mae_mph": baseline_mae,
                  "persistence_rmse_mph": baseline_rmse}
        (folder / ("no_mc_metrics.json" if args.mc_passes == 1 else "clean_mc_metrics.json")).write_text(
            json.dumps(output, indent=2))
        print(json.dumps(output))
        return
    for kind in ("random", "persistent", "clustered", "temporal_block"):
        for rate in args.rates:
            for seed in args.seeds:
                xx, mm, hidden = corrupt(x, input_mask, adjacency, kind, rate, np.random.default_rng(seed))
                for imputation, input_values in (("train_median", xx),
                                                 ("causal_forward_fill", forward_fill_input(xx, mm))):
                    scores = evaluate(input_values.astype(np.float32), mm)
                    baseline_mae, baseline_rmse = persistence(input_values)
                    result.append(dict(kind=kind, rate=rate, corruption_seed=seed,
                                       imputation=imputation, **scores,
                                       persistence_mae_mph=baseline_mae,
                                       persistence_rmse_mph=baseline_rmse,
                                       mae_ratio=scores["mae_mph"]/clean_mae,
                                       newly_hidden_observations=hidden))
                    print(json.dumps(result[-1]), flush=True)
    summary = []
    for kind in ("random", "persistent", "clustered", "temporal_block"):
        for rate in args.rates:
            for imputation in ("train_median", "causal_forward_fill"):
                group = [row for row in result if row["kind"] == kind and row["rate"] == rate
                         and row.get("imputation") == imputation]
                maes = [row["mae_mph"] for row in group]
                summary.append({"kind": kind, "rate": rate, "imputation": imputation,
                                "mae_mean_mph": float(np.mean(maes)),
                                "mae_sd_mph": float(np.std(maes, ddof=1)) if len(maes)>1 else None,
                                "mae_ratio_to_clean": float(np.mean(maes)/clean_mae),
                                "coverage_95_mean": float(np.mean([row["coverage_95"] for row in group])),
                                "width_95_mean_mph": float(np.mean([row["width_95_mph"] for row in group])),
                                "epistemic_variance_mean_mph2": float(np.mean([row["epistemic_variance_mph2"] for row in group])),
                                "aleatoric_variance_mean_mph2": float(np.mean([row["aleatoric_variance_mph2"] for row in group]))})
    artifact = {"source_run": str(folder), "dataset": dataset,
                "mc_passes": args.mc_passes, "inference_seed": args.inference_seed,
                "sampled_test_windows": len(indices),
                "original_missing_input_fraction": float((~input_mask).mean()),
                "test_targets": int(target_mask.sum()), "rows": result, "summary": summary}
    (folder / "robustness.json").write_text(json.dumps(artifact, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in ("random", "persistent", "clustered", "temporal_block"):
        subset = [row for row in result if row["kind"] == kind and row.get("imputation") == "train_median"]
        groups = [[row["mae_mph"] for row in subset if row["rate"] == rate]
                  for rate in args.rates]
        means = [clean_mae] + [float(np.mean(group)) for group in groups]
        sds = [0.] + [float(np.std(group, ddof=1)) if len(group)>1 else 0. for group in groups]
        plt.errorbar([0] + args.rates, means, yerr=sds, marker="o", capsize=2, label=kind)
    plt.xlabel("Artificial failure rate")
    plt.ylabel("MAE (mph)")
    plt.title(f"{dataset}: {len(indices)} sampled test windows")
    plt.grid(alpha=.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "robustness.png", dpi=160)
    plt.close()
    for kind in ("random", "persistent", "clustered", "temporal_block"):
        subset = [row for row in result if row["kind"] == kind and row.get("imputation") == "train_median"]
        widths = [clean["width_95_mph"]] + [float(np.mean([row["width_95_mph"] for row in subset
                                                           if row["rate"] == rate])) for rate in args.rates]
        plt.plot([0] + args.rates, widths, marker="o", label=kind)
    plt.xlabel("Artificial failure rate")
    plt.ylabel("Mean 95% interval width (mph)")
    plt.title(f"{dataset}: {len(indices)} sampled test windows, {args.mc_passes} MC pass(es)")
    plt.grid(alpha=.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "robustness_width.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mc-passes", type=int, default=1)
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--rates", type=float, nargs="+", default=[.1, .2, .3, .4, .5])
    parser.add_argument("--clean-only", action="store_true")
    run(parser.parse_args())
