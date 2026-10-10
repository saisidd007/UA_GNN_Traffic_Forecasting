"""Independent saved-prediction audit and calibration plot in physical units."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from corrected_experiment import load_data, prepare


def analyze(folder):
    folder = Path(folder)
    config = json.loads((folder / "metrics.json").read_text())
    with np.load(folder / "predictions.npz") as saved:
        mean, target = saved["mean"], saved["target"]
        epistemic, aleatoric = saved["epistemic"], saved["aleatoric"]
        valid = saved["target_observed"].astype(bool)
    raw, _, _ = load_data(config["dataset"])
    values, _, center, scale, splits, _ = prepare(
        raw, train_ratio=config.get("train_ratio", .7),
        val_ratio=config.get("val_ratio", .1 if config.get("split") == "70/10/20" else .15))
    error = (mean-target)*scale[None, :, None]
    physical_target = (target*scale[None, :, None]+center[None, :, None])[valid]
    physical_mean = (mean*scale[None, :, None]+center[None, :, None])[valid]
    centered_target = physical_target-physical_target.mean(dtype=np.float64)
    centered_mean = physical_mean-physical_mean.mean(dtype=np.float64)
    sigma = np.sqrt(np.maximum(epistemic+aleatoric, 1e-12))*scale[None, :, None]
    levels = np.arange(.1, 1., .05)
    from scipy.special import ndtri
    coverage = [float((np.abs(error[valid]) <= ndtri((1+level)/2)*sigma[valid]).mean()) for level in levels]
    # Closed-form CRPS of N(mean, sigma^2), evaluated only where targets exist.
    z = error[valid]/sigma[valid]
    crps = sigma[valid] * (z*(2*ndtr(z)-1) + 2*np.exp(-z*z/2)/math.sqrt(2*math.pi) - 1/math.sqrt(math.pi))
    width95 = 2*1.95996398454*sigma[valid]
    miss95 = np.maximum(np.abs(error[valid])-1.95996398454*sigma[valid], 0)
    normalized_error = (mean-target)[valid]
    result = {"normalized_mae": float(np.abs(normalized_error).mean()),
              "normalized_rmse": float(np.sqrt(np.square(normalized_error).mean())),
              "coverage_ece_10_to_95": float(np.mean(np.abs(levels-np.array(coverage)))),
              "r2": float(1-np.sum(np.square(physical_mean-physical_target), dtype=np.float64)/
                          np.sum(np.square(centered_target), dtype=np.float64)),
              "pearson_r": float(np.sum(centered_mean*centered_target, dtype=np.float64)/
                                 np.sqrt(np.sum(centered_mean**2, dtype=np.float64)*
                                         np.sum(centered_target**2, dtype=np.float64))),
              "gaussian_crps_mph": float(crps.mean()),
              "interval_score_95_mph": float((width95 + (2/.05)*miss95).mean()),
              "epistemic_variance_mph2": float((epistemic*scale[None, :, None]**2)[valid].mean()),
              "aleatoric_variance_mph2": float((aleatoric*scale[None, :, None]**2)[valid].mean()),
              "coverage_95_without_epistemic_component": float((np.abs(error[valid]) <=
                   1.95996398454*np.sqrt(np.maximum(aleatoric, 0))[valid]*
                   np.broadcast_to(scale[None, :, None], mean.shape)[valid]).mean()),
              "coverage_95_without_aleatoric_component": float((np.abs(error[valid]) <=
                   1.95996398454*np.sqrt(np.maximum(epistemic, 0))[valid]*
                   np.broadcast_to(scale[None, :, None], mean.shape)[valid]).mean()),
              "coverage_by_horizon_95": [], "width_by_horizon_95_mph": [],
              "mae_by_horizon_mph": [], "rmse_by_horizon_mph": []}
    for step in range(mean.shape[-1]):
        step_error, step_sigma = error[:, :, step][valid[:, :, step]], sigma[:, :, step][valid[:, :, step]]
        result["coverage_by_horizon_95"].append(float((np.abs(step_error) <= 1.95996398454*step_sigma).mean()))
        result["width_by_horizon_95_mph"].append(float((2*1.95996398454*step_sigma).mean()))
        result["mae_by_horizon_mph"].append(float(np.abs(step_error).mean()))
        result["rmse_by_horizon_mph"].append(float(np.sqrt(np.square(step_error).mean())))
    if len(mean) == len(splits["test"]):
        # Paired whole-day block bootstrap; overlapping forecast windows are
        # strongly dependent, so individual-window resampling is misleading.
        last = values[splits["test"]+11]
        persistence_error = (last[:, :, None]-target)*scale[None, :, None]
        model_sums = (np.abs(error)*valid).sum(axis=(1, 2))
        base_sums = (np.abs(persistence_error)*valid).sum(axis=(1, 2))
        counts = valid.sum(axis=(1, 2))
        block_starts = np.arange(0, len(mean), 288)
        model_blocks = np.add.reduceat(model_sums, block_starts)
        base_blocks = np.add.reduceat(base_sums, block_starts)
        count_blocks = np.add.reduceat(counts, block_starts)
        rng = np.random.default_rng(123)
        draws = rng.integers(len(block_starts), size=(1000, len(block_starts)))
        differences = ((model_blocks[draws].sum(1)-base_blocks[draws].sum(1)) /
                       count_blocks[draws].sum(1))
        result["model_minus_persistence_mae_mph"] = float((model_sums.sum()-base_sums.sum())/counts.sum())
        result["paired_daily_block_bootstrap_95ci_mph"] = np.percentile(differences, [2.5, 97.5]).tolist()
    (folder / "prediction_audit.json").write_text(json.dumps(result, indent=2))
    plt.figure(figsize=(5, 5))
    plt.plot(levels, coverage, "o-", label="Empirical")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Nominal Gaussian interval coverage")
    plt.ylabel("Empirical coverage")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "calibration.png", dpi=160)
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    steps = np.arange(1, mean.shape[-1]+1)*5
    axes[0].plot(steps, result["mae_by_horizon_mph"], marker="o")
    axes[0].set(xlabel="Forecast horizon (minutes)", ylabel="MAE (mph)")
    axes[1].plot(steps, result["coverage_by_horizon_95"], marker="o")
    axes[1].axhline(.95, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Forecast horizon (minutes)", ylabel="95% coverage", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "horizon_metrics.png", dpi=160)
    plt.close(fig)
    history_file = folder / "training.json"
    if history_file.exists():
        history = json.loads(history_file.read_text())
        plt.figure(figsize=(6, 3.5))
        plt.plot([row["epoch"] for row in history], [row["val_mse"] for row in history], marker=".")
        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE (normalized)")
        plt.grid(alpha=.2)
        plt.tight_layout()
        plt.savefig(folder / "validation_curve.png", dpi=160)
        plt.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    print(json.dumps(analyze(parser.parse_args().folder), indent=2))
