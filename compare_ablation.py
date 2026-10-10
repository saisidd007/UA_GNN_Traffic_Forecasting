"""Independently compare matched saved forecast arrays in physical units."""

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_experiment import load_data, prepare


def compare(reference, ablation, block_windows=14, resamples=1000):
    reference, ablation = Path(reference), Path(ablation)
    first = json.loads((reference / "metrics.json").read_text())
    second = json.loads((ablation / "metrics.json").read_text())
    losses = (first.get("training_loss"), second.get("training_loss"))
    valid_losses = (losses[0] == losses[1] and losses[0] in
                    ("linear_mse_to_nll_v2", "mse_only_no_aleatoric_v2"))
    if losses == ("linear_mse_to_nll_v2", "mse_only_no_aleatoric_v2") and second.get("no_aleatoric"):
        valid_losses = True
    if not valid_losses:
        raise ValueError("Runs have incompatible or undeclared training-loss versions")
    for key in ("dataset", "seed", "split", "test_windows", "train_windows_per_epoch",
                "val_windows", "mc_passes", "batch_size", "hidden", "max_epochs",
                "early_stopping_patience", "scheduler_factor", "train_sensor_mask_rate",
                "loss_nll_ramp_epochs", "graph_threshold"):
        if first[key] != second[key]:
            raise ValueError(f"Unmatched experiment field {key}")
    with np.load(reference / "predictions.npz") as saved:
        mean_ref, target_ref, mask_ref = saved["mean"], saved["target"], saved["target_observed"]
    with np.load(ablation / "predictions.npz") as saved:
        mean_abl, target_abl, mask_abl = saved["mean"], saved["target"], saved["target_observed"]
    np.testing.assert_array_equal(target_ref, target_abl)
    np.testing.assert_array_equal(mask_ref, mask_abl)
    raw, _, _ = load_data(first["dataset"])
    _, _, _, scale, _, _ = prepare(raw, train_ratio=.7, val_ratio=.1)
    physical_ref = np.abs((mean_ref-target_ref)*scale[None, :, None])
    physical_abl = np.abs((mean_abl-target_ref)*scale[None, :, None])
    per_ref = (physical_ref*mask_ref).sum(axis=(1, 2), dtype=np.float64)
    per_abl = (physical_abl*mask_ref).sum(axis=(1, 2), dtype=np.float64)
    counts = mask_ref.sum(axis=(1, 2))
    starts = np.arange(0, len(counts), block_windows)
    ref_blocks = np.add.reduceat(per_ref, starts)
    abl_blocks = np.add.reduceat(per_abl, starts)
    count_blocks = np.add.reduceat(counts, starts)
    rng = np.random.default_rng(123)
    draws = rng.integers(len(starts), size=(resamples, len(starts)))
    differences = ((abl_blocks[draws].sum(1)-ref_blocks[draws].sum(1)) /
                   count_blocks[draws].sum(1))
    ref_mae, abl_mae = per_ref.sum()/counts.sum(), per_abl.sum()/counts.sum()
    result = {"reference_run": str(reference), "ablation_run": str(ablation),
              "reference_training_loss": losses[0], "ablation_training_loss": losses[1],
              "reference_epochs_completed": first["epochs_completed"],
              "ablation_epochs_completed": second["epochs_completed"],
              "matched_windows": len(counts), "matched_observed_targets": int(counts.sum()),
              "reference_mae_mph": float(ref_mae), "ablation_mae_mph": float(abl_mae),
              "ablation_minus_reference_mae_mph": float(abl_mae-ref_mae),
              "relative_mae_increase_pct": float(100*(abl_mae/ref_mae-1)),
              "block_windows": block_windows, "bootstrap_resamples": resamples,
              "descriptive_95ci_mph": np.percentile(differences, [2.5, 97.5]).tolist()}
    (ablation / "matched_comparison.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("ablation")
    parser.add_argument("--block-windows", type=int, default=14)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.ablation, args.block_windows), indent=2))
