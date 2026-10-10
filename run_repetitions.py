"""Run a frozen configuration across seeds and aggregate final test metrics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from corrected_experiment import persistence_baseline


def main(args):
    if (args.no_diffusion or args.no_dropout or args.no_aleatoric) and not args.paper:
        raise ValueError("Paper ablations require --paper")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    rows = []
    for seed in args.seeds:
        folder = root / f"seed_{seed}"
        command = [sys.executable, "paper_experiment.py" if args.paper else "corrected_experiment.py",
                   "--dataset", args.dataset, "--output", str(folder), "--seed", str(seed),
                   "--epochs", str(args.epochs), "--train-windows", str(args.train_windows),
                   "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                   "--mc-passes", str(args.mc_passes)]
        if args.paper:
            command.extend(("--hidden", str(args.hidden)))
            if args.no_diffusion:
                command.append("--no-diffusion")
            if args.no_dropout:
                command.append("--no-dropout")
            if args.no_aleatoric:
                command.append("--no-aleatoric")
        else:
            command.extend(("--val-ratio", ".1"))
            if args.no_graph:
                command.append("--no-graph")
        if args.val_windows:
            command.extend(("--val-windows", str(args.val_windows)))
        if args.test_windows:
            command.extend(("--test-windows", str(args.test_windows)))
        if args.resume and (folder / "metrics.json").exists():
            print(f"Using completed run {folder}", flush=True)
        else:
            if args.resume and folder.exists():
                if not (folder / "latest.pt").exists():
                    raise FileNotFoundError(f"Cannot resume {folder}: latest.pt is missing")
                command.append("--resume")
            subprocess.run(command, check=True)
        completed = json.loads((folder / "metrics.json").read_text())
        expected = {"dataset": args.dataset, "seed": seed,
                    "batch_size": args.batch_size, "mc_passes": args.mc_passes,
                    "max_epochs": args.epochs}
        if args.paper:
            expected.update(train_windows_per_epoch=args.train_windows,
                            hidden=args.hidden, no_diffusion=args.no_diffusion,
                            no_dropout=args.no_dropout, no_aleatoric=args.no_aleatoric)
            if args.val_windows is not None:
                expected["val_windows"] = args.val_windows
            if args.test_windows is not None:
                expected["test_windows"] = args.test_windows
        elif completed.get("windows", {}).get("train") != args.train_windows or completed.get("no_graph") != args.no_graph:
            raise ValueError(f"{folder}: small-model training configuration differs")
        elif (args.val_windows is not None and completed["windows"]["val"] != args.val_windows) or (
                args.test_windows is not None and completed["windows"]["test"] != args.test_windows):
            raise ValueError(f"{folder}: small-model evaluation window sample differs")
        for key, value in expected.items():
            if completed.get(key) != value:
                raise ValueError(f"{folder}: {key}={completed.get(key)!r}, expected {value!r}")
        rows.append(completed)
    summary = {"dataset": args.dataset, "paper_interpretation": args.paper,
               "no_diffusion": args.no_diffusion, "no_dropout": args.no_dropout,
               "no_aleatoric": args.no_aleatoric,
               "seeds": args.seeds, "training_windows_per_epoch": args.train_windows,
               "validation_windows": args.val_windows,
               "test_windows": args.test_windows,
               "batch_size": args.batch_size, "epochs_max": args.epochs,
               "best_epochs": [row["best_epoch"] for row in rows],
               "results": {}}
    summary["matched_persistence"] = persistence_baseline(
        args.dataset, val_ratio=.1, test_windows=args.test_windows)
    for key in ("mae_mph", "rmse_mph", "coverage_50", "coverage_80", "coverage_90", "coverage_95"):
        values = [row[key] for row in rows]
        summary["results"][key] = {"mean": float(np.mean(values)),
                                   "sd": float(np.std(values, ddof=1)) if len(values)>1 else None,
                                   "each": values}
    for key in ("mae_by_horizon_mph",):
        values = np.array([row[key] for row in rows])
        summary["results"][key] = {"mean": values.mean(0).tolist(),
                                   "sd": values.std(0, ddof=1).tolist() if len(values)>1 else None}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29, 42, 71])
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--no-diffusion", action="store_true")
    parser.add_argument("--no-dropout", action="store_true")
    parser.add_argument("--no-aleatoric", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-windows", type=int, default=8192)
    parser.add_argument("--val-windows", type=int)
    parser.add_argument("--test-windows", type=int)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mc-passes", type=int, default=10)
    main(parser.parse_args())
