"""Compare Gaussian MC estimates on identical windows and dropout RNG streams."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(args):
    folder = Path(args.run)
    rows = []
    for passes in args.passes:
        command = [sys.executable, "robustness_experiment.py", str(folder),
                   "--windows", str(args.windows), "--batch-size", str(args.batch_size),
                   "--threads", str(args.threads), "--mc-passes", str(passes),
                   "--inference-seed", str(args.seed), "--clean-only"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        rows.append(json.loads((folder / "clean_mc_metrics.json").read_text()))
    result = {"source_run": str(folder), "test_windows": args.windows,
              "inference_seed": args.seed, "rows": rows}
    (folder / "mc_stability.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, nargs="+", default=[5, 10, 30, 50])
    main(parser.parse_args())
