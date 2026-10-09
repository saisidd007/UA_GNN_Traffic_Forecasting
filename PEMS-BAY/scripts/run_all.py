"""
Run-all wrapper for the traffic-flow-gnn experiments.

This script performs the following steps (in order):
 - sanity-check Python environment (torch availability)
 - (optional) train a 50-epoch model if no best checkpoint exists
 - run MC-dropout inference (per-horizon metrics)
 - run sensor-dropout robustness experiment
 - run multiple-runs analysis
 - generate calibration/sharpness plots

The script stops on first fatal error and prints actionable instructions.
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DATA = ROOT / 'data'

def check_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def run(cmd, cwd=None):
    print(f"\n$ {cmd}")
    rc = subprocess.call(cmd, shell=True, cwd=cwd)
    if rc != 0:
        print(f"\nERROR: Command failed with exit code {rc}: {cmd}")
        return False
    return True

def main():
    print("Run-all: traffic-flow-gnn pipeline")
    print(f"Workspace: {ROOT}")

    # 1) Check torch availability
    print('\n1) Checking for PyTorch...')
    if not check_torch():
        skip = os.environ.get('SKIP_TORCH', '')
        if str(skip) == '1':
            print('\nWARNING: PyTorch is not importable; SKIP_TORCH=1 set — running synthetic fallback.')
            # Run synthetic generation + plotting scripts so downstream results exist for inspection
            run('python scripts/generate_synthetic_metrics.py', cwd=str(ROOT))
            run('python scripts/generate_synthetic_dropout.py', cwd=str(ROOT))
            run('python scripts/generate_synthetic_runs.py', cwd=str(ROOT))
            # calibration and plotting
            run('python scripts/calibration_analysis.py', cwd=str(ROOT))
            print('\nSynthetic fallback completed. Exiting run-all.')
            sys.exit(0)
        else:
            print('\nERROR: PyTorch is not importable in this Python environment.')
            print('Please activate your training virtualenv and install requirements:')
            print('\n    .venv\\Scripts\\Activate.ps1')
            print('    pip install -r requirements.txt')
            print('\nOr install CPU-only wheel:')
            print('    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
            sys.exit(2)

    # 2) Optionally train (if no best model)
    best_ckpt = RESULTS / 'enhanced_best_model.pt'
    if not best_ckpt.exists():
        print('\n2) No best checkpoint found - starting training (50 epochs)')
        if not run('python enhanced_train.py --epochs 50', cwd=str(ROOT)):
            print('\nTraining failed; aborting run-all.')
            sys.exit(3)
    else:
        print('\n2) Found existing checkpoint:', best_ckpt)

    # 3) Run inference and per-horizon metrics
    print('\n3) Running MC-dropout inference & per-horizon metrics')
    if not run('python notebooks/analysis_50epoch.py', cwd=str(ROOT)):
        print('\nInference step failed; aborting run-all.')
        sys.exit(4)

    # 4) Sensor dropout experiment
    print('\n4) Running sensor-dropout robustness experiment')
    if not run('python scripts/sensor_dropout_experiment.py', cwd=str(ROOT)):
        print('\nSensor-dropout experiment failed; aborting run-all.')
        sys.exit(5)

    # 5) Multiple runs analysis
    print('\n5) Running multiple-runs analysis')
    if not run('python scripts/multiple_runs_analysis.py', cwd=str(ROOT)):
        print('\nMultiple-runs analysis failed; aborting run-all.')
        sys.exit(6)

    # 6) Calibration & plotting
    print('\n6) Generating calibration and reliability plots')
    if not run('python scripts/calibration_analysis.py', cwd=str(ROOT)):
        print('\nCalibration plotting failed; you can run the script manually later.')

    print('\nRun-all completed. Check the `results/` folder for generated CSVs/PNGs.')

if __name__ == "__main__":
    main()
