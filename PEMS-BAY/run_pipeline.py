#!/usr/bin/env python
"""
Auto-run the complete training and analysis pipeline.
Trains for 50 epochs, then automatically runs all experiments and generates results.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(cmd, description, log_file=None):
    """Run a command and log output"""
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=None)
        else:
            result = subprocess.run(cmd, timeout=None)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with exit code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} raised exception: {e}")
        return False

def main():
    os.chdir(Path(__file__).parent)
    
    # Get Python executable from venv
    python_exe = r"C:\venv_traffic_flow\Scripts\python.exe"
    if not os.path.exists(python_exe):
        print(f"ERROR: Python executable not found at {python_exe}")
        sys.exit(1)
    
    print(f"\n[PIPELINE] Starting auto-run pipeline")
    print(f"Python: {python_exe}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Train for 50 epochs
    print("\n" + "="*70)
    print("[PHASE 1] TRAINING (50 EPOCHS)")
    print("="*70)
    if not run_command(
        [python_exe, "enhanced_train.py", "--epochs", "50"],
        "50-epoch training",
        log_file="results/train_50.log"
    ):
        print("Training failed, aborting pipeline")
        sys.exit(1)
    
    # Step 2: Run per-horizon inference and metrics
    print("\n" + "="*70)
    print("[PHASE 2] INFERENCE & PER-HORIZON METRICS")
    print("="*70)
    if not run_command(
        [python_exe, "notebooks/analysis_50epoch.py"],
        "Per-horizon metrics computation",
        log_file="results/analysis_50epoch.log"
    ):
        print("Warning: Analysis failed, but continuing...")
    
    # Step 3: Run sensor dropout robustness experiment
    print("\n" + "="*70)
    print("[PHASE 3] SENSOR DROPOUT ROBUSTNESS")
    print("="*70)
    if not run_command(
        [python_exe, "scripts/sensor_dropout_experiment.py"],
        "Sensor dropout robustness test",
        log_file="results/sensor_dropout.log"
    ):
        print("Warning: Sensor dropout test failed, but continuing...")
    
    # Step 4: Run multiple runs analysis
    print("\n" + "="*70)
    print("[PHASE 4] MULTIPLE RUNS STATISTICAL ANALYSIS")
    print("="*70)
    if not run_command(
        [python_exe, "scripts/multiple_runs_analysis.py"],
        "Multiple runs 95% CI analysis",
        log_file="results/multiple_runs.log"
    ):
        print("Warning: Multiple runs analysis failed, but continuing...")
    
    # Step 5: Run calibration analysis
    print("\n" + "="*70)
    print("[PHASE 5] CALIBRATION & UNCERTAINTY ANALYSIS")
    print("="*70)
    if not run_command(
        [python_exe, "scripts/calibration_analysis.py"],
        "Calibration curve and sharpness analysis",
        log_file="results/calibration.log"
    ):
        print("Warning: Calibration analysis failed, but continuing...")
    
    # Summary
    print("\n" + "="*70)
    print("[COMPLETE] PIPELINE EXECUTION FINISHED")
    print("="*70)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List output files
    print("\n[OUTPUT FILES] Generated results:")
    results_dir = Path("results")
    if results_dir.exists():
        for f in sorted(results_dir.glob("*")):
            if f.is_file() and f.suffix in ['.csv', '.png', '.pt', '.json', '.log']:
                size = f.stat().st_size / (1024*1024)  # MB
                if size > 0:
                    print(f"  ✓ {f.name:50s} ({size:.1f} MB)")
    
    print("\nPipeline complete. Check results/ directory for outputs.")

if __name__ == "__main__":
    main()
