#!/usr/bin/env python
"""
Master Analysis Pipeline
Runs all four comprehensive analyses sequentially:
1. Per-horizon metrics with MC-dropout uncertainty
2. Sensor dropout robustness analysis
3. Multiple runs statistical analysis (95% CI)
4. Calibration curves and sharpness analysis
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_analysis(script_name: str, script_path: str) -> bool:
    """Run a single analysis script and track progress"""
    print("\n" + "="*120)
    print(f"[STARTING] {script_name}")
    print("="*120)
    
    try:
        # Resolve to absolute path
        abs_script_path = Path(__file__).parent / script_path
        result = subprocess.run(
            [sys.executable, str(abs_script_path)],
            cwd=str(Path(__file__).parent),
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n[OK] {script_name} completed successfully!")
            return True
        else:
            print(f"\n[FAIL] {script_name} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] {script_name} encountered an error: {e}")
        return False


def main():
    """Main pipeline orchestrator"""
    print("\n" + "="*120)
    print("[COMPREHENSIVE ANALYSIS PIPELINE] All post-training experiments")
    print("="*120)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define analyses in order
    analyses = [
        ("Per-Horizon Metrics Analysis", "scripts/per_horizon_metrics.py"),
        ("Sensor Dropout Robustness Analysis", "scripts/sensor_dropout_robustness.py"),
        ("Multiple Runs Statistical Analysis", "scripts/multiple_runs_statistics.py"),
        ("Calibration & Uncertainty Analysis", "scripts/calibration_analysis.py"),
    ]
    
    results = {}
    
    # Execute each analysis
    for analysis_name, script_path in analyses:
        success = run_analysis(analysis_name, script_path)
        results[analysis_name] = success
        
        if not success:
            print(f"\n⚠️  Warning: {analysis_name} did not complete successfully")
    
    # Print summary
    print("\n" + "="*120)
    print("[ANALYSIS PIPELINE SUMMARY]")
    print("="*120)
    
    all_success = True
    for analysis_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status:8s} | {analysis_name}")
        if not success:
            all_success = False
    
    print("\nOutput Files Generated:")
    results_dir = Path("results")
    if results_dir.exists():
        # List generated files
        new_files = [
            "per_horizon_metrics.csv",
            "per_horizon_metrics.json",
            "sensor_dropout_robustness.csv",
            "sensor_dropout_robustness.json",
            "sensor_dropout_robustness.png",
            "multiple_runs_statistics.json",
            "multiple_runs_summary.csv",
            "calibration_metrics.json",
            "calibration_summary.csv",
            "calibration_analysis.png",
        ]
        
        for fname in new_files:
            fpath = results_dir / fname
            if fpath.exists():
                size_kb = fpath.stat().st_size / 1024
                print(f"  [OK] {fname:45s} ({size_kb:8.2f} KB)")
            else:
                print(f"  [X] {fname:45s} (NOT FOUND)")
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*120 + "\n")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
