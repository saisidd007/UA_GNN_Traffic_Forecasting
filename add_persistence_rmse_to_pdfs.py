"""Update the three existing seed PDFs using cached results, without inference."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,123,2026])
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent
    # Validate all original JSON reports before starting work.
    from report_run_paths import report_root
    for seed in args.seeds:
        name='UA_GNN_Final_Results.json' if seed==42 else f'UA_GNN_Final_Results_seed{seed}.json'
        if not (report_root(seed)/name).exists():
            parser.error(f'Missing saved report: {report_root(seed)/name}; no analysis will be rerun')
    for seed in args.seeds:
        subprocess.run([sys.executable,'-u',str(repo/'generate_results_pdf.py'),
                        '--seed',str(seed),'--tables-only'],cwd=repo,check=True)
    print('Updated PDFs saved with _with_persistence_RMSE suffix. Originals preserved.',flush=True)


if __name__=='__main__':
    main()
