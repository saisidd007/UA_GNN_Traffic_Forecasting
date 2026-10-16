"""Generate the existing full results-and-graphs report for seeds 123 and 2026."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[123,2026])
    parser.add_argument('--datasets',nargs='+',choices=['METR-LA','PEMS-BAY'],default=['METR-LA','PEMS-BAY'])
    parser.add_argument('--results-root',help='Optional result-root override')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent
    for seed in args.seeds:
        print(f'Generating seed-{seed} report from saved full-model results; no training.',flush=True)
        command=[sys.executable,'-u',str(repo/'generate_results_pdf.py'),'--seed',str(seed),
                 '--generate-graphs','--datasets',*args.datasets]
        if args.results_root:
            command.extend(['--results-root',str(Path(args.results_root).resolve())])
        subprocess.run(command,cwd=repo,check=True)
    print('Requested seed reports completed.',flush=True)


if __name__=='__main__':
    main()
