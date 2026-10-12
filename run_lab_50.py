"""Native Windows/Linux launcher for the benchmarked T1000 lab configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-root', default='paper_results/lab_t1000_full50_seed42')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error('threads and batch-size must be positive')
    repo = Path(__file__).resolve().parent
    root = Path(args.output_root)
    if not root.is_absolute():
        root = repo / root
    import torch
    if not torch.cuda.is_available():
        parser.error(f'CUDA required: this Python environment cannot use GPU ({torch.__version__})')
    root.mkdir(parents=True, exist_ok=True)
    for dataset in ['METR-LA', 'PEMS-BAY']:
        output = root / dataset
        marker = output / 'completed.json'
        if args.resume and marker.exists():
            completed = json.loads(marker.read_text())
            if completed.get('epochs_completed') == 50:
                print(f'{dataset}: 50 epochs and testing already completed; skipping.', flush=True)
                continue
        extra = []
        if args.resume and (output / 'latest.pt').exists():
            extra = ['--resume']
        elif output.exists():
            parser.error(f'Existing output has no selected resume checkpoint: {output}. Use --resume or a new --output-root.')
        command = [sys.executable, '-u', str(repo / 'paper_experiment.py'),
                   '--dataset', dataset, '--output', str(output), '--seed', '42',
                   '--epochs', '50', '--patience', '50', '--hidden', '64',
                   '--batch-size', str(args.batch_size), '--threads', str(args.threads),
                   '--device', 'cuda', '--train-windows', '0', '--mc-passes', '10',
                   '--log-every', '25', *extra]
        (root / f'{dataset}_command.json').write_text(json.dumps(command, indent=2))
        print(f'Starting {dataset}: GPU, seed 42, 50 full-data epochs, batch {args.batch_size}, threads {args.threads}', flush=True)
        with (root / f'{dataset}.log').open('a', encoding='utf-8') as logfile:
            process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace', bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    logfile.write(line)
                    logfile.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                print('\nStopped. Resume with: python run_lab_50.py --resume', flush=True)
                raise SystemExit(130)
        if code:
            print(f'{dataset} stopped with exit code {code}. Check {root / (dataset + ".log")}. Resume after fixing the error.', flush=True)
            raise SystemExit(code)
    print(f'Both datasets completed. Results: {root}', flush=True)


if __name__ == '__main__':
    main()
