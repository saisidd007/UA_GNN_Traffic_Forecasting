"""Locate existing full-model results without copying predictions or checkpoints."""
from pathlib import Path


def report_root(seed, override=None):
    if override:
        return Path(override).resolve()
    return Path('paper_results/lab_t1000_full50_seed42' if seed==42 else
                'paper_results/lab_full50_three_seed_ablations').resolve()


def dataset_run(root, dataset, seed):
    root=Path(root)
    # Preserve the original seed-42 layout and support dataset/seed_N/reference.
    nested=root/dataset/f'seed_{seed}'/'reference'
    return nested if (nested/'metrics.json').exists() else root/dataset
