"""Report available CPUs/GPU and benchmark the actual model without long training."""
import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from corrected_experiment import load_data
from paper_model import PaperUAGNN, graph_edges, transitions


def available_cpus():
    logical = os.cpu_count() or 1
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else logical
    quota = None
    try:
        limit, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if limit != 'max':
            quota = int(limit) / int(period)
    except (OSError, ValueError):
        pass
    usable = min(affinity, max(1, math.floor(quota))) if quota else affinity
    return dict(logical_cpus=logical, affinity_cpus=affinity,
                cpu_quota=quota, usable_threads=usable, cpu_name=platform.processor())


def benchmark(dataset, device, threads, batch_size, repeats):
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    raw, _, adjacency = load_data(dataset)
    graph = torch.tensor(np.where(adjacency > .1, adjacency, 0), dtype=torch.float32, device=device)
    forward, reverse = transitions(graph)
    edges = graph_edges(graph)
    model = PaperUAGNN().to(device)
    x = torch.randn(batch_size, 12, raw.shape[1], device=device)
    mask = torch.ones_like(x, dtype=torch.bool)
    clock = torch.zeros(batch_size, 12, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    timings = []
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for iteration in range(repeats + 1):
        sync()
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(x, mask, clock, clock, forward, reverse, edges)
        (.5 * (variance.log() + mean.square()/variance)).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        sync()
        train_seconds = time.perf_counter() - started
        model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            model(x, mask, clock, clock, forward, reverse, edges)
        sync()
        validation_seconds = time.perf_counter() - started
        if iteration:
            timings.append((train_seconds, validation_seconds))
    return dict(dataset=dataset, device=device, threads=threads, batch_size=batch_size,
                train_batch_seconds=statistics.median(t[0] for t in timings),
                validation_batch_seconds=statistics.median(t[1] for t in timings),
                peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', default='lab_hardware_report.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error('batch-size and repeats must be positive')
    hardware = available_cpus()
    hardware.update(platform=platform.platform(), torch_version=torch.__version__,
                    cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hardware.update(gpu_name=props.name, gpu_total_memory_mib=props.total_memory/2**20)
    try:
        hardware['system_details'] = subprocess.run(
            ['lscpu'], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        hardware['ram_details'] = Path('/proc/meminfo').read_text().splitlines()[:3]
    except OSError:
        pass
    maximum = hardware['usable_threads']
    candidates = sorted({n for n in [1, 2, 4, 8, 16, maximum] if n <= maximum})
    devices = ['cuda'] if torch.cuda.is_available() else ['cpu']
    report = dict(hardware=hardware, thread_candidates=candidates, benchmarks=[],
                  note='Short synthetic-batch benchmark, not a full-epoch runtime estimate. Physical CPU cores differ from logical threads.')
    print(json.dumps(hardware, indent=2), flush=True)
    output = Path(args.output)
    for device in devices:
        for threads in candidates:
            for dataset in ['METR-LA', 'PEMS-BAY']:
                try:
                    row = benchmark(dataset, device, threads, args.batch_size, args.repeats)
                except torch.cuda.OutOfMemoryError:
                    row = dict(dataset=dataset, device=device, threads=threads, error='CUDA out of memory; rerun with a smaller batch size')
                report['benchmarks'].append(row)
                print(json.dumps(row), flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                output.write_text(json.dumps(report, indent=2))
    scores = {}
    for threads in candidates:
        rows = [r for r in report['benchmarks'] if r['threads'] == threads]
        if len(rows) == 2 and all('error' not in r for r in rows):
            scores[threads] = sum(r['train_batch_seconds'] + r['validation_batch_seconds'] for r in rows)
    report['recommended_threads_for_tested_batch'] = min(scores, key=scores.get) if scores else None
    output.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {output}. Send this JSON file for review.", flush=True)


if __name__ == '__main__':
    main()
