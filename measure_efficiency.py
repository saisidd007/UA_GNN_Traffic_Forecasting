"""Measure inference time and peak resident memory on the current machine."""

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from corrected_experiment import TrafficModel, load_data
from paper_model import PaperUAGNN, graph_edges, transitions


def run(args):
    torch.set_num_threads(args.threads)
    folder = Path(args.run)
    config = json.loads((folder / "metrics.json").read_text())
    _, _, adjacency = load_data(config["dataset"])
    nodes = len(adjacency)
    model_type = config.get("architecture")
    if model_type == "paper_interpretation_v1":
        graph = torch.tensor(np.where(adjacency > .1, adjacency, 0), dtype=torch.float32)
        forward, reverse = transitions(graph)
        edges = graph_edges(graph)
        model = PaperUAGNN(hidden=config["hidden"], use_diffusion=not config["no_diffusion"])
        values = torch.zeros(args.batch_size, 12, nodes)
        mask = torch.ones_like(values, dtype=torch.bool)
        clock = torch.zeros(args.batch_size, 12, dtype=torch.long)
        predict = lambda: model(values, mask, clock, clock, forward, reverse, edges)
    else:
        graph = torch.tensor((adjacency > .1).astype(np.float32))
        graph.fill_diagonal_(1)
        graph /= graph.sum(1, keepdim=True).clamp_min(1)
        if config.get("no_graph"):
            graph = torch.eye(nodes)
        model = TrafficModel()
        values = torch.zeros(args.batch_size, nodes, 12)
        mask = torch.ones_like(values, dtype=torch.bool)
        predict = lambda: model(values, mask, graph)
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        for _ in range(args.warmup):
            predict()
        elapsed = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            predict()
            elapsed.append(1000*(time.perf_counter()-start))
    result = {"dataset": config["dataset"], "architecture": model_type or "corrected_small",
              "parameters_total": total_parameters,
              "parameters_active": config.get("parameters", total_parameters),
              "device": "cpu", "torch_version": torch.__version__,
              "platform": platform.platform(), "logical_cpus": os.cpu_count(),
              "allowed_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "torch_threads": args.threads, "batch_size": args.batch_size,
              "mc_passes": 1,
              "warmup": args.warmup, "repetitions": args.repetitions,
              "median_batch_ms": float(np.median(elapsed)),
              "p95_batch_ms": float(np.percentile(elapsed, 95)),
              "forecast_windows_per_second": float(args.batch_size/(np.median(elapsed)/1000)),
              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (folder / "efficiency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    run(parser.parse_args())
