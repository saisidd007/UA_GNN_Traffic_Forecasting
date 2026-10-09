import time
import torch
import json
from pathlib import Path
import os
import sys
# ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.enhanced_gnn import create_enhanced_model
from src.utils.dataset import PEMSBayDataset


def measure():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # create model (use hidden_channels=64 to match saved checkpoint in this repo)
    model = create_enhanced_model(in_channels=12, hidden_channels=64, out_channels=12)
    model.to(device)
    model.eval()

    # try to load checkpoint
    ckpt_path = Path('results') / 'enhanced_best_model.pt'
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            print('Loaded checkpoint:', ckpt_path)
        except Exception as e:
            print('Failed to load checkpoint:', e)
    else:
        print('Checkpoint not found, using random init')

    # load dataset to get sample
    dataset = PEMSBayDataset('data')
    sample = dataset.get(0)
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)

    # prepare input shape expected: model.forward expects x as [num_nodes, in_channels]

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, edge_index)

    # measure latency
    iters = 200
    times = []
    with torch.no_grad():
        for i in range(iters):
            t0 = time.perf_counter()
            _ = model(x, edge_index)
            t1 = time.perf_counter()
            times.append((t1 - t0)*1000)

    import statistics
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)

    # parameter count
    total_params = sum(p.numel() for p in model.parameters())
    params_k = total_params/1000.0

    # memory estimate for parameters (float32)
    param_bytes = total_params * 4
    param_mb = param_bytes / (1024*1024)

    # try psutil for RSS
    try:
        import psutil
        p = psutil.Process(os.getpid())
        rss_mb = p.memory_info().rss / (1024*1024)
    except Exception:
        rss_mb = None

    results = {
        'device': str(device),
        'mean_latency_ms': mean_ms,
        'std_latency_ms': std_ms,
        'total_params': int(total_params),
        'params_k': params_k,
        'param_memory_mb': param_mb,
        'process_rss_mb': rss_mb
    }

    print(json.dumps(results, indent=2))
    out = Path('results') / 'inference_benchmark_proposed.json'
    out.write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    measure()
