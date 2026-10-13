"""Generate the four figures in final_results.pdf from completed lab runs.

Saved training and predictions supply three figures; sensor-dropout robustness
requires fresh inference using best.pt. No model is retrained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

from corrected_experiment import load_data, prepare
from paper_model import PaperUAGNN, enable_mc_dropout, graph_edges, transitions


def robustness(folder, config, raw, timestamps, adjacency, args):
    values, observed, _, scale, splits, _ = prepare(raw, train_ratio=config['train_ratio'], val_ratio=config['val_ratio'])
    indices = splits['test'][np.linspace(0, len(splits['test'])-1,
                                       min(args.robustness_windows, len(splits['test'])), dtype=int)]
    x = np.stack([values[i:i+12] for i in indices])
    masks = np.stack([observed[i:i+12] for i in indices])
    target = np.stack([values[i+12:i+24].T for i in indices])
    valid = np.stack([observed[i+12:i+24].T for i in indices])
    times = pd.to_datetime(timestamps)
    tod = (times.hour*12 + times.minute//5).to_numpy(dtype=np.int64)
    dow = times.dayofweek.to_numpy(dtype=np.int64)
    clock = np.stack([tod[i:i+12] for i in indices])
    days = np.stack([dow[i:i+12] for i in indices])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    graph = torch.tensor(np.where(adjacency > config['graph_threshold'], adjacency, 0), dtype=torch.float32, device=device)
    forward, reverse = transitions(graph)
    edges = graph_edges(graph)
    model = PaperUAGNN(hidden=config['hidden'], dropout=0 if config['no_dropout'] else .3,
                       use_diffusion=not config['no_diffusion']).to(device)
    model.load_state_dict(torch.load(folder/'best.pt', map_location=device, weights_only=True))

    def evaluate(drop):
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(42)
        enable_mc_dropout(model)
        error_sum, covered, count = 0., 0, 0
        with torch.no_grad():
            for start in range(0, len(x), args.batch_size):
                stop = start + args.batch_size
                xx = torch.tensor(np.where(drop[start:stop, None, :], 0, x[start:stop]), device=device)
                mm = torch.tensor(masks[start:stop] & ~drop[start:stop, None, :], device=device)
                pp = torch.tensor(clock[start:stop], device=device)
                dd = torch.tensor(days[start:stop], device=device)
                samples = [model(xx, mm, pp, dd, forward, reverse, edges)
                           for _ in range(1 if config['no_dropout'] else args.mc_passes)]
                means = torch.stack([s[0] for s in samples])
                mean = means.mean(0).cpu().numpy()
                epi = means.var(0, unbiased=False).cpu().numpy()
                alea = torch.stack([s[1] for s in samples]).mean(0).cpu().numpy()
                if config['no_aleatoric']:
                    alea[:] = 0
                error = np.abs((mean-target[start:stop])*scale[None, :, None])
                sigma = np.sqrt(np.maximum(epi+alea, 0))*scale[None, :, None]
                selected = valid[start:stop]
                error_sum += error[selected].sum(dtype=np.float64)
                covered += int((error[selected] <= 1.959964*sigma[selected]).sum())
                count += int(selected.sum())
        return dict(mae_mph=float(error_sum/count), coverage_95=covered/count)

    rows = []
    clean = evaluate(np.zeros((len(x), raw.shape[1]), dtype=bool))
    rows.append(dict(rate=0., corruption_seed=None, **clean))
    for seed in args.corruption_seeds:
        uniform = np.random.default_rng(seed).random((len(x), raw.shape[1]))
        for rate in args.failure_rates:
            row = dict(rate=rate, corruption_seed=seed, **evaluate(uniform < rate))
            rows.append(row)
            print(f"{config['dataset']}: sensor dropout {rate:.0%}, seed {seed}, MAE {row['mae_mph']:.4f}", flush=True)
    return dict(device=str(device), sampled_test_windows=len(indices), mc_passes=args.mc_passes,
                corruption_seeds=args.corruption_seeds, pattern='random sensors per historical window',
                imputation='training median (zero in scaled space)', inference_seed=42,
                note='Error bars describe corruption-seed variability, not training-seed variability.', rows=rows)


def generate(folder, args):
    config = json.loads((folder/'metrics.json').read_text())
    if config['epochs_completed'] != 50 or config['seed'] != 42:
        raise ValueError(f'{folder}: expected completed 50-epoch seed-42 run')
    if config['architecture'] != 'paper_interpretation_v1':
        raise ValueError('This graph generator requires the manuscript architecture interpretation')
    history = json.loads((folder/'training.json').read_text())
    raw, timestamps, adjacency = load_data(config['dataset'])
    _, observed, center, scale, splits, boundaries = prepare(raw, train_ratio=config['train_ratio'], val_ratio=config['val_ratio'])
    out = folder/'final_graphs'
    out.mkdir(exist_ok=True)
    manifest = dict(dataset=config['dataset'], source_run=str(folder.resolve()), seed=config['seed'],
                    epochs_completed=config['epochs_completed'], best_epoch=config['best_epoch'], figures=[],
                    note='New figures from corrected run; original PDF values and configuration are not reused.')
    with PdfPages(out/'all_graphs.pdf') as bundle:
        def save(fig, name):
            fig.tight_layout()
            fig.savefig(out/(name+'.png'), dpi=200)
            fig.savefig(out/(name+'.pdf'))
            bundle.savefig(fig)
            plt.close(fig)
            manifest['figures'].append(name)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['train_loss'] for r in history], color='tab:blue')
        axes[0].set(xlabel='Epoch', ylabel='Training objective', title='MSE → Gaussian NLL (first 10 epochs)')
        axes[0].axvline(11, color='gray', linestyle=':', label='NLL-only training begins')
        axes[0].legend(fontsize=8)
        axes[1].plot(epochs, [r['val_mse'] for r in history], color='tab:orange')
        axes[1].axvline(config['best_epoch'], color='green', linestyle='--', label='Selected epoch')
        axes[1].set(xlabel='Epoch', ylabel='Validation MSE (normalized)', title='Validation error')
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"{config['dataset']} · seed 42 · 50 epochs")
        save(fig, 'training_validation_curves')

        # Correlations use training observations only; missing values stay missing.
        subset = raw[:boundaries[0], :min(args.correlation_sensors, raw.shape[1])].copy()
        subset[~observed[:boundaries[0], :subset.shape[1]]] = np.nan
        corr = pd.DataFrame(subset).corr(min_periods=100).to_numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        fig.colorbar(im, ax=ax, label='Pearson correlation')
        ax.set(xlabel='Sensor index', ylabel='Sensor index', title=f"{config['dataset']}: training-only sensor correlations")
        np.savetxt(out/'sensor_correlations.csv', corr, delimiter=',')
        save(fig, 'sensor_correlation_heatmap')

        with np.load(folder/'predictions.npz') as predictions:
            mean, target = predictions['mean'], predictions['target']
            if len(mean) != len(splits['test']):
                raise ValueError('24-hour alignment requires full, ordered test predictions')
            sensor, step = args.sensor_index, args.horizon_step-1
            if sensor >= raw.shape[1]:
                raise ValueError(f'sensor-index must be below {raw.shape[1]}')
            # Forecast origin i+11 predicts target i+12+step; align by target time.
            times = pd.DatetimeIndex(pd.to_datetime(timestamps))[splits['test']+12+step]
            candidates = [day for day in times.normalize().unique()
                          if int((times.normalize() == day).sum()) == 288]
            if not candidates:
                raise ValueError('No complete 24-hour test day exists')
            day = pd.Timestamp(args.day) if args.day else candidates[0]
            chosen = np.flatnonzero(times.normalize() == day.normalize())
            selected_times = times[chosen]
            if len(chosen) != 288 or not np.all(selected_times[1:]-selected_times[:-1] == pd.Timedelta(minutes=5)):
                raise ValueError('Requested day must contain a continuous 24-hour test period')
            valid = predictions['target_observed'][chosen, sensor, step].astype(bool)
            actual = target[chosen, sensor, step]*scale[sensor]+center[sensor]
            actual = np.where(valid, actual, np.nan)
            predicted = mean[chosen, sensor, step]*scale[sensor]+center[sensor]
            variance = predictions['epistemic'][chosen, sensor, step]+predictions['aleatoric'][chosen, sensor, step]
            std = np.sqrt(np.maximum(variance, 0))*scale[sensor]
            lower, upper = predicted-1.959964*std, predicted+1.959964*std
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(times[chosen], lower, upper, alpha=.25, label='95% predictive interval')
            ax.plot(times[chosen], actual, color='black', lw=1, label='Observed speed')
            ax.plot(times[chosen], predicted, lw=1, label='Predicted speed')
            ax.set(xlabel='Target time', ylabel='Traffic speed (mph)',
                   title=f"{config['dataset']}: sensor index {sensor}, {args.horizon_step*5}-minute forecasts, {day.date()}")
            ax.legend()
            ax.grid(alpha=.2)
            save(fig, 'traffic_24h_uncertainty')
            pd.DataFrame(dict(timestamp=times[chosen], actual_mph=actual, prediction_mph=predicted,
                              lower95_mph=lower, upper95_mph=upper, target_observed=valid)).to_csv(out/'traffic_24h.csv', index=False)
            manifest.update(profile_day=str(day.date()), sensor_index=sensor, horizon_minutes=args.horizon_step*5)

        artifact_path = out/'sensor_dropout_results.json'
        checkpoint_hash = hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
        if args.reuse_robustness and artifact_path.exists():
            artifact = json.loads(artifact_path.read_text())
            if artifact.get('checkpoint_sha256') != checkpoint_hash:
                raise ValueError('Cached robustness belongs to a different checkpoint; omit --reuse-robustness')
        else:
            artifact = robustness(folder, config, raw, timestamps, adjacency, args)
            artifact['checkpoint_sha256'] = checkpoint_hash
            artifact_path.write_text(json.dumps(artifact, indent=2))
        rows = artifact['rows']
        rates = sorted({r['rate'] for r in rows})
        grouped = [[r['mae_mph'] for r in rows if r['rate'] == rate] for rate in rates]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(np.array(rates)*100, [np.mean(g) for g in grouped],
                    yerr=[np.std(g, ddof=1) if len(g)>1 else 0 for g in grouped], marker='o', capsize=4)
        ax.set(xlabel='Artificial sensor dropout (%)', ylabel='MAE (mph)',
               title=f"{config['dataset']}: {artifact['sampled_test_windows']} test windows, {artifact['mc_passes']} MC passes")
        ax.grid(alpha=.2)
        save(fig, 'sensor_dropout_robustness')
        manifest['robustness'] = {k:v for k,v in artifact.items() if k != 'rows'}
    (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"{config['dataset']}: graphs saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='paper_results/lab_t1000_full50_seed42')
    parser.add_argument('--datasets', nargs='+', choices=['METR-LA', 'PEMS-BAY'], default=['METR-LA', 'PEMS-BAY'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--robustness-windows', type=int, default=128)
    parser.add_argument('--mc-passes', type=int, default=10)
    parser.add_argument('--corruption-seeds', type=int, nargs='+', default=[17,29,41])
    parser.add_argument('--failure-rates', type=float, nargs='+', default=[.1,.2,.3,.4,.5])
    parser.add_argument('--sensor-index', type=int, default=0)
    parser.add_argument('--horizon-step', type=int, choices=range(1,13), default=12)
    parser.add_argument('--correlation-sensors', type=int, default=16)
    parser.add_argument('--day', help='Optional YYYY-MM-DD; omit to choose the first complete test day')
    parser.add_argument('--reuse-robustness', action='store_true', help='Reuse existing sensor dropout evaluation')
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.robustness_windows, args.mc_passes, args.correlation_sensors) < 1 or args.sensor_index < 0:
        parser.error('Counts must be positive; sensor-index must be nonnegative')
    if any(not 0 < r <= 1 for r in args.failure_rates):
        parser.error('failure-rates must be in (0,1]')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable in this Python environment')
    torch.set_num_threads(args.threads)
    root = Path(args.results_root).resolve()
    # Check both runs before doing work; avoid silently reading incomplete outputs.
    for dataset in args.datasets:
        for name in ['metrics.json','training.json','predictions.npz','best.pt']:
            if not (root/dataset/name).exists():
                parser.error(f'Missing {root/dataset/name}; wait for training/testing to finish')
    for dataset in args.datasets:
        generate(root/dataset, args)


if __name__ == '__main__':
    main()
