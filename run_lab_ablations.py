"""Checklist runner: three full-model seeds and full-test sensor evaluation.

Reuses seed-42 full models from run_lab_50.py. Runs sequentially on the lab GPU.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

VARIANTS = {'reference': [], 'no_diffusion': ['--no-diffusion'],
            'no_dropout': ['--no-dropout'], 'no_aleatoric': ['--no-aleatoric']}
SEEDS = (42, 123, 2026)


def expected(dataset, seed, variant):
    return dict(dataset=dataset, seed=seed, architecture='paper_interpretation_v1',
                hidden=64, batch_size=8, torch_threads=2, max_epochs=50,
                early_stopping_patience=50, train_windows_per_epoch=0,
                mc_passes=10, train_ratio=.7, val_ratio=.1, split='70/10/20',
                scheduler_factor=.5, loss_nll_ramp_epochs=10, train_sensor_mask_rate=.1,
                graph_threshold=.1, no_diffusion=variant=='no_diffusion',
                device='cuda', val_windows=3416 if dataset=='METR-LA' else 5200,
                test_windows=6844 if dataset=='METR-LA' else 10413,
                no_dropout=variant=='no_dropout', no_aleatoric=variant=='no_aleatoric',
                training_loss='mse_only_no_aleatoric_v2' if variant=='no_aleatoric' else 'linear_mse_to_nll_v2')


def completed(folder, settings):
    if not all((folder/name).exists() for name in ['completed.json','metrics.json','predictions.npz','best.pt','training.json']):
        return False
    metrics=json.loads((folder/'metrics.json').read_text())
    changed=[key for key,value in settings.items() if metrics.get(key)!=value]
    if changed:
        raise ValueError(f'{folder}: completed run settings differ: {changed}')
    if metrics['epochs_completed']!=50 or json.loads((folder/'completed.json').read_text())['epochs_completed']!=50:
        raise ValueError(f'{folder}: expected all 50 epochs')
    return True


def make_plan(root, existing, datasets, include_ablations=False):
    tasks=[]
    for dataset in datasets:
        for seed in SEEDS:
            for variant in (VARIANTS if include_ablations else ['reference']):
                reuse=seed==42 and variant=='reference'
                folder=existing/dataset if reuse else root/dataset/f'seed_{seed}'/variant
                tasks.append(dict(dataset=dataset,seed=seed,variant=variant,
                                  folder=str(folder),reuse_existing=reuse))
    # Train the two new full-model seeds first, then all missing variants.
    return sorted(tasks,key=lambda task: (task['variant']!='reference', task['dataset'],task['seed'],task['variant']))


def write_summary(root, tasks, paired=False):
    rows=[]
    for task in tasks:
        folder=Path(task['folder'])
        if not completed(folder,expected(task['dataset'],task['seed'],task['variant'])):
            continue
        metrics=json.loads((folder/'metrics.json').read_text())
        rows.append(dict(dataset=task['dataset'],seed=task['seed'],variant=task['variant'],
            mae_mph=metrics['mae_mph'],rmse_mph=metrics['rmse_mph'],
            coverage_95=metrics['coverage_95'],width_95_mph=metrics['width_95_mph'],
            best_epoch=metrics['best_epoch'],run=str(folder)))
    aggregate=[]
    for dataset in sorted({t['dataset'] for t in tasks}):
        for variant in VARIANTS:
            group=[r for r in rows if r['dataset']==dataset and r['variant']==variant]
            if not group:
                continue
            entry=dict(dataset=dataset,variant=variant,seeds=[r['seed'] for r in group],n=len(group))
            for field in ['mae_mph','rmse_mph','coverage_95','width_95_mph']:
                entry[field+'_mean']=float(np.mean([r[field] for r in group]))
                entry[field+'_sd']=float(np.std([r[field] for r in group],ddof=1)) if len(group)>1 else None
            aggregate.append(entry)
    pairs=[]
    if paired:
        from compare_ablation import compare
        for task in tasks:
            if task['variant']=='reference':
                continue
            reference=next(t for t in tasks if t['dataset']==task['dataset'] and t['seed']==task['seed'] and t['variant']=='reference')
            result=compare(reference['folder'],task['folder'],block_windows=288)
            pairs.append(dict(dataset=task['dataset'],seed=task['seed'],variant=task['variant'],**result))
    summary=dict(seeds=list(SEEDS),epochs=50,rows=rows,aggregate=aggregate,paired_comparisons=pairs,
                 completed_runs=len(rows),planned_runs=len(tasks),
                 note='SD across training seeds; paired bootstrap intervals describe sampled test blocks. No-aleatoric also changes training to MSE-only; no-dropout has one effective deterministic inference pass.')
    temporary=root/'ablation_summary.tmp'
    temporary.write_text(json.dumps(summary,indent=2))
    temporary.replace(root/'ablation_summary.json')
    (root/'multi_seed_summary.json').write_text(json.dumps(summary,indent=2))
    if rows:
        with (root/'per_seed_results.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if aggregate:
        with (root/'three_seed_summary.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
    if paired:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        includes_ablations=any(t['variant']!='reference' for t in tasks)
        with PdfPages(root/('ablation_graphs.pdf' if includes_ablations else 'multi_seed_graphs.pdf')) as pdf:
            for dataset in sorted({r['dataset'] for r in aggregate}):
                subset=[r for r in aggregate if r['dataset']==dataset]
                fig,axes=plt.subplots(1,2,figsize=(12,4))
                for ax,metric,label in zip(axes,['mae_mph','coverage_95'],['MAE (mph)','95% interval coverage']):
                    ax.bar(range(len(subset)),[r[metric+'_mean'] for r in subset],
                           yerr=[r[metric+'_sd'] or 0 for r in subset],capsize=4)
                    ax.set_xticks(range(len(subset)),[r['variant'].replace('_',' ') for r in subset],rotation=15)
                    ax.set_ylabel(label)
                    if metric=='coverage_95':
                        ax.axhline(.95,color='black',linestyle='--')
                        ax.set_ylim(0,1)
                fig.suptitle(f'{dataset}: seeds 42, 123, 2026; mean and sample SD')
                fig.tight_layout()
                fig.savefig(root/f'{dataset}_{"ablations" if includes_ablations else "full_model"}.png',dpi=200)
                pdf.savefig(fig)
                plt.close(fig)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-seed42',default='paper_results/lab_t1000_full50_seed42')
    parser.add_argument('--output-root',default='paper_results/lab_full50_three_seed_ablations')
    parser.add_argument('--datasets',nargs='+',choices=['METR-LA','PEMS-BAY'],default=['METR-LA','PEMS-BAY'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true',help='Print schedule without training or requiring a GPU')
    parser.add_argument('--summarize-only',action='store_true',help='Rebuild summary/plots from completed runs')
    parser.add_argument('--include-ablations',action='store_true',
                        help='Explicitly add separately trained ablations; not required by the latest checklist')
    parser.add_argument('--with-sensor-failures',action='store_true',default=True,
                        help='Run full-test sensor evaluation after training (enabled by default)')
    parser.add_argument('--skip-sensor-failures',dest='with_sensor_failures',action='store_false',
                        help='Only train/test models and produce ablation summaries')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent
    def resolve(path):
        path=Path(path)
        return path.resolve() if path.is_absolute() else (repo/path).resolve()
    root,existing=resolve(args.output_root),resolve(args.existing_seed42)
    tasks=make_plan(root,existing,args.datasets,args.include_ablations)
    if args.plan_only:
        print(json.dumps(tasks,indent=2))
        print(f'{len(tasks)} configurations; completed runs are reused automatically. No training started.')
        return
    for task in tasks:
        if task['reuse_existing'] and not completed(Path(task['folder']),expected(task['dataset'],task['seed'],task['variant'])):
            parser.error(f"Missing completed seed-42 reference: {task['folder']}")
    root.mkdir(parents=True,exist_ok=True)
    (root/'experiment_plan.json').write_text(json.dumps(tasks,indent=2))
    if args.summarize_only:
        if not all(completed(Path(t['folder']),expected(t['dataset'],t['seed'],t['variant'])) for t in tasks):
            parser.error('Some runs are incomplete; cannot produce a three-seed comparison yet')
        write_summary(root,tasks,paired=True)
        return
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA is required. Use the activated lab GPU environment.')
    for task in tasks:
        folder=Path(task['folder'])
        settings=expected(task['dataset'],task['seed'],task['variant'])
        if completed(folder,settings):
            print(f"Reusing completed {task['dataset']} seed {task['seed']} {task['variant']}",flush=True)
            continue
        extra=[]
        if args.resume and (folder/'latest.pt').exists():
            extra=['--resume']
        elif folder.exists():
            parser.error(f'Existing incomplete run: {folder}. Use --resume if its checkpoint exists; otherwise inspect it or select a new output root.')
        folder.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(repo/'paper_experiment.py'),'--dataset',task['dataset'],
                 '--output',str(folder),'--seed',str(task['seed']),'--epochs','50','--patience','50',
                 '--hidden','64','--batch-size','8','--threads','2','--device','cuda',
                 '--train-windows','0','--mc-passes','10','--log-every','25',*VARIANTS[task['variant']],*extra]
        status=dict(state='running',task=task,command=command)
        (root/'runner_status.json').write_text(json.dumps(status,indent=2))
        print(f"Starting {task['dataset']} seed {task['seed']} {task['variant']}",flush=True)
        with (folder.parent/(task['variant']+'.log')).open('a',encoding='utf-8') as logfile:
            process=subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace',bufsize=1)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    logfile.write(line)
                    logfile.flush()
                code=process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                status['state']='stopped'
                (root/'runner_status.json').write_text(json.dumps(status,indent=2))
                raise SystemExit(130)
        if code:
            status.update(state='failed',exit_code=code)
            (root/'runner_status.json').write_text(json.dumps(status,indent=2))
            raise SystemExit(code)
        if not completed(folder,settings):
            raise RuntimeError(f'{folder}: child exited without completed run artifacts')
        write_summary(root,tasks)
    print('Full-model runs complete; calculating requested summaries...',flush=True)
    write_summary(root,tasks,paired=True)
    (root/'runner_status.json').write_text(json.dumps(dict(state='training_completed',runs=len(tasks)),indent=2))
    print(f'Completed. Summaries, CSV tables and ablation plots: {root}',flush=True)
    if args.with_sensor_failures:
        command=[sys.executable,'-u',str(repo/'run_lab_sensor_failures.py'),
                 '--reference-root',str(root),'--existing-seed42',str(existing),
                 '--output-root',str(root/'sensor_failures'),
                 '--datasets',*args.datasets]
        if args.resume:
            command.append('--resume')
        (root/'runner_status.json').write_text(json.dumps(dict(state='sensor_failure_evaluation',command=command),indent=2))
        try:
            subprocess.run(command,cwd=repo,check=True)
        except subprocess.CalledProcessError as error:
            (root/'runner_status.json').write_text(json.dumps(dict(state='sensor_failure_failed',exit_code=error.returncode),indent=2))
            raise
    (root/'runner_status.json').write_text(json.dumps(dict(state='completed',runs=len(tasks),sensor_failures=args.with_sensor_failures),indent=2))
    print(f'Entire experiment queue completed: {root}',flush=True)


if __name__=='__main__':
    main()
