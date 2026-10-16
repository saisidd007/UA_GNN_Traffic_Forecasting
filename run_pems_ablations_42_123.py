"""Resume/train only PEMS-BAY ablations for seeds 42 and 123; no other jobs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_lab_ablations import completed, expected

VARIANTS={'no_aleatoric':['--no-aleatoric'], 'no_diffusion':['--no-diffusion'],
          'no_dropout':['--no-dropout']}


def make_tasks(root,existing):
    tasks=[]
    for seed in [42,123]:
        reference=existing/'PEMS-BAY' if seed==42 else root/'PEMS-BAY'/f'seed_{seed}'/'reference'
        for variant in VARIANTS:
            tasks.append(dict(seed=seed,variant=variant,
                folder=root/'PEMS-BAY'/f'seed_{seed}'/variant,reference=reference))
    return tasks


def summarize(tasks,report,paired=False):
    rows=[]
    sources={}
    for task in tasks:
        sources[(task['seed'],'reference')]=task['reference']
        sources[(task['seed'],task['variant'])]=task['folder']
    for (seed,variant),folder in sources.items():
        if not completed(folder,expected('PEMS-BAY',seed,variant)):
            continue
        m=json.loads((folder/'metrics.json').read_text())
        rows.append(dict(dataset='PEMS-BAY',seed=seed,variant=variant,mae_mph=m['mae_mph'],
            rmse_mph=m['rmse_mph'],coverage_95=m['coverage_95'],width_95_mph=m['width_95_mph'],
            best_epoch=m['best_epoch'],run=str(folder)))
    aggregates=[]
    for variant in ['reference',*VARIANTS]:
        group=[r for r in rows if r['variant']==variant]
        if not group:
            continue
        entry=dict(variant=variant,seeds=[r['seed'] for r in group],n=len(group))
        for key in ['mae_mph','rmse_mph','coverage_95','width_95_mph']:
            entry[key+'_mean']=float(np.mean([r[key] for r in group]))
            entry[key+'_sd']=float(np.std([r[key] for r in group],ddof=1)) if len(group)>1 else None
        aggregates.append(entry)
    pairs=[]
    if paired:
        from compare_ablation import compare
        for task in tasks:
            pairs.append(dict(seed=task['seed'],variant=task['variant'],
                **compare(task['reference'],task['folder'],block_windows=288)))
    summary=dict(dataset='PEMS-BAY',seeds=[42,123],rows=rows,aggregate=aggregates,paired_comparisons=pairs,
                 complete=len(rows)==8,note='Means and sample SD across two training seeds only; no-aleatoric uses MSE-only training.')
    temporary=report/'summary.tmp'
    temporary.write_text(json.dumps(summary,indent=2))
    temporary.replace(report/'summary.json')
    for name,data in [('per_seed_results.csv',rows),('two_seed_summary.csv',aggregates)]:
        if data:
            with (report/name).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    if paired:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(report/'PEMS_BAY_ablations_seeds42_123.pdf') as pdf:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,metric,label in zip(axes,['mae_mph','rmse_mph'],['MAE (mph)','RMSE (mph)']):
                ax.bar(range(len(aggregates)),[r[metric+'_mean'] for r in aggregates],
                       yerr=[r[metric+'_sd'] or 0 for r in aggregates],capsize=4)
                ax.set_xticks(range(len(aggregates)),[r['variant'].replace('_',' ') for r in aggregates],rotation=15)
                ax.set_ylabel(label)
            fig.suptitle('PEMS-BAY: seeds 42 and 123, mean +/- sample SD')
            fig.tight_layout()
            fig.savefig(report/'ablation_errors.png',dpi=200)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',default='paper_results/lab_full50_three_seed_ablations')
    parser.add_argument('--existing-seed42',default='paper_results/lab_t1000_full50_seed42')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent
    def resolve(value):
        path=Path(value)
        return path.resolve() if path.is_absolute() else (repo/path).resolve()
    root,existing=resolve(args.output_root),resolve(args.existing_seed42)
    tasks=make_tasks(root,existing)
    if args.plan_only:
        print(json.dumps(tasks,default=str,indent=2))
        print('Only 6 PEMS-BAY ablations: seeds 42 and 123; completed runs skipped, partial runs resumed.')
        return
    for task in tasks:
        if not completed(task['reference'],expected('PEMS-BAY',task['seed'],'reference')):
            parser.error(f"Missing completed full model: {task['reference']}. Full models will not be retrained.")
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA required; activate the lab GPU environment')
    report=root/'PEMS-BAY'/'ablation_report_seeds42_123'
    report.mkdir(parents=True,exist_ok=True)
    (report/'plan.json').write_text(json.dumps(tasks,default=str,indent=2))
    for task in tasks:
        folder,seed,variant=task['folder'],task['seed'],task['variant']
        if completed(folder,expected('PEMS-BAY',seed,variant)):
            print(f'Reusing PEMS-BAY seed {seed} {variant}',flush=True)
            continue
        extra=[]
        if (folder/'latest.pt').exists():
            extra=['--resume']
        elif folder.exists():
            parser.error(f'{folder}: existing folder has no epoch checkpoint. Inspect it before retrying; no files were deleted.')
        folder.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(repo/'paper_experiment.py'),'--dataset','PEMS-BAY',
            '--output',str(folder),'--seed',str(seed),'--epochs','50','--patience','50',
            '--hidden','64','--batch-size','8','--threads','2','--device','cuda',
            '--train-windows','0','--mc-passes','10','--log-every','25',*VARIANTS[variant],*extra]
        print(f'Starting PEMS-BAY seed {seed} {variant}'+(' from checkpoint' if extra else ''),flush=True)
        (report/'status.json').write_text(json.dumps(dict(state='running',seed=seed,variant=variant,command=command),indent=2))
        with (folder.parent/(variant+'.log')).open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    log.write(line)
                    log.flush()
                code=process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                (report/'status.json').write_text(json.dumps(dict(state='stopped',seed=seed,variant=variant)))
                raise SystemExit(130)
        if code:
            (report/'status.json').write_text(json.dumps(dict(state='failed',seed=seed,variant=variant,exit_code=code)))
            raise SystemExit(code)
        if not completed(folder,expected('PEMS-BAY',seed,variant)):
            raise RuntimeError(f'{folder}: did not complete all 50 epochs and final testing')
        summarize(tasks,report)
    summarize(tasks,report,paired=True)
    (report/'status.json').write_text(json.dumps(dict(state='completed',dataset='PEMS-BAY',seeds=[42,123])))
    print(f'All requested PEMS-BAY ablations completed. Results: {report}',flush=True)


if __name__=='__main__':
    main()
