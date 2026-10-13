"""Full-test, repeated sensor-failure evaluation of the three saved full models.

No training. A case saves after all its test batches; --resume skips saved cases.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from corrected_experiment import load_data, prepare
from paper_model import PaperUAGNN, enable_mc_dropout, graph_edges, transitions


def failure_mask(generator, pattern, batch, nodes, rate, persistent_uniform):
    uniform = (np.broadcast_to(persistent_uniform, (batch,nodes)) if pattern=='persistent'
               else generator.random((batch,nodes)))
    return uniform < rate


def evaluate_case(model, device, values, observed, scale, indices, tod, dow,
                  forward, reverse, edges, pattern, rate, seed, args):
    generator=np.random.default_rng(seed)
    persistent=np.random.default_rng(seed).random(values.shape[1])
    torch.manual_seed(args.inference_seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.inference_seed)
    enable_mc_dropout(model)
    absolute, squared, covered, widths, count=0.,0.,0,0.,0
    batch_count=(len(indices)+args.batch_size-1)//args.batch_size
    with torch.no_grad():
        for batch,start in enumerate(range(0,len(indices),args.batch_size),1):
            chosen=indices[start:start+args.batch_size]
            history=chosen[:,None]+np.arange(12)[None,:]
            future=chosen[:,None]+12+np.arange(12)[None,:]
            drop=failure_mask(generator,pattern,len(chosen),values.shape[1],rate,persistent)
            x=np.where(drop[:,None,:],0,values[history])
            mask=observed[history] & ~drop[:,None,:]
            inputs=[torch.tensor(item,device=device) for item in [x,mask,tod[history],dow[history]]]
            samples=[model(*inputs,forward,reverse,edges) for _ in range(args.mc_passes)]
            means=torch.stack([s[0] for s in samples])
            predicted=means.mean(0).cpu().numpy()
            variance=(means.var(0,unbiased=False)+torch.stack([s[1] for s in samples]).mean(0)).cpu().numpy()
            target=values[future].transpose(0,2,1)
            valid=observed[future].transpose(0,2,1)
            error=((predicted-target)*scale[None,:,None])[valid]
            sigma=(np.sqrt(np.maximum(variance,0))*scale[None,:,None])[valid]
            absolute+=np.abs(error).sum(dtype=np.float64)
            squared+=np.square(error).sum(dtype=np.float64)
            covered+=int((np.abs(error)<=1.959964*sigma).sum())
            widths+=(2*1.959964*sigma).sum(dtype=np.float64)
            count+=len(error)
            if batch==1 or batch%25==0 or batch==batch_count:
                print(f'{pattern} rate={rate:.0%} corruption_seed={seed} batch={batch}/{batch_count}',flush=True)
    return dict(mae_mph=float(absolute/count),rmse_mph=float(np.sqrt(squared/count)),
                coverage_95=covered/count,width_95_mph=float(widths/count),observed_test_targets=count)


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2))
    temporary.replace(path)


def evaluate_run(folder, output, args):
    config=json.loads((folder/'metrics.json').read_text())
    if config['epochs_completed']!=50 or any(config[k] for k in ['no_dropout','no_diffusion','no_aleatoric']):
        raise ValueError(f'{folder}: requires completed 50-epoch full-model reference')
    raw,timestamps,adjacency=load_data(config['dataset'])
    values,observed,_,scale,splits,_=prepare(raw,train_ratio=config['train_ratio'],val_ratio=config['val_ratio'])
    indices=splits['test']
    if len(indices)!=config['test_windows']:
        raise ValueError('Dataset/split no longer matches the saved run test-window count')
    if args.windows:
        indices=indices[np.linspace(0,len(indices)-1,min(args.windows,len(indices)),dtype=int)]
    times=pd.to_datetime(timestamps)
    tod=(times.hour*12+times.minute//5).to_numpy(dtype=np.int64)
    dow=times.dayofweek.to_numpy(dtype=np.int64)
    device=torch.device(args.device)
    graph=torch.tensor(np.where(adjacency>config['graph_threshold'],adjacency,0),dtype=torch.float32,device=device)
    forward,reverse=transitions(graph)
    edges=graph_edges(graph)
    model=PaperUAGNN(hidden=config['hidden']).to(device)
    model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True))
    identity=dict(dataset=config['dataset'],training_seed=config['seed'],checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),
                  test_windows=len(indices),full_test_set=len(indices)==len(splits['test']),
                  batch_size=args.batch_size,threads=args.threads,mc_passes=args.mc_passes,
                  inference_seed=args.inference_seed,patterns=args.patterns,corruption_seeds=args.corruption_seeds,
                  failure_rates=args.failure_rates,device=args.device,torch_version=torch.__version__,
                  imputation='training median in scaled space',
                  persistent_definition='Same randomly selected failed sensors throughout the complete evaluation period')
    artifact=dict(settings=identity,rows=[])
    if output.exists():
        if not args.resume:
            raise ValueError(f'{output} exists; use --resume to reuse evaluated cases')
        artifact=json.loads(output.read_text())
        if artifact['settings']!=identity:
            raise ValueError('Sensor-failure resume settings/checkpoint differ; select another output root')
    for pattern in args.patterns:
        for rate in args.failure_rates:
            for seed in ([args.corruption_seeds[0]] if rate==0 else args.corruption_seeds):
                if any(r['pattern']==pattern and r['rate']==rate and r['corruption_seed']==seed for r in artifact['rows']):
                    continue
                print(f"{config['dataset']} training seed {config['seed']}: {pattern}, {rate:.0%}, corruption seed {seed}",flush=True)
                result=evaluate_case(model,device,values,observed,scale,indices,tod,dow,forward,reverse,edges,pattern,rate,seed,args)
                artifact['rows'].append(dict(pattern=pattern,rate=rate,corruption_seed=seed,**result))
                atomic_json(output,artifact)
    artifact['completed']=True
    atomic_json(output,artifact)
    return artifact


def summarize(root, artifacts):
    rows=[]
    for a in artifacts:
        for row in a['rows']:
            rows.append(dict(dataset=a['settings']['dataset'],training_seed=a['settings']['training_seed'],**row))
    with (root/'sensor_failure_per_case.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(root/'sensor_failure_graphs.pdf') as pdf:
        for dataset in sorted({r['dataset'] for r in rows}):
            for pattern in sorted({r['pattern'] for r in rows}):
                selected=[r for r in rows if r['dataset']==dataset and r['pattern']==pattern]
                rates=sorted({r['rate'] for r in selected})
                local=[]
                for rate in rates:
                    group=[r for r in selected if r['rate']==rate]
                    entry=dict(dataset=dataset,pattern=pattern,rate=rate,
                               training_seeds=sorted({r['training_seed'] for r in group}))
                    for metric in ['mae_mph','rmse_mph','coverage_95','width_95_mph']:
                        seed_means=[np.mean([r[metric] for r in group if r['training_seed']==s]) for s in entry['training_seeds']]
                        entry[metric+'_mean']=float(np.mean(seed_means))
                        entry[metric+'_sd_across_training_seeds']=float(np.std(seed_means,ddof=1)) if len(seed_means)>1 else None
                        entry[metric+'_sd_across_all_cases']=float(np.std([r[metric] for r in group],ddof=1)) if len(group)>1 else 0.
                    local.append(entry)
                    summaries.append(entry)
                fig,axes=plt.subplots(1,2,figsize=(11,4))
                for ax,metric in zip(axes,['mae_mph','rmse_mph']):
                    ax.errorbar(np.array(rates)*100,[r[metric+'_mean'] for r in local],
                                yerr=[r[metric+'_sd_across_training_seeds'] if r[metric+'_sd_across_training_seeds'] is not None else r[metric+'_sd_across_all_cases'] for r in local],marker='o',capsize=3)
                    ax.set(xlabel='Sensor failure (%)',ylabel=metric.replace('_',' ').upper())
                    ax.grid(alpha=.2)
                fig.suptitle(f'{dataset}: {pattern} outages; mean +/- SD across training-seed means' if len(local[0]['training_seeds'])>1 else f'{dataset}: {pattern} outages; SD across corruption seeds')
                fig.tight_layout()
                fig.savefig(root/f'{dataset}_{pattern}.png',dpi=200)
                pdf.savefig(fig)
                plt.close(fig)
    atomic_json(root/'sensor_failure_summary.json',dict(summary=summaries,
        note='Corruption repetitions are averaged within each training seed. Primary error bars are SD across those training-seed means; detailed per-case CSV also preserves corruption-seed variation.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',default='paper_results/lab_full50_three_seed_ablations')
    parser.add_argument('--existing-seed42',default='paper_results/lab_t1000_full50_seed42')
    parser.add_argument('--output-root',default='paper_results/lab_full_test_sensor_failures')
    parser.add_argument('--datasets',nargs='+',choices=['METR-LA','PEMS-BAY'],default=['METR-LA','PEMS-BAY'])
    parser.add_argument('--training-seeds',type=int,nargs='+',default=[42,123,2026])
    parser.add_argument('--corruption-seeds',type=int,nargs='+',default=[17,29,41])
    parser.add_argument('--failure-rates',type=float,nargs='+',default=[0,.1,.2,.3,.4,.5])
    parser.add_argument('--patterns',nargs='+',choices=['random','persistent'],default=['random','persistent'])
    parser.add_argument('--windows',type=int,default=0,help='0 = complete test set')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--mc-passes',type=int,default=10)
    parser.add_argument('--inference-seed',type=int,default=42)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.batch_size,args.threads,args.mc_passes)<1 or args.windows<0 or any(not 0<=r<=1 for r in args.failure_rates):
        parser.error('Invalid counts/rates')
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable in this Python environment')
    torch.set_num_threads(args.threads)
    root=Path(args.output_root).resolve()
    sources=[]
    for dataset in args.datasets:
        for seed in args.training_seeds:
            folder=Path(args.existing_seed42)/dataset if seed==42 else Path(args.reference_root)/dataset/f'seed_{seed}'/'reference'
            if not (folder/'completed.json').exists():
                parser.error(f'Missing completed full model: {folder}')
            saved=json.loads((folder/'metrics.json').read_text())
            if saved['seed']!=seed or saved['dataset']!=dataset:
                parser.error(f'Source run dataset/seed mismatch: {folder}')
            sources.append((dataset,seed,folder))
    root.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for dataset,seed,folder in sources:
        artifact=evaluate_run(folder,root/f'{dataset}_seed_{seed}.json',args)
        if artifact['settings']['training_seed']!=seed or artifact['settings']['dataset']!=dataset:
            raise ValueError('Source run dataset/seed mismatch')
        artifacts.append(artifact)
    summarize(root,artifacts)
    print(f'Full-test sensor-failure results and graphs saved: {root}',flush=True)


if __name__=='__main__':
    main()
