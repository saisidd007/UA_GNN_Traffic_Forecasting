"""Build a results PDF from the completed one-seed lab runs and their figures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

import numpy as np
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analyze_predictions import analyze
from corrected_experiment import load_data, prepare
from report_run_paths import dataset_run, report_root


def number(value, digits=4):
    return 'Not available' if value is None else f'{value:.{digits}f}'


def mape(folder, config):
    raw, _, _ = load_data(config['dataset'])
    _, _, center, scale, _, _ = prepare(raw, train_ratio=config['train_ratio'], val_ratio=config['val_ratio'])
    total, count = 0., 0
    with np.load(folder/'predictions.npz') as saved:
        mean, target, valid = saved['mean'], saved['target'], saved['target_observed']
        for start in range(0, len(mean), 128):
            yy = target[start:start+128]*scale[None,:,None]+center[None,:,None]
            error = np.abs((mean[start:start+128]-target[start:start+128])*scale[None,:,None])
            mask = valid[start:start+128].astype(bool) & (yy > 1.)
            total += (error[mask]/yy[mask]).sum(dtype=np.float64)
            count += int(mask.sum())
    return dict(mape_percent=float(100*total/count) if count else None,
                mape_target_threshold_mph=1., mape_observed_targets=count)


def persistence_rmse_by_horizon(config, baseline):
    """CPU baseline calculation only; never loads a trained model or predictions."""
    raw,_,_=load_data(config['dataset'])
    values,observed,_,scale,splits,_=prepare(raw,train_ratio=config['train_ratio'],val_ratio=config['val_ratio'])
    indices=splits['test']
    if config['test_windows']!=len(indices):
        raise ValueError('Expected full-test results; split/window count differs')
    squares=np.zeros(12,dtype=np.float64)
    absolute=np.zeros(12,dtype=np.float64)
    counts=np.zeros(12,dtype=np.int64)
    for start in range(0,len(indices),128):
        chosen=indices[start:start+128]
        future=chosen[:,None]+12+np.arange(12)[None,:]
        target=values[future].transpose(0,2,1)
        valid=observed[future].transpose(0,2,1)
        error=(values[chosen+11,:,None]-target)*scale[None,:,None]
        squares+=(np.square(error)*valid).sum(axis=(0,1),dtype=np.float64)
        absolute+=(np.abs(error)*valid).sum(axis=(0,1),dtype=np.float64)
        counts+=valid.sum(axis=(0,1))
    if np.any(counts==0):
        raise ValueError('No observed targets at one or more horizons')
    # Refuse a different dataset/split rather than silently changing the baseline.
    if int(counts.sum())!=baseline['observed_test_targets'] or not np.isclose(
            np.sqrt(squares.sum()/counts.sum()),baseline['rmse_mph'],rtol=1e-5,atol=1e-5):
        raise ValueError('Recomputed persistence baseline differs from the saved report')
    np.testing.assert_allclose(absolute/counts,baseline['mae_by_horizon_mph'],rtol=1e-5,atol=1e-5)
    return np.sqrt(squares/counts).tolist()


def build_report(root, output, datasets, refresh_analysis=False, seed=42, cached_report=None):
    fonts = Path(reportlab.__file__).resolve().parent/'fonts'
    pdfmetrics.registerFont(TTFont('ReportSans', str(fonts/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSansBold', str(fonts/'VeraBd.ttf')))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'ReportSans'
    styles['Title'].fontName = styles['Heading1'].fontName = styles['Heading2'].fontName = 'ReportSansBold'
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 13
    story, summaries = [], {}
    if cached_report:
        saved=json.loads(Path(cached_report).read_text())
        summaries={dataset:saved[dataset] for dataset in datasets}
        for dataset,s in summaries.items():
            if s['metrics']['seed']!=seed or s['metrics']['dataset']!=dataset:
                raise ValueError('Cached report seed/dataset mismatch')

    def text(content, style='Normal'):
        story.append(Paragraph(escape(str(content)), styles[style]))
        story.append(Spacer(1,8))

    def table(rows, headings=('Item','Value')):
        cells = [[Paragraph(escape(str(cell)), styles['Normal']) for cell in row] for row in [headings,*rows]]
        widths = [185,330] if len(headings)==2 else [515/len(headings)]*len(headings)
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf5')),
                              ('GRID',(0,0),(-1,-1),.3,colors.lightgrey),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                              ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t)
        story.append(Spacer(1,12))

    # Validate outputs before computing analyses or creating the report.
    for dataset in ([] if cached_report else datasets):
        folder = dataset_run(root,dataset,seed)
        for filename in ['metrics.json','training.json','predictions.npz','persistence_baseline.json',
                         'final_graphs/figure_manifest.json','final_graphs/sensor_dropout_results.json',
                         'final_graphs/training_validation_curves.png','final_graphs/sensor_correlation_heatmap.png',
                         'final_graphs/traffic_24h_uncertainty.png','final_graphs/sensor_dropout_robustness.png']:
            if not (folder/filename).is_file():
                raise FileNotFoundError(f'Missing {folder/filename}. Finish training and run generate_final_graphs.py first.')
        metrics = json.loads((folder/'metrics.json').read_text())
        if metrics['epochs_completed'] != 50 or metrics['seed'] != seed or metrics['dataset'] != dataset:
            raise ValueError(f'{folder}: expected dataset-matched, completed 50-epoch seed-{seed} run')
        if metrics['architecture'] != 'paper_interpretation_v1':
            raise ValueError(f'{folder}: architecture is not the supported manuscript interpretation')
        audit_path = folder/'prediction_audit.json'
        if refresh_analysis or not audit_path.exists():
            print(f'{dataset}: computing saved-prediction analysis...', flush=True)
            analyze(folder)
        audit = json.loads(audit_path.read_text())
        baseline = json.loads((folder/'persistence_baseline.json').read_text())
        if baseline['observed_test_targets'] != metrics['observed_test_targets']:
            raise ValueError(f'{dataset}: baseline and model target counts differ')
        history = json.loads((folder/'training.json').read_text())
        if len(history) != 50 or min(history,key=lambda r:r['val_mse'])['epoch'] != metrics['best_epoch']:
            raise ValueError(f'{dataset}: training history does not match selected checkpoint')
        summaries[dataset] = dict(metrics=metrics, audit=audit, persistence=baseline,
            additional_metrics=mape(folder,metrics),
            figure_manifest=json.loads((folder/'final_graphs/figure_manifest.json').read_text()),
            robustness=json.loads((folder/'final_graphs/sensor_dropout_results.json').read_text()),
            training_seconds=sum(r['epoch_seconds'] for r in history))

    for dataset,s in summaries.items():
        if 'rmse_by_horizon_mph' not in s['persistence']:
            print(f'{dataset}: calculating missing persistence horizon RMSE on CPU only...',flush=True)
            s['persistence']['rmse_by_horizon_mph']=persistence_rmse_by_horizon(s['metrics'],s['persistence'])

    text('UA-GNN Traffic Forecasting', 'Title')
    text('Corrected implementation: single-seed, 50-epoch experimental results', 'Heading2')
    text(f'This report uses saved test predictions, validation-selected checkpoints and measured results from the latest lab runs. Each dataset was trained with seed {seed}. The manuscript leaves some architecture details unspecified, so this implementation is an interpretation rather than an exact reproduction.')
    rows = []
    for dataset,s in summaries.items():
        m=s['metrics']
        rows.append([dataset,number(m['mae_mph']),number(m['rmse_mph']),
                     number(100*m['coverage_95'],2)+'%',str(m['best_epoch'])])
    table(rows,('Dataset','MAE (mph)','RMSE (mph)','95% coverage','Best epoch'))
    text('Protocol: chronological 70/10/20 split; training-only scaling and causal missing-input handling; missing targets excluded; checkpoint selection on validation data; final test evaluation using 10 MC-dropout passes.')
    text('Coverage alone does not establish useful uncertainty: interval width and calibration must also be reported. One training seed does not establish consistency across seeds.')

    for dataset,s in summaries.items():
        m,a,b=s['metrics'],s['audit'],s['persistence']
        folder=dataset_run(root,dataset,seed)
        story.append(PageBreak())
        text(dataset+' - model and training configuration','Heading1')
        table([['Architecture',m['architecture']],['Sensors',207 if dataset=='METR-LA' else 325],
            ['History / forecast horizon','12 / 12 time steps (60 / 60 minutes)'],
            ['Spatio-temporal blocks','4; temporal dilations 1, 2, 4, 8'],
            ['Diffusion / graph attention','Bidirectional 2-hop diffusion; 4 attention heads'],
            ['Hidden dimension / BiLSTM hidden size',m['hidden']],['Parameters',m['parameters']],
            ['Dropout','0.3'],['Optimizer','Adam; initial learning rate 0.001'],
            ['Scheduler','ReduceLROnPlateau; factor 0.5; patience 10'],
            ['Training objective','Linear replacement of MSE by Gaussian NLL over 10 epochs'],
            ['Gradient clipping','Maximum norm 5.0'],['Training sensor masking','10%'],
            ['Seed / completed epochs',f"{m['seed']} / {m['epochs_completed']}"],
            ['Selected epoch',m['best_epoch']],['Batch size / CPU threads',f"{m['batch_size']} / {m['torch_threads']}"],
            ['Device',m['device']],['Training windows','All windows' if m['train_windows_per_epoch']==0 else m['train_windows_per_epoch']],
            ['Validation / test windows',f"{m['val_windows']} / {m['test_windows']}"],
            ['MC passes',m['mc_passes']],['Graph non-self edges',m['graph_edges_without_self_loops']],
            ['Training time (excludes final test)',number(s['training_seconds']/3600,2)+' hours']])

        story.append(PageBreak())
        text(dataset+' - test performance and uncertainty','Heading1')
        table([['MAE / RMSE (mph)',f"{number(m['mae_mph'])} / {number(m['rmse_mph'])}"],
            ['Normalized MAE / RMSE',f"{number(a['normalized_mae'])} / {number(a['normalized_rmse'])}"],
            ['R-squared / Pearson correlation',f"{number(a['r2'])} / {number(a['pearson_r'])}"],
            ['MAPE (observed targets above 1 mph)',number(s['additional_metrics']['mape_percent'])+'%'],
            ['Epistemic variance (normalized)',number(m['epistemic_variance_normalized'])],
            ['Aleatoric variance (normalized)',number(m['aleatoric_variance_normalized'])],
            ['Total variance (normalized)',number(m['epistemic_variance_normalized']+m['aleatoric_variance_normalized'])],
            ['Epistemic / aleatoric variance (mph squared)',f"{number(a['epistemic_variance_mph2'])} / {number(a['aleatoric_variance_mph2'])}"],
            ['Coverage ECE (nominal 10%-95%)',number(a['coverage_ece_10_to_95'])],
            ['Gaussian CRPS (mph)',number(a['gaussian_crps_mph'])],
            ['Observed test targets',m['observed_test_targets']]])
        table([[f'{level}%',number(100*m[f'coverage_{level}'],2)+'%',number(m[f'width_{level}_mph'])]
               for level in [50,80,90,95]],('Nominal interval','Observed coverage','Mean width (mph)'))
        table([[str(step*5),number(a['mae_by_horizon_mph'][step-1]),number(a['rmse_by_horizon_mph'][step-1])]
               for step in [3,6,9,12]],('Horizon (minutes)','MAE (mph)','RMSE (mph)'))

        story.append(PageBreak())
        text(dataset+' - matched persistence comparison','Heading1')
        table([['Corrected model',number(m['mae_mph']),number(m['rmse_mph'])],
               ['Persistence',number(b['mae_mph']),number(b['rmse_mph'])],
               ['Error reduction (%)',number(100*(1-m['mae_mph']/b['mae_mph']),2),
                number(100*(1-m['rmse_mph']/b['rmse_mph']),2)]],('Method','MAE (mph)','RMSE (mph)'))
        table([[str((i+1)*5),number(m['mae_by_horizon_mph'][i]),number(b['mae_by_horizon_mph'][i]),
                number(a['rmse_by_horizon_mph'][i]),number(b['rmse_by_horizon_mph'][i])]
               for i in range(12)],('Horizon (minutes)','Model MAE (mph)','Persistence MAE (mph)',
                                    'Model RMSE (mph)','Persistence RMSE (mph)'))
        ci=a.get('paired_daily_block_bootstrap_95ci_mph')
        if ci:
            text('Descriptive paired day-block bootstrap 95% interval for model minus persistence MAE: '
                 f'[{number(ci[0])}, {number(ci[1])}] mph. This reflects test-day sampling, not variation across training seeds.')

        manifest=s['figure_manifest']
        captions=[('training_validation_curves','Training objective and validation error. The objective changes from MSE to Gaussian NLL; their values cannot be directly compared as the same loss.'),
            ('sensor_correlation_heatmap','Pearson correlations for the selected sensor subset, using observed training readings only. Correlation is descriptive and is not proof of model benefit.'),
            ('traffic_24h_uncertainty',f"Sensor index {manifest['sensor_index']}, {manifest['profile_day']}, {manifest['horizon_minutes']}-minute forecasts. Bands are marginal Gaussian 95% predictive intervals. Negative lower bounds, if present, are a limitation of the Gaussian formulation."),
            ('sensor_dropout_robustness',f"Random historical sensor failures evaluated on {s['robustness']['sampled_test_windows']} sampled test windows with {s['robustness']['mc_passes']} MC passes. Error bars are standard deviations across corruption seeds, using one trained model; this is not a full-test or multi-training-seed robustness result.")]
        for name,caption in captions:
            story.append(PageBreak())
            text(dataset+' - '+name.replace('_',' '),'Heading1')
            image=Image(str(folder/'final_graphs'/(name+'.png')))
            factor=min(515/image.imageWidth,570/image.imageHeight)
            image.drawWidth=image.imageWidth*factor
            image.drawHeight=image.imageHeight*factor
            story.append(image)
            story.append(Spacer(1,12))
            text(caption)

    story.append(PageBreak())
    text('Scope and remaining evidence','Heading1')
    text('This report contains the full-model single-seed results and the four graph types in the supplied results document for each selected dataset. It does not reuse the original document\'s numerical claims or training configuration.')
    text('Ablations without diffusion, BiLSTM, graph attention or dropout require separately trained and matched runs. Published-model efficiency comparisons require matched hardware/protocol measurements. Those results cannot be inferred from the two full-model checkpoints and are not populated here.')
    text('The sensor-dropout figure measures random failures of historical sensor inputs, with training-median imputation. It does not cover every failure pattern. Forecast plots describe a specified day and sensor; they do not establish anticipation of congestion or calibration across the entire test set.')
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('ReportSans',8)
        canvas.drawString(40,22,'UA-GNN | corrected single-seed lab experiments')
        canvas.drawRightString(A4[0]-40,22,str(doc.page))
    temporary=output.with_suffix('.tmp.pdf')
    SimpleDocTemplate(str(temporary),pagesize=A4,leftMargin=40,rightMargin=40,
                      topMargin=36,bottomMargin=36).build(story,onFirstPage=footer,onLaterPages=footer)
    temporary.replace(output)
    output.with_suffix('.json').write_text(json.dumps(summaries,indent=2))
    print(f'Report saved: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',help='Optional override of the results root')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--datasets',nargs='+',choices=['METR-LA','PEMS-BAY'],default=['METR-LA','PEMS-BAY'])
    parser.add_argument('--output',help='Default: RESULTS_ROOT/UA_GNN_Final_Results.pdf')
    parser.add_argument('--refresh-analysis',action='store_true')
    parser.add_argument('--tables-only',action='store_true',help='Reuse original report JSON and existing graphs; calculate missing persistence RMSE only')
    parser.add_argument('--generate-graphs',action='store_true',help='Generate all four figures first (includes extra checkpoint inference)')
    args=parser.parse_args()
    if args.tables_only and (args.generate_graphs or args.refresh_analysis):
        parser.error('--tables-only cannot regenerate graphs or analysis')
    root=report_root(args.seed,args.results_root)
    if args.generate_graphs:
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('generate_final_graphs.py')),
                        '--results-root',str(root),'--seed',str(args.seed),'--datasets',*args.datasets],check=True)
    filename='UA_GNN_Final_Results.pdf' if args.seed==42 else f'UA_GNN_Final_Results_seed{args.seed}.pdf'
    cached=root/Path(filename).with_suffix('.json') if args.tables_only else None
    if cached and not cached.exists():
        parser.error(f'Saved report JSON missing: {cached}. This mode will not rerun analysis.')
    if args.tables_only:
        filename=filename.replace('.pdf','_with_persistence_RMSE.pdf')
    output=Path(args.output).resolve() if args.output else root/filename
    build_report(root,output,args.datasets,args.refresh_analysis,args.seed,cached)


if __name__=='__main__':
    main()
