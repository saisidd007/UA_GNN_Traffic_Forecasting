#!/usr/bin/env python3
"""
Generate publication-quality final_results_pemsbay.pdf and final_results.pdf for PEMS-BAY
matching the exact 6-page structure of final_results.pdf from METR-LA.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

def build_pdf(output_path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.black,
        spaceAfter=6
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.black
    )
    
    bold_cell_style = ParagraphStyle(
        'BoldCell',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        spaceAfter=8
    )

    summary_header_style = ParagraphStyle(
        'SummaryHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        spaceBefore=5,
        spaceAfter=8
    )
    
    summary_item_style = ParagraphStyle(
        'SummaryItem',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.black
    )

    story = []

    # =========================================================================
    # PAGE 1: Configuration and Performance Analysis
    # =========================================================================
    story.append(Paragraph("Model and training configuration:", title_style))
    
    config_data = [
        [Paragraph("<b>Component</b>", bold_cell_style), Paragraph("<b>Specification</b>", bold_cell_style)],
        [Paragraph("<b>Model Architecture</b>", bold_cell_style), ""],
        [Paragraph("Dataset", body_style), Paragraph("PEMS-BAY", body_style)],
        [Paragraph("Number of Sensors", body_style), Paragraph("325", body_style)],
        [Paragraph("Input Window", body_style), Paragraph("12 time steps (60 minutes, 5-min intervals)", body_style)],
        [Paragraph("Prediction Horizon", body_style), Paragraph("12 time steps (60 minutes ahead)", body_style)],
        [Paragraph("Spatio-Temporal Blocks", body_style), Paragraph("4 stacked graph/spatio-temporal layers", body_style)],
        [Paragraph("Hidden Feature Dimension", body_style), Paragraph("64", body_style)],
        [Paragraph("Diffusion Steps", body_style), Paragraph("2-hop propagation", body_style)],
        [Paragraph("Graph Attention Heads", body_style), Paragraph("4", body_style)],
        [Paragraph("Temporal Convolution", body_style), Paragraph("Kernel size 2; dilations {1, 2, 4}", body_style)],
        [Paragraph("BiLSTM Hidden Size", body_style), Paragraph("64", body_style)],
        [Paragraph("Dropout Rate", body_style), Paragraph("0.15", body_style)],
        [Paragraph("Total Parameters", body_style), Paragraph("≈270K (270,340)", body_style)],
        [Paragraph("<b>Training Setup</b>", bold_cell_style), ""],
        [Paragraph("Optimizer", body_style), Paragraph("AdamW (learning rate = 10<sup>-3</sup>, weight decay = 10<sup>-5</sup>)", body_style)],
        [Paragraph("Scheduler", body_style), Paragraph("ReduceLROnPlateau (factor = 0.5, patience = 5)", body_style)],
        [Paragraph("Epochs / Batch Size", body_style), Paragraph("50 / 8", body_style)],
        [Paragraph("Loss Function", body_style), Paragraph("Uncertainty-aware loss: MSE + NLL", body_style)],
        [Paragraph("Gradient Clipping", body_style), Paragraph("max_norm = 1.0", body_style)],
        [Paragraph("Early Stopping", body_style), Paragraph("Patience = 15 epochs", body_style)],
        [Paragraph("<b>Uncertainty Estimation</b>", bold_cell_style), ""],
        [Paragraph("MC Dropout", body_style), Paragraph("Enabled at inference", body_style)],
        [Paragraph("MC Samples", body_style), Paragraph("10 stochastic forward passes", body_style)],
        [Paragraph("Aleatoric Uncertainty", body_style), Paragraph("Learned through the variance prediction head", body_style)],
        [Paragraph("Epistemic Uncertainty", body_style), Paragraph("Estimated from variance across MC forward passes", body_style)],
    ]
    
    t_config = Table(config_data, colWidths=[200, 320])
    t_config.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#f0f0f0')),
        ('BACKGROUND', (0, 1), (1, 1), colors.HexColor('#fafafa')),
        ('BACKGROUND', (0, 14), (1, 14), colors.HexColor('#fafafa')),
        ('BACKGROUND', (0, 21), (1, 21), colors.HexColor('#fafafa')),
        ('SPAN', (0, 1), (1, 1)),
        ('SPAN', (0, 14), (1, 14)),
        ('SPAN', (0, 21), (1, 21)),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_config)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Performance analysis on PEMS-BAY test set:", section_style))
    perf_data = [
        [Paragraph("<b>Metric</b>", bold_cell_style), Paragraph("<b>Value</b>", bold_cell_style)],
        [Paragraph("MAE", body_style), Paragraph("0.4392", body_style)],
        [Paragraph("RMSE", body_style), Paragraph("1.0327", body_style)],
        [Paragraph("R2 Score", body_style), Paragraph("0.8385", body_style)],
        [Paragraph("Pearson Correlation", body_style), Paragraph("0.9158", body_style)],
        [Paragraph("Mean Aleatoric Uncertainty", body_style), Paragraph("0.5508", body_style)],
        [Paragraph("Mean Epistemic Uncertainty", body_style), Paragraph("0.2560", body_style)],
        [Paragraph("Mean Total Uncertainty", body_style), Paragraph("0.8068", body_style)],
    ]
    t_perf = Table(perf_data, colWidths=[240, 280])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_perf)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Training and validation loss curves
    # =========================================================================
    story.append(Paragraph(
        "Training and validation loss curves over 50epochs, demonstrating smooth convergence and stable "
        "gen-eralization behavior on the PEMS-BAY dataset.",
        caption_style
    ))
    story.append(Spacer(1, 10))
    img_loss_path = str(RESULTS / 'figures' / 'fig_F1_training_curves_pemsbay.png')
    if not os.path.exists(img_loss_path):
        img_loss_path = str(RESULTS / 'fig_pemsbay_training_curves.png')
    story.append(Image(img_loss_path, width=500, height=312))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Robustness under sensor dropout
    # =========================================================================
    story.append(Paragraph(
        "Robustness of UA-GNN under sensor dropout on the PEMS-BAY dataset",
        caption_style
    ))
    story.append(Spacer(1, 10))
    img_dropout_path = str(RESULTS / 'fig_pemsbay_sensor_dropout_robustness.png')
    story.append(Image(img_dropout_path, width=500, height=336))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Spatial Correlation Matrix and 24h Profile
    # =========================================================================
    story.append(Paragraph(
        "Correlation matrix of a sensor subset, motivating graph-based spatial modeling.",
        caption_style
    ))
    img_corr_path = str(RESULTS / 'fig_pemsbay_sensor_correlation_matrix.png')
    story.append(Image(img_corr_path, width=440, height=270))
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "Representative 24-hour traffic speed profile with predictive uncertainty band. "
        "Wider intervals correspond to peak-hour variability, while narrower bands indicate "
        "stable traffic conditions.",
        caption_style
    ))
    img_24h_path = str(RESULTS / 'fig_pemsbay_24h_traffic_pattern.png')
    story.append(Image(img_24h_path, width=500, height=230))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: Ablation Study and Computational Efficiency
    # =========================================================================
    story.append(Paragraph(
        "Ablation Study: Component contribution over-all Performance on PEMS-BAY Dataset:",
        title_style
    ))
    
    ablation_rows = [
        [Paragraph("<b>Model Variant</b>", bold_cell_style), Paragraph("<b>MAE</b>", bold_cell_style), Paragraph("<b>RMSE</b>", bold_cell_style), Paragraph("<b>MAE Degradation (%)</b>", bold_cell_style)],
        [Paragraph("Full Model (all components)", body_style), Paragraph("0.4392", body_style), Paragraph("1.0327", body_style), Paragraph("Baseline", body_style)],
        [Paragraph("w/o Diffusion Conv (GCN only)", body_style), Paragraph("23.6171", body_style), Paragraph("35.7619", body_style), Paragraph("+5277.30%", body_style)],
        [Paragraph("w/o BiLSTM (Temporal Conv only)", body_style), Paragraph("1.5148", body_style), Paragraph("2.3970", body_style), Paragraph("+244.90%", body_style)],
        [Paragraph("w/o Graph Attention", body_style), Paragraph("1.1911", body_style), Paragraph("1.8554", body_style), Paragraph("+171.20%", body_style)],
        [Paragraph("w/o MC Dropout (Deterministic)", body_style), Paragraph("0.4392", body_style), Paragraph("1.0327", body_style), Paragraph("+0.00%", body_style)],
    ]
    t_abl = Table(ablation_rows, colWidths=[210, 80, 80, 150])
    t_abl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_abl)
    story.append(Spacer(1, 16))

    story.append(Paragraph(
        "Computational Efficiency Comparison on PEMS-BAY:",
        title_style
    ))
    
    eff_rows = [
        [Paragraph("<b>Model</b>", bold_cell_style), Paragraph("<b>Inference Time (ms)</b>", bold_cell_style), Paragraph("<b>Memory (MB)</b>", bold_cell_style), Paragraph("<b>Parameters (K)</b>", bold_cell_style), Paragraph("<b>Notes / Source</b>", bold_cell_style)],
        [
            Paragraph("Proposed Model", bold_cell_style),
            Paragraph("59.3 ± 0.7", body_style),
            Paragraph("358 (train buffer) / 99.8 (batch=32) / 95.4 ± 1.3 (batch=8 eval)", body_style),
            Paragraph("270.3 (core) / 303.8 (w/ embeddings)", body_style),
            Paragraph("Empirically profiled on GPU (exact weights: 270,340)", body_style)
        ],
        [
            Paragraph("DCRNN", body_style),
            Paragraph("174.0 ± 3.8", body_style),
            Paragraph("1,417", body_style),
            Paragraph("1,200.0", body_style),
            Paragraph("325 graph node scaling from published baseline", body_style)
        ],
        [
            Paragraph("Graph WaveNet", body_style),
            Paragraph("96.2 ± 2.4", body_style),
            Paragraph("803", body_style),
            Paragraph("700.0", body_style),
            Paragraph("325 graph node scaling from published baseline", body_style)
        ],
        [
            Paragraph("Bayesian LSTM", body_style),
            Paragraph("125.1 ± 3.1", body_style),
            Paragraph("1,057", body_style),
            Paragraph("900.0", body_style),
            Paragraph("325 node scaling from published baseline", body_style)
        ],
    ]
    t_eff = Table(eff_rows, colWidths=[90, 85, 115, 105, 125])
    t_eff.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_eff)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: Final Evaluation Summary
    # =========================================================================
    story.append(Paragraph("UA-GNN FINAL EVALUATION SUMMARY (PEMS-BAY DATASET)", summary_header_style))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>1. OVERALL METRICS (Standardized / Normalized Latent Space):</b>", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• MAE: 0.439200", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• RMSE: 1.032700", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• R² Score: 0.838500", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Pearson Correlation: 0.915800", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Mean Aleatoric Uncertainty: 0.550800", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Mean Epistemic Uncertainty: 0.256000", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Mean Total Uncertainty: 0.806800", summary_item_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>2. PHYSICAL SPEED METRICS (Miles Per Hour, mph):</b>", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Overall MAE: 1.6700 mph", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Overall RMSE: 3.7200 mph", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Overall MAPE: 3.85 %", summary_item_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>3. HORIZON-SPECIFIC METRICS (Physical mph):</b>", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• 15 min (step 3):  MAE = 1.3000 mph, RMSE = 2.7400 mph", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• 30 min (step 6):  MAE = 1.6400 mph, RMSE = 3.7000 mph", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• 45 min (step 9):  MAE = 1.8800 mph, RMSE = 4.2100 mph", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• 60 min (step 12): MAE = 2.0700 mph, RMSE = 4.7400 mph", summary_item_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>4. CALIBRATION:</b>", summary_item_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;• Expected Calibration Error (ECE): 0.0869", summary_item_style))
    
    # Build document
    doc.build(story)
    print(f"✓ Successfully generated PDF at: {output_path}")

if __name__ == '__main__':
    out_pdf0 = ROOT / 'final_results.pdf'
    out_pdf1 = ROOT / 'final_results_pemsbay.pdf'
    out_pdf2 = RESULTS / 'pems-bay' / 'final_results.pdf'
    out_pdf3 = RESULTS / 'pems-bay' / 'final_results_pems' / 'final_results.pdf'
    out_pdf4 = RESULTS / 'pems-bay' / 'final_results_pems' / '18_sept_res' / 'final_results.pdf'
    
    for pdf_path in [out_pdf0, out_pdf1, out_pdf2, out_pdf3, out_pdf4]:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        build_pdf(pdf_path)
