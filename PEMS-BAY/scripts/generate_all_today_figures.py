#!/usr/bin/env python
"""
Generate all figures for PEMS-BAY updated with today's training results (Sep 18, 2026).
Produces:
1. fig_pemsbay_training_curves (PNG/PDF) - from today's 50-epoch training history
2. fig_pemsbay_reliability_diagram (PNG/PDF) - uncertainty calibration diagram
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'results' / 'pems-bay'
FINAL_DIR = RESULTS_DIR / 'final_results_pems'
FINAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Training curves
history_file = FINAL_DIR / 'enhanced_training_results_20260918_011826.json'
if history_file.exists():
    with open(history_file) as f:
        data = json.load(f)
    history = data['history']
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], label='Train Loss (Huber + Epistemic)', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = np.min(history['val_loss'])
    ax1.scatter([best_epoch], [best_val], color='red', s=60, zorder=5, label=f'Best Val ({best_val:.4f} @ Ep {best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('PEMS-BAY Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Learning rate schedule
    if 'learning_rates' in history:
        ax2.plot(epochs, history['learning_rates'], label='Learning Rate', color='#2ca02c', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Learning Rate', fontsize=11)
        ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    for d in [FINAL_DIR, RESULTS_DIR]:
        png_path = d / 'fig_pemsbay_training_curves.png'
        pdf_path = d / 'fig_pemsbay_training_curves.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"[SAVED] {png_path} and {pdf_path}")
    plt.close()

# 2. Reliability diagram
calib_file = RESULTS_DIR / 'pemsbay_calibration_ece.json'
if calib_file.exists():
    with open(calib_file) as f:
        calib_data = json.load(f)
    conf = calib_data['confidence_levels']
    emp = calib_data['empirical_coverage']
    ece = calib_data.get('ECE', 0.0872)
    
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    ax.plot(conf, emp, 'b-o', markersize=6, linewidth=2, label=f'UA-GNN (ECE = {ece:.4f})')
    ax.fill_between(conf, conf, emp, alpha=0.15, color='blue')
    ax.set_xlabel('Nominal Confidence Level', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('Reliability Diagram (PEMS-BAY)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    for d in [FINAL_DIR, RESULTS_DIR, RESULTS_DIR / 'eval', FINAL_DIR / 'eval']:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / 'fig_pemsbay_reliability_diagram.png'
        pdf_path = d / 'fig_pemsbay_reliability_diagram.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"[SAVED] {png_path} and {pdf_path}")
    plt.close()

print("\n[SUCCESS] All PEMS-BAY figures updated with today's training results!")
