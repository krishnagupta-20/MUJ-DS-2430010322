"""Evaluate the trained model on the held-out test set. Run after train.py:  python src/test.py [--quick]"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize
from model import build_swin
from preprocessing import load_data, CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Single-seed test accuracies from the earlier CNN / ViT / Hybrid / ConvNeXt notebook (typed in, not re-computed here)
EARLIER = [
    ('Baseline CNN', 0.9291, 111370), ('Hybrid Conv-Transformer', 0.9272, 305194),
    ('CBAM-CNN', 0.9097, 117295), ('ConvNeXt-style', 0.8947, 221146), ('ViT-Small', 0.8636, 205834),
]


def ece(y_true, y_prob, n_bins=15):
    conf, pred = y_prob.max(1), y_prob.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    with open(os.path.join(ROOT, 'models', 'config.json')) as f:
        cfg = json.load(f)
    model = build_swin(name='Swin_Tiny', dropout=cfg['dropout'], drop_path=cfg['drop_path'])
    model.load_weights(os.path.join(ROOT, 'models', 'swin_tiny.weights.h5'))

    *_, x_test, y_test = load_data(quick=a.quick)
    prob = model.predict(x_test, batch_size=512, verbose=0)
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
    auc = roc_auc_score(label_binarize(y_test, classes=list(range(10))), prob, average='macro')
    acc = accuracy_score(y_test, pred)

    res = pd.DataFrame([{'Model': 'Swin-Tiny (pure transformer)', 'Accuracy': acc, 'Precision (macro)': p,
                         'Recall (macro)': r, 'F1 (macro)': f1, 'ROC-AUC (macro)': auc,
                         'ECE': ece(y_test, prob), 'NLL': log_loss(y_test, prob, labels=list(range(10))),
                         'Parameters': cfg['params'], 'Epochs run': cfg['epochs_run']}]).round(4)
    res.to_csv(os.path.join(ROOT, 'results', 'results.csv'), index=False)
    print(res.T)

    rows = [{'Model': n, 'Accuracy': acc_, 'Parameters': par, 'Source': 'Earlier notebook (single seed)'}
            for n, acc_, par in EARLIER]
    rows.append({'Model': 'Swin-Tiny (this work)', 'Accuracy': round(acc, 4),
                 'Parameters': cfg['params'], 'Source': 'This work'})
    pd.DataFrame(rows).sort_values('Accuracy', ascending=False).to_csv(
        os.path.join(ROOT, 'results', 'comparison.csv'), index=False)

    cm = confusion_matrix(y_test, pred, labels=list(range(10)))
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=100)
    plt.title(f'Swin-Tiny confusion matrix, % of each true class (acc {acc:.4f})')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'results', 'confusion_matrix.png'), dpi=150)
    print("Saved results/results.csv, results/comparison.csv, results/confusion_matrix.png")


if __name__ == '__main__':
    main()
