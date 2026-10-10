"""Evaluate all trained models on the held-out Fashion-MNIST test set.

    python src/test.py            # after src/train.py has produced models/*.weights.h5
    python src/test.py --quick    # evaluate on the 1,000-image quick subset (use after train.py --quick)

Outputs
    results/results.csv               accuracy, macro P/R/F1, specificity, ROC-AUC per model
    results/per_class_metrics.csv     per-class precision/recall/F1/specificity/AUC per model
    results/efficiency_calibration.csv  params, train time, latency, ECE, NLL per model
    results/comparison.csv            this work vs published reference results
    results/mcnemar_pairwise.csv      pairwise McNemar exact tests (Bonferroni threshold)
    results/findings_summary.txt      auto-generated findings from the numbers
    results/confusion_matrix.png      row-normalised confusion matrices, all models
    results/training_curve.png        train/validation loss + accuracy curves, all models
    figures/per_class_f1_heatmap.png, figures/mcnemar_pvalues.png, figures/vit_attention_rollout.png
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize

from model import REGISTRY, TransformerBlock, build_model, slug
from preprocessing import (CLASS_NAMES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED, ensure_dirs,
                           get_splits)

# Reference rows carried over from the earlier notebook. NOT independently verified:
# check each accuracy and the source against the original papers before submitting.
PUBLISHED = [
    ("VGG16", 0.935, "26,000,000"),
    ("GoogLeNet", 0.937, "Not reported"),
    ("MobileNet", 0.950, "Not reported"),
    ("DenseNet-BC", 0.954, "768,000"),
    ("WRN-28-10 + Random Erasing", 0.963, "Not reported"),
]
PUBLISHED_SOURCE = "UNVERIFIED - carried over from earlier notebook (cited there as arXiv:1802.07589)"


# ------------------------------------------------------------------ metrics
def specificity_per_class(cm):
    total, out = cm.sum(), []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.array(out)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    conf, pred = y_prob.max(1), y_prob.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def evaluate_model(model, name, x_test, y_test):
    labels = list(range(10))
    y_prob = model.predict(x_test, batch_size=512, verbose=0)
    y_pred = y_prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    y_bin = label_binarize(y_test, classes=labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    spec = specificity_per_class(cm)
    rep = classification_report(y_test, y_pred, labels=labels, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    per_class = pd.DataFrame({
        "Model": name, "Class": CLASS_NAMES,
        "Precision": [rep[c]["precision"] for c in CLASS_NAMES],
        "Recall": [rep[c]["recall"] for c in CLASS_NAMES],
        "F1": [rep[c]["f1-score"] for c in CLASS_NAMES],
        "Specificity": spec,
        "ROC-AUC": roc_auc_score(y_bin, y_prob, average=None)})
    return dict(name=name, y_prob=y_prob, y_pred=y_pred, cm=cm, per_class=per_class,
                accuracy=accuracy_score(y_test, y_pred),
                precision_macro=p, recall_macro=r, f1_macro=f1,
                precision_weighted=pw, recall_weighted=rw, f1_weighted=f1w,
                specificity_macro=spec.mean(),
                roc_auc_macro=roc_auc_score(y_bin, y_prob, average="macro"),
                ece=expected_calibration_error(y_test, y_prob),
                nll=log_loss(y_test, y_prob, labels=labels))


def latency_ms_per_1000(model, x_test, reps=3):
    xb = x_test[:1000]
    model.predict(xb, batch_size=500, verbose=0)           # warm-up
    t0 = time.time()
    for _ in range(reps):
        model.predict(xb, batch_size=500, verbose=0)
    return (time.time() - t0) / reps * 1000 * (1000 / len(xb))


def mcnemar_exact(y_true, pred_a, pred_b):
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int(np.sum(a_ok & ~b_ok))
    only_b = int(np.sum(~a_ok & b_ok))
    n_disc = only_a + only_b
    p = binomtest(min(only_a, only_b), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    return only_a, only_b, p


# ------------------------------------------------------------------ attention rollout
def vit_attention_rollout(vit, imgs):
    """Attention rollout (Abnar & Zuidema, 2020) for the ViT-Small model."""
    blocks = [l for l in vit.layers if isinstance(l, TransformerBlock)]
    x = vit.get_layer("patch_embed")(imgs)
    B, s, D = tf.shape(x)[0], x.shape[1], x.shape[-1]
    N = s * s
    x = vit.get_layer("pos_embed")(tf.reshape(x, (B, N, D)))
    rollout = tf.eye(N, batch_shape=[B])
    for blk in blocks:
        x, scores = blk(x, training=False, return_attention=True)
        a = tf.reduce_mean(scores, axis=1) + tf.eye(N)
        a = a / tf.reduce_sum(a, axis=-1, keepdims=True)
        rollout = tf.matmul(a, rollout)
    imp = tf.reshape(tf.reduce_mean(rollout, axis=1), (B, s, s, 1))
    imp = tf.image.resize(imp, (28, 28), method="bilinear")[..., 0].numpy()
    imp -= imp.min(axis=(1, 2), keepdims=True)
    return imp / (imp.max(axis=(1, 2), keepdims=True) + 1e-8)


# ------------------------------------------------------------------ plots
def plot_training_curves(names):
    palette = dict(zip(REGISTRY, sns.color_palette("tab10", len(REGISTRY))))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    drawn = False
    for name in names:
        path = os.path.join(RESULTS_DIR, f"history_{slug(name)}.csv")
        if not os.path.exists(path):
            continue
        h = pd.read_csv(path)
        c = palette[name]
        axes[0].plot(h["loss"], "--", color=c, alpha=0.5)
        axes[0].plot(h["val_loss"], color=c, label=name)
        axes[1].plot(h["accuracy"], "--", color=c, alpha=0.5)
        axes[1].plot(h["val_accuracy"], color=c, label=name)
        drawn = True
    if not drawn:
        plt.close(fig)
        print("No history_*.csv files found - training_curve.png skipped (run train.py first).")
        return
    axes[0].set_title("Loss (dashed = train, solid = validation)"); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Accuracy (dashed = train, solid = validation)"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curve.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        cm_pct = r["cm"] / r["cm"].sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=100)
        ax.set_title(f"{name}  (acc {r['accuracy']:.4f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=60, labelsize=8); ax.tick_params(axis="y", labelsize=8)
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="use the quick (1,000-image) test subset")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    ensure_dirs()

    hp_path = os.path.join(MODELS_DIR, "best_hyperparameters.json")
    if not os.path.exists(hp_path):
        sys.exit("models/best_hyperparameters.json not found - run `python src/train.py` first.")
    best_cfg = json.load(open(hp_path))

    *_, x_test, y_test = get_splits(quick=args.quick, seed=args.seed)

    # ---- load trained models ----
    trained = {}
    for name in REGISTRY:
        w_path = os.path.join(MODELS_DIR, f"{slug(name)}.weights.h5")
        if name in best_cfg and os.path.exists(w_path):
            m = build_model(name, **best_cfg[name])
            m.load_weights(w_path)
            trained[name] = m
        else:
            print(f"[warn] no trained weights for {name} - skipped")
    if not trained:
        sys.exit("No trained models found in models/ - run `python src/train.py` first.")

    results = {n: evaluate_model(m, n, x_test, y_test) for n, m in trained.items()}

    # ---- results.csv ----
    summary = pd.DataFrame([{
        "Model": n, "Family": REGISTRY[n]["family"],
        "Hyperparameters": json.dumps(best_cfg[n]),
        "Accuracy": r["accuracy"],
        "Precision (macro)": r["precision_macro"], "Recall (macro)": r["recall_macro"],
        "F1 (macro)": r["f1_macro"], "F1 (weighted)": r["f1_weighted"],
        "Specificity (macro)": r["specificity_macro"], "ROC-AUC (macro)": r["roc_auc_macro"],
    } for n, r in results.items()]).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    summary.round(4).to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print(summary.drop(columns="Hyperparameters").round(4).to_string(index=False))

    pd.concat([r["per_class"] for r in results.values()]).round(4).to_csv(
        os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False)

    # ---- efficiency + calibration ----
    times_path = os.path.join(RESULTS_DIR, "training_times.csv")
    times = pd.read_csv(times_path).set_index("Model")["Train Time (s)"].to_dict() if os.path.exists(times_path) else {}
    epochs_run = {}
    for n in trained:
        hp = os.path.join(RESULTS_DIR, f"history_{slug(n)}.csv")
        epochs_run[n] = len(pd.read_csv(hp)) if os.path.exists(hp) else np.nan
    eff = pd.DataFrame([{
        "Model": n, "Family": REGISTRY[n]["family"], "Params": m.count_params(),
        "Epochs Run": epochs_run[n], "Train Time (s)": times.get(n, np.nan),
        "Latency (ms / 1000 imgs)": round(latency_ms_per_1000(m, x_test), 1),
        "Accuracy": round(results[n]["accuracy"], 4),
        "ECE": round(results[n]["ece"], 4), "NLL": round(results[n]["nll"], 4),
    } for n, m in trained.items()]).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    eff["Acc per 100k params"] = (eff["Accuracy"] / (eff["Params"] / 1e5)).round(4)
    eff.to_csv(os.path.join(RESULTS_DIR, "efficiency_calibration.csv"), index=False)

    # ---- comparison with published results ----
    rows = [{"Model": n, "Accuracy": a, "Precision": "Not reported", "Recall": "Not reported",
             "F1": "Not reported", "AUC": "Not reported", "Parameters": p, "Source": PUBLISHED_SOURCE}
            for n, a, p in PUBLISHED]
    for n, r in results.items():
        rows.append({"Model": f"{n} (this work)", "Accuracy": round(r["accuracy"], 4),
                     "Precision": round(r["precision_macro"], 4), "Recall": round(r["recall_macro"], 4),
                     "F1": round(r["f1_macro"], 4), "AUC": round(r["roc_auc_macro"], 4),
                     "Parameters": f"{trained[n].count_params():,}", "Source": "This work"})
    pd.DataFrame(rows).sort_values("Accuracy", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "comparison.csv"), index=False)

    # ---- McNemar pairwise tests ----
    names = list(results)
    n_pairs = max(1, len(names) * (len(names) - 1) // 2)
    alpha = 0.05 / n_pairs
    pmat = pd.DataFrame(np.nan, index=names, columns=names)
    prow = []
    for a, b in itertools.combinations(names, 2):
        oa, ob, p = mcnemar_exact(y_test, results[a]["y_pred"], results[b]["y_pred"])
        pmat.loc[a, b] = pmat.loc[b, a] = p
        prow.append({"Model A": a, "Model B": b,
                     "Acc diff (A-B)": round(results[a]["accuracy"] - results[b]["accuracy"], 4),
                     "A right, B wrong": oa, "B right, A wrong": ob, "p-value": round(p, 4),
                     f"Significant (Bonferroni {alpha:.4f})": p < alpha})
    if prow:
        pd.DataFrame(prow).sort_values("p-value").to_csv(
            os.path.join(RESULTS_DIR, "mcnemar_pairwise.csv"), index=False)
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(pmat.astype(float), annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=0.1,
                    mask=pmat.isna())
        plt.title("McNemar p-values (green = no detectable difference)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "mcnemar_pvalues.png"), dpi=150)
        plt.close()

    # ---- figures ----
    plot_confusion_matrices(results)
    plot_training_curves(list(REGISTRY))

    f1_table = pd.DataFrame({n: r["per_class"].set_index("Class")["F1"] for n, r in results.items()})
    plt.figure(figsize=(10, 5))
    sns.heatmap(f1_table, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Per-class F1 by model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "per_class_f1_heatmap.png"), dpi=150)
    plt.close()

    if "ViT-Small" in trained:
        idx = [int(np.where(y_test == c)[0][0]) for c in range(10) if (y_test == c).any()]
        maps = vit_attention_rollout(trained["ViT-Small"], x_test[idx])
        fig, axes = plt.subplots(3, len(idx), figsize=(1.8 * len(idx), 6), squeeze=False)
        for j, i in enumerate(idx):
            axes[0, j].imshow(x_test[i].squeeze(), cmap="gray"); axes[0, j].set_title(CLASS_NAMES[y_test[i]], fontsize=8)
            axes[1, j].imshow(maps[j], cmap="jet")
            axes[2, j].imshow(x_test[i].squeeze(), cmap="gray"); axes[2, j].imshow(maps[j], cmap="jet", alpha=0.45)
            for r in range(3):
                axes[r, j].axis("off")
        plt.suptitle("ViT-Small attention rollout (one test image per class)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "vit_attention_rollout.png"), dpi=150)
        plt.close(fig)

    # ---- findings summary (generated from the numbers, so text cannot drift from results) ----
    lines = [f"Best overall: {summary.loc[0, 'Model']} ({summary.loc[0, 'Accuracy']:.4f})"]
    conv = summary[summary["Family"].isin(["CNN", "Modern CNN"])]
    attn = summary[summary["Family"].isin(["Transformer", "CNN+Transformer"])]
    if len(conv) and len(attn):
        bc, ba = conv.iloc[0], attn.iloc[0]
        _, _, p = mcnemar_exact(y_test, results[bc["Model"]]["y_pred"], results[ba["Model"]]["y_pred"])
        lines += [f"Best convolution-based model: {bc['Model']} ({bc['Accuracy']:.4f})",
                  f"Best attention/transformer-based model: {ba['Model']} ({ba['Accuracy']:.4f})",
                  f"Gap: {(ba['Accuracy'] - bc['Accuracy']) * 100:+.2f} points | McNemar p = {p:.4f} "
                  f"({'distinguishable' if p < 0.05 else 'NOT statistically distinguishable'} at 0.05)"]
    cnn_only = summary[summary["Family"] == "CNN"]
    if "ViT-Small" in results and len(cnn_only):
        ref = cnn_only.iloc[0]
        d = (results["ViT-Small"]["accuracy"] - ref["Accuracy"]) * 100
        lines.append(f"Pure ViT vs best reference CNN ({ref['Model']}): {d:+.2f} points")
        if "Hybrid Conv-Transformer" in results:
            hg = (results["Hybrid Conv-Transformer"]["accuracy"] - results["ViT-Small"]["accuracy"]) * 100
            lines.append(f"Hybrid vs pure ViT: {hg:+.2f} points (positive = convolutional priors helped)")
    lines.append("Calibration (ECE, lower is better): " + ", ".join(f"{n}={r['ece']:.4f}" for n, r in results.items()))
    lines.append("Note: single seed. Run `python src/train.py --seed-study` before claiming any ranking.")
    open(os.path.join(RESULTS_DIR, "findings_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
