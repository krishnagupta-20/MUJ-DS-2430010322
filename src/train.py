"""Hyper-parameter tuning + final training of all five models.

    python src/train.py                 # full run (30 epochs, 6-epoch tuning sweep)
    python src/train.py --quick         # tiny subset, 2 epochs: checks the pipeline only
    python src/train.py --models "ViT-Small" "Baseline CNN"
    python src/train.py --skip-tuning   # reuse models/best_hyperparameters.json
    python src/train.py --seed-study    # extra: retrain each model with several seeds (slow)

Outputs
    models/<model>.weights.h5          best weights (restored by early stopping)
    models/best_hyperparameters.json   selected configuration per model
    results/tuning_results.csv         every configuration tried + validation accuracy
    results/history_<model>.csv        per-epoch loss/accuracy (train + validation)
    results/training_times.csv         wall-clock training time per model
    results/convergence.csv            epochs run, best epoch, train-val gap
    results/seed_study.csv             (only with --seed-study)
Re-running skips models that already have weights + history, so an interrupted run resumes.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
import keras
from keras import callbacks
from sklearn.metrics import accuracy_score

from model import REGISTRY, build_model, slug
from preprocessing import (SEED, MODELS_DIR, RESULTS_DIR, ensure_dirs, get_splits)

BEST_HP_PATH = os.path.join(MODELS_DIR, "best_hyperparameters.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny subset + 2 epochs (pipeline check only)")
    p.add_argument("--epochs", type=int, default=None, help="final-training epochs (default 30; 2 if --quick)")
    p.add_argument("--tune-epochs", type=int, default=None, help="epochs per tuning run (default 6; 1 if --quick)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=6, help="early-stopping patience on val_loss")
    p.add_argument("--models", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    p.add_argument("--skip-tuning", action="store_true", help="reuse models/best_hyperparameters.json")
    p.add_argument("--seed-study", action="store_true", help="retrain each model with extra seeds")
    p.add_argument("--seed-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def make_optimizer(epochs, n_train, args):
    """AdamW, 5% linear warm-up then cosine decay, gradient clipping 1.0."""
    steps = math.ceil(n_train / args.batch_size) * epochs
    warm = max(1, int(0.05 * steps))
    sched = keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=max(1, steps - warm), alpha=0.01,
        warmup_target=args.lr, warmup_steps=warm)
    return keras.optimizers.AdamW(learning_rate=sched, weight_decay=args.weight_decay, clipnorm=1.0)


def compile_and_train(model, data, epochs, args, patience, verbose=0):
    x_train, y_train, x_val, y_val = data
    model.compile(optimizer=make_optimizer(epochs, len(x_train), args),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = ([callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                    restore_best_weights=True)] if patience else [])
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=args.batch_size, callbacks=cbs, verbose=verbose)
    return hist, time.time() - t0


def tune(models_to_tune, data, args, tune_epochs):
    rows, best = [], {}
    for name in models_to_tune:
        grid = REGISTRY[name]["grid"]
        keys = list(grid)
        best_acc, best_hp = -1.0, None
        for vals in itertools.product(*grid.values()):
            hp = dict(zip(keys, vals))
            keras.utils.set_random_seed(args.seed)
            m = build_model(name, **hp)
            h, _ = compile_and_train(m, data, tune_epochs, args, patience=None)
            acc = max(h.history["val_accuracy"])
            rows.append({"Model": name, "Hyperparameters": json.dumps(hp), "Best Val Acc": round(acc, 4)})
            print(f"{name:26s} {str(hp):38s} val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc, best_hp = acc, hp
            del m
            keras.backend.clear_session()
        best[name] = best_hp
    return pd.DataFrame(rows), best


def merge_csv(path, new_df, key="Model"):
    """Write new_df to path, keeping rows of models not present in new_df (supports --models)."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old[key].isin(new_df[key])]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def main():
    args = parse_args()
    ensure_dirs()
    epochs = args.epochs or (2 if args.quick else 30)
    tune_epochs = args.tune_epochs or (1 if args.quick else 6)
    keras.utils.set_random_seed(args.seed)

    print("TensorFlow:", __import__("tensorflow").__version__, "| Keras:", keras.__version__)
    x_train, y_train, x_val, y_val, x_test, y_test = get_splits(quick=args.quick, seed=args.seed)
    data = (x_train, y_train, x_val, y_val)
    print(f"train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    # ---- 1. hyper-parameter tuning (selection by validation accuracy only) ----
    best_cfg = {}
    if os.path.exists(BEST_HP_PATH):
        best_cfg = json.load(open(BEST_HP_PATH))
    if args.skip_tuning:
        missing = [m for m in args.models if m not in best_cfg]
        if missing:
            sys.exit(f"--skip-tuning but no stored hyper-parameters for: {missing}")
    else:
        tuning_df, new_best = tune(args.models, data, args, tune_epochs)
        merge_csv(os.path.join(RESULTS_DIR, "tuning_results.csv"), tuning_df)
        best_cfg.update(new_best)
        json.dump(best_cfg, open(BEST_HP_PATH, "w"), indent=2)
        print("\nSelected configurations:")
        for k, v in new_best.items():
            print(f"  {k}: {v}")

    # ---- 2. final training ----
    time_rows = []
    for name in args.models:
        w_path = os.path.join(MODELS_DIR, f"{slug(name)}.weights.h5")
        h_path = os.path.join(RESULTS_DIR, f"history_{slug(name)}.csv")
        if os.path.exists(w_path) and os.path.exists(h_path):
            print(f"[skip] {name}: weights + history already exist")
            continue
        print(f"\n===== Training {name} with {best_cfg[name]} =====")
        keras.utils.set_random_seed(args.seed)
        m = build_model(name, **best_cfg[name])
        h, secs = compile_and_train(m, data, epochs, args, patience=args.patience, verbose=2)
        m.save_weights(w_path)
        pd.DataFrame(h.history).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(h_path, index=False)
        time_rows.append({"Model": name, "Train Time (s)": round(secs, 1), "Params": m.count_params()})
        print(f"{name}: {len(h.history['loss'])} epochs, {secs:.1f}s, {m.count_params():,} params")
        del m
        keras.backend.clear_session()
    if time_rows:
        merge_csv(os.path.join(RESULTS_DIR, "training_times.csv"), pd.DataFrame(time_rows))

    # ---- 3. convergence summary ----
    conv = []
    for name in args.models:
        h_path = os.path.join(RESULTS_DIR, f"history_{slug(name)}.csv")
        if not os.path.exists(h_path):
            continue
        hh = pd.read_csv(h_path)
        conv.append({
            "Model": name, "Epochs Run": len(hh),
            "Best Epoch (val loss)": int(hh["val_loss"].values.argmin()) + 1,
            "Final Train Acc": round(hh["accuracy"].iloc[-1], 4),
            "Final Val Acc": round(hh["val_accuracy"].iloc[-1], 4),
            "Train-Val Gap": round(hh["accuracy"].iloc[-1] - hh["val_accuracy"].iloc[-1], 4)})
    if conv:
        merge_csv(os.path.join(RESULTS_DIR, "convergence.csv"), pd.DataFrame(conv))
        print("\n" + pd.DataFrame(conv).to_string(index=False))

    # ---- 4. optional seed-variance study ----
    if args.seed_study:
        rows = []
        for name in args.models:
            accs = []
            for s in range(args.seed_runs):
                keras.utils.set_random_seed(args.seed + s)
                m = build_model(name, **best_cfg[name])
                compile_and_train(m, data, epochs, args, patience=args.patience, verbose=0)
                accs.append(accuracy_score(y_test, m.predict(x_test, batch_size=512, verbose=0).argmax(1)))
                del m
                keras.backend.clear_session()
            rows.append({"Model": name, "Runs": len(accs), "Mean Acc": np.mean(accs),
                         "Std": np.std(accs, ddof=1) if len(accs) > 1 else float("nan"),
                         "Min": np.min(accs), "Max": np.max(accs)})
            print(f"{name}: {np.mean(accs):.4f} +/- {rows[-1]['Std']:.4f}")
        merge_csv(os.path.join(RESULTS_DIR, "seed_study.csv"), pd.DataFrame(rows).round(4))


if __name__ == "__main__":
    main()
