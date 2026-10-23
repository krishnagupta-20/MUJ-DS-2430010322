"""Train the Swin-style transformer. Run from the project root:  python src/train.py [--tune] [--quick]"""
import argparse, itertools, json, math, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import callbacks
from model import build_swin
from preprocessing import load_data, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY = 128, 1e-3, 0.05


def make_optimizer(epochs, n_train):
    steps = math.ceil(n_train / BATCH_SIZE) * epochs
    warm = max(1, int(0.05 * steps))
    sched = keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=max(1, steps - warm), alpha=0.01,
        warmup_target=LEARNING_RATE, warmup_steps=warm)
    opt = keras.optimizers.AdamW(learning_rate=sched, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
    opt.exclude_from_weight_decay(var_names=['bias', 'gamma', 'beta', 'rel_bias_table'])
    return opt


def fit(model, data, epochs, patience=6, verbose=2):
    x_train, x_val, y_train, y_val = data[:4]
    model.compile(optimizer=make_optimizer(epochs, len(x_train)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cbs = [callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                   restore_best_weights=True)] if patience else []
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=BATCH_SIZE, callbacks=cbs, verbose=verbose)
    return hist, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', action='store_true', help='short sweep over dropout / drop-path (selects on validation accuracy)')
    ap.add_argument('--quick', action='store_true', help='tiny subset, 1-2 epochs, pipeline check only')
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--drop-path', type=float, default=0.1)
    a = ap.parse_args()

    epochs = 2 if a.quick else 30
    tune_epochs = 1 if a.quick else 6
    data = load_data(quick=a.quick)
    hp = {'dropout': a.dropout, 'drop_path': a.drop_path}

    if a.tune:
        best = -1.0
        for d, p in itertools.product([0.0, 0.1], [0.0, 0.1]):
            keras.utils.set_random_seed(SEED)
            m = build_swin(name='tune', dropout=d, drop_path=p)
            h, _ = fit(m, data, tune_epochs, patience=None, verbose=0)
            acc = max(h.history['val_accuracy'])
            print(f"dropout={d} drop_path={p} val_acc={acc:.4f}")
            if acc > best:
                best, hp = acc, {'dropout': d, 'drop_path': p}
            keras.backend.clear_session()
    print("Selected hyper-parameters:", hp)

    keras.utils.set_random_seed(SEED)
    model = build_swin(name='Swin_Tiny', **hp)
    hist, secs = fit(model, data, epochs)
    print(f"Trained {len(hist.history['loss'])} epochs in {secs:.1f}s, {model.count_params():,} params")

    os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    model.save_weights(os.path.join(ROOT, 'models', 'swin_tiny.weights.h5'))
    with open(os.path.join(ROOT, 'models', 'config.json'), 'w') as f:
        json.dump({**hp, 'epochs_run': len(hist.history['loss']), 'train_seconds': round(secs, 1),
                   'params': int(model.count_params())}, f, indent=2)
    with open(os.path.join(ROOT, 'results', 'history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history['loss'], '--', label='train'); ax[0].plot(hist.history['val_loss'], label='validation')
    ax[0].set_title('Loss'); ax[0].set_xlabel('Epoch'); ax[0].legend()
    ax[1].plot(hist.history['accuracy'], '--', label='train'); ax[1].plot(hist.history['val_accuracy'], label='validation')
    ax[1].set_title('Accuracy (train is measured on augmented images)'); ax[1].set_xlabel('Epoch'); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'results', 'training_curve.png'), dpi=150)
    print("Saved weights, config.json, history.json and results/training_curve.png")


if __name__ == '__main__':
    main()
