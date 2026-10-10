"""Data loading, preprocessing, stratified split and dataset analysis for Fashion-MNIST.

Run `python src/preprocessing.py` to regenerate the dataset-analysis figures/tables.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
MODELS_DIR = os.path.join(ROOT, "models")


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)


def load_raw():
    """Fashion-MNIST via keras (downloaded and cached on first use)."""
    import keras
    return keras.datasets.fashion_mnist.load_data()


def preprocess(x):
    """uint8 (N,28,28) -> float32 (N,28,28,1) scaled to [0, 1]."""
    return np.expand_dims(x.astype("float32") / 255.0, -1)


def get_splits(quick=False, seed=SEED, val_size=0.10):
    """Return x_train, y_train, x_val, y_val, x_test, y_test.

    54,000 train / 6,000 validation (stratified) / 10,000 test.
    quick=True uses tiny subsets, only to check the pipeline runs end-to-end.
    """
    (x_full, y_full), (x_test, y_test) = load_raw()
    x_full, x_test = preprocess(x_full), preprocess(x_test)
    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=val_size, random_state=seed, stratify=y_full)
    if quick:
        x_train, y_train = x_train[:3000], y_train[:3000]
        x_val, y_val = x_val[:600], y_val[:600]
        x_test, y_test = x_test[:1000], y_test[:1000]
    return x_train, y_train, x_val, y_val, x_test, y_test


def dataset_analysis():
    """Class distribution table + plot, imbalance ratio, sample-image grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dirs()
    (x_full, y_full), _ = load_raw()
    unique, counts = np.unique(y_full, return_counts=True)
    dist = pd.DataFrame({"Class": [CLASS_NAMES[i] for i in unique], "Count": counts})
    dist.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"), index=False)

    ratio = counts.max() / counts.min()
    print(dist.to_string(index=False))
    print(f"Imbalance ratio (max/min): {ratio:.3f} -> "
          f"{'balanced' if ratio < 1.5 else 'imbalanced'}")

    plt.figure(figsize=(9, 4))
    sns.barplot(data=dist, x="Class", y="Count", hue="Class", legend=False, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution - Fashion-MNIST Training Set")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    for i in range(20):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_full[i], cmap="gray")
        plt.title(CLASS_NAMES[y_full[i]], fontsize=8)
        plt.axis("off")
    plt.suptitle("Sample Images from Fashion-MNIST")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sample_images.png"), dpi=150)
    plt.close()
    print("Saved figures/class_distribution.png and figures/sample_images.png")


if __name__ == "__main__":
    dataset_analysis()
