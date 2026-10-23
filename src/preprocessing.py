"""Data loading, normalisation and the stratified train/validation split (identical to the notebook)."""
import numpy as np
import keras
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data(seed=SEED, quick=False):
    (x_full, y_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_full = np.expand_dims(x_full.astype('float32') / 255.0, -1)
    x_test = np.expand_dims(x_test.astype('float32') / 255.0, -1)
    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=0.10, random_state=seed, stratify=y_full)
    if quick:   # tiny subsets, only to check the pipeline runs
        x_train, y_train = x_train[:3000], y_train[:3000]
        x_val, y_val = x_val[:600], y_val[:600]
        x_test, y_test = x_test[:1000], y_test[:1000]
    return x_train, x_val, y_train, y_val, x_test, y_test
