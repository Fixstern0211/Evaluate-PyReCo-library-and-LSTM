import random
import numpy as np

def split_datasets(arr: np.ndarray, train_frac: float):
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    T = len(arr)
    if T == 0:
        raise ValueError("Empty array.")

    split = int(T * train_frac)
    train, test = arr[:split], arr[split:]
    return train, test, split  # split is the split point (start index of test set)


def sliding_window(dataset: np.ndarray, n_in: int, n_out: int = 1):
    T, D = dataset.shape
    B = T - n_in - n_out + 1
    X = np.stack([dataset[i:i + n_in] for i in range(B)], axis=0)
    y = np.stack([dataset[i + n_in:i + n_in + n_out] for i in range(B)], axis=0)
    return X, y

from typing import List, Tuple
# ============================================================================
# cross validation function for time series data - specifically designed for time series data
# ============================================================================
def timeseries_cv_split(X: np.ndarray, y: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Forward-chaining time series cross-validation split (Time Series Split)

    Preserves temporal order to avoid using "future" data to predict the "past".

    Unlike PyReCo's built-in cross_val, this function does not shuffle the time order.

    Example (n_splits=4):
        Fold 1: train [0:20%]   → val [20%:25%]
        Fold 2: train [0:40%]   → val [40%:45%]
        Fold 3: train [0:60%]   → val [60%:65%]
        Fold 4: train [0:80%]   → val [80%:85%]

    Parameters:
        X: feature matrix (n_samples, n_features)
        y: target matrix (n_samples, n_outputs)
        n_splits: number of folds

    Returns:
        List of tuples: [(X_train_1, y_train_1, X_val_1, y_val_1), ...]
    """
    n_samples = X.shape[0]
    splits = []

    # calculate the validation set size for each fold (approximately 5% of total data)
    val_size = max(1, n_samples // (n_splits * 4))

    for i in range(1, n_splits + 1):
        # training set: from start to current fold position
        train_end = (n_samples * i) // (n_splits + 1)

        # validation set: a small segment immediately following the training set
        val_start = train_end
        val_end = min(val_start + val_size, n_samples)

        # ensure there is at least some data
        if train_end < 1 or val_end <= val_start:
            continue

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        splits.append((X_train, y_train, X_val, y_val))

    return splits