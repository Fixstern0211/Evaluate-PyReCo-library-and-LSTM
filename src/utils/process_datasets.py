import random
import numpy as np

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def split_datasets(arr: np.ndarray, frac: float):
    T = len(arr)
    split = int(T * frac)
    return arr[:split], arr[split:]

def sliding_window(dataset: np.ndarray, n_in: int, n_out: int = 1):
    T, D = dataset.shape
    B = T - n_in - n_out + 1
    X = np.stack([dataset[i:i + n_in] for i in range(B)], axis=0)
    y = np.stack([dataset[i + n_in:i + n_in + n_out] for i in range(B)], axis=0)
    return X, y
