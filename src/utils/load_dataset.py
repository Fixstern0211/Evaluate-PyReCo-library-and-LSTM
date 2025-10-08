from reservoirpy import datasets
import numpy as np
import random

def lorenz63_dataset(n_timesteps=15000, rho=28.0, sigma=10.0, beta=8.0/3.0, h=0.01, x0=(1.0,1.0,1.0)):
    """Wrapper around reservoirpy.datasets.lorenz. Returns (T, 3)."""
    arr = datasets.lorenz(n_timesteps=n_timesteps, rho=rho, sigma=sigma, beta=beta, h=h, x0=list(x0))
    return np.asarray(arr, dtype=float)


def mackey_glass_dataset(n_timesteps=20000, tau=17, a=0.2, b=0.1, n=10, x0=1.2, h=1.0, seed=None):
    """Wrapper around reservoirpy.datasets.mackey_glass. Returns (T, 1)."""
    arr = datasets.mackey_glass(n_timesteps=n_timesteps, tau=tau, a=a, b=b, n=n, x0=x0, h=h, seed=seed)
    return np.asarray(arr, dtype=float)


def santafe_dataset():
    """Load the canonical Santa-Fe laser dataset (~10093 steps)."""
    arr = datasets.santafe_laser()
    return np.asarray(arr, dtype=float).reshape(-1, 1)


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


