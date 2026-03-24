from reservoirpy import datasets
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
    
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


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x.astype(np.float64, copy=False)

def _normalize_name(name: str) -> str:
    name = name.lower()
    if name in {"lorenz", "lorenz63"}:
        return "lorenz"
    if name in {"mackeyglass", "mackey-glass", "mg"}:
        return "mackeyglass"
    if name in {"santafe", "santa_fe", "laser", "santafe-laser", "santa_fe_laser"}:
        return "santafe"
    raise ValueError(f"Unknown dataset name: {name}")

def load_data(
    name: str,
    length: int,
    *,
    seed: int,
    random_window: bool = False,
    noise: float = 0.0,
    **kwargs
):
    """
    统一调取三类数据：
      - name: 'lorenz' | 'mackeyglass' | 'santafe'（支持常见别名）
      - length: 期望长度。lorenz/mackeyglass 会用它生成；santafe 为固定长度，若传入更小会裁剪，传入更大将报错
      - seed: 仅用于需要随机窗口/噪声的本函数，或者传递给 mackey_glass_series（其本身支持 seed）
      - random_window: True 时在可行的情况下随机截取窗口（需要 length 不为 None 且 length <= 原始长度）
      - noise: 为数据加零均值高斯噪声（标准差 = noise）；0 表示不加
      - **kwargs: 透传到具体数据函数，例如 rho/sigma/beta/h/x0（lorenz），tau 等（mackey-glass）
    返回：
      data: np.ndarray, shape (T, D)
      meta: dict，包含 name/Dout/length/seed/source 等
    """
    rng = np.random.default_rng(seed)
    ds = _normalize_name(name)

    if ds == "lorenz":
        # lorenz 接受 n_timesteps
        nt = length if length is not None else kwargs.pop("n_timesteps", 15000)
        data = lorenz63_dataset(n_timesteps=nt, **kwargs)  # (T,3)

    elif ds == "mackeyglass":
        # mackey glass 接受 n_timesteps 和 seed
        nt = length if length is not None else kwargs.pop("n_timesteps", 20000)
        # 将 seed 传进包装函数，以保证可复现（如不需要可移除）
        data = mackey_glass_dataset(n_timesteps=nt, seed=seed, **kwargs)  # (T,1 or T,?)

    elif ds == "santafe":
        # 固定长度数据集
        full = santafe_dataset()  # (T_full, 1)
        T_full = full.shape[0]
        if length is None or length == T_full:
            data = full
        elif length < T_full:
            if random_window:
                start = int(rng.integers(0, T_full - length + 1))
            else:
                start = 0
            data = full[start:start + length]
        else:
            raise ValueError(f"Requested length {length} > Santa Fe length {T_full}. "
                             f"Choose length <= {T_full} or turn to a different dataset.")
    else:
        raise AssertionError("unreachable")

    data = _ensure_2d(data)

    if noise and noise > 0.0:
        data = data + rng.normal(0.0, noise, size=data.shape)

    meta = {
        "name": ds,
        "length": int(data.shape[0]),
        "Dout": int(data.shape[1]),
        "seed": seed,
        "random_window": bool(random_window),
        "noise": float(noise),
        "source": "reservoirpy.datasets",
    }
    return data, meta


def _sliding_window(data: np.ndarray, n_in: int, n_out: int = 1):
    """
    Create sliding windows for time series prediction.

    Parameters:
        data: (T, D) array
        n_in: number of input timesteps (lookback window)
        n_out: number of output timesteps (prediction horizon)

    Returns:
        X: (N, n_in, D) input windows
        Y: (N, n_out, D) target windows
    """
    T, D = data.shape
    N = T - n_in - n_out + 1

    if N <= 0:
        raise ValueError(f"Data too short: {T} timesteps, need at least {n_in + n_out}")

    X = np.stack([data[i:i + n_in] for i in range(N)], axis=0)
    Y = np.stack([data[i + n_in:i + n_in + n_out] for i in range(N)], axis=0)

    return X, Y


def load(
    name: str,
    n_samples: int = 5000,
    *,
    seed: int = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    n_in: int = 100,
    n_out: int = 1,
    standardize: bool = True,
    **kwargs
):
    """
    Unified data loading function with same interface as pyreco.datasets.load().

    Supports: 'lorenz', 'mackeyglass', 'santafe' (and their aliases)

    Parameters:
        name: dataset name ('lorenz', 'mackeyglass', 'santafe')
        n_samples: number of timesteps to generate/use
        seed: random seed for reproducibility
        train_fraction: fraction of data for training (default 0.7)
        val_fraction: fraction of data for validation (default 0.15)
        n_in: input window size (default 100)
        n_out: output window size (default 1)
        standardize: whether to standardize the data (default True)
        **kwargs: additional arguments passed to dataset generators

    Returns:
        X_train, Y_train: training windows (N_train, n_in, D), (N_train, n_out, D)
        X_val, Y_val: validation windows
        X_test, Y_test: test windows
        scaler: fitted StandardScaler (or None if standardize=False)

    Example:
        >>> Xtr, Ytr, Xva, Yva, Xte, Yte, scaler = load(
        ...     'santafe', n_samples=5000, seed=42,
        ...     train_fraction=0.7, val_fraction=0.15, n_in=100
        ... )
    """
    # Load raw data
    data, meta = load_data(name, length=n_samples, seed=seed, **kwargs)
    n_total = len(data)

    # Calculate split sizes (matching pyreco.datasets.load semantics):
    # train_fraction: fraction of TOTAL data for train+val
    # val_fraction: fraction of TRAIN portion used for validation
    # test: everything after train+val
    n_trainval = int(n_total * train_fraction)
    n_test = n_total - n_trainval
    n_val = int(n_trainval * val_fraction)
    n_train = n_trainval - n_val

    if n_test <= 0:
        raise ValueError(
            f"Invalid split: train_fraction={train_fraction} "
            f"leaves no data for test."
        )

    # Split data: train | val | test (all chronological)
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Check sufficient data for windows
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        min_len = n_in + n_out
        if len(split_data) < min_len:
            raise ValueError(
                f"{split_name} data has only {len(split_data)} timesteps but needs at least "
                f"{min_len} (n_in={n_in} + n_out={n_out}). "
                f"Increase n_samples or adjust train_fraction/val_fraction."
            )

    # Standardize (fit on train only)
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)

    # Create sliding windows
    X_train, Y_train = _sliding_window(train_data, n_in, n_out)
    X_val, Y_val = _sliding_window(val_data, n_in, n_out)
    X_test, Y_test = _sliding_window(test_data, n_in, n_out)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler

