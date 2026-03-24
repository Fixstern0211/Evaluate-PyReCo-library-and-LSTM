import src.utils.load_dataset as load_data
import src.utils.process_datasets as process_datasets
from sklearn.preprocessing import StandardScaler
import argparse
from pyreco import datasets
import numpy as np

def main():

    parser = argparse.ArgumentParser(description="Process time series dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--length", type=int, default=1000, help="Length of the time series")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction of data to use for training")
    parser.add_argument("--n_in", type=int, default=10, help="Number of input time steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    data, meta = load_data(args.dataset, length=args.length, seed=args.seed)
    Dout = meta["Dout"]
    print(f"Data shape: {data.shape}, Output dimension: {Dout}")

    # Split
    train, test, split = process_datasets.split_datasets(data, args.train_frac)
    n_tr = int(0.85 * len(train))
    series_train = train[:n_tr]

    print(f"Train: {series_train.shape}, Test: {test.shape}")

    # Standardize
    scaler = StandardScaler().fit(series_train)
    series_train = scaler.transform(series_train)
    series_test = scaler.transform(test)

    # Create windows
    Xtr, Ytr = process_datasets.sliding_window(series_train, args.n_in, 1)
    Xte, Yte = process_datasets.sliding_window(series_test, args.n_in, 1)

    pyreco_Xtr, pyreco_Ytr, pyreco_Xte, pyreco_Yte = datasets.load('lorenz63', n_samples=args.length, train_fraction=args.train_frac, n_in=args.n_in, n_out=1, seed=args.seed)