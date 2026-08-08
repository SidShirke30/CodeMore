"""
preprocess.py

Loads the Daily Minimum Temperatures (Melbourne, Australia, 1981-1990) time
series dataset, scales it, and builds sliding-window input/target sequences
suitable for training an RNN/LSTM/GRU forecaster.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "daily-min-temperatures.csv")
CSV_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

WINDOW_SIZE = 30   # number of past days used to predict the next day
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15


def _ensure_downloaded():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)


def load_raw_series() -> pd.Series:
    """Loads the raw temperature series indexed by date."""
    _ensure_downloaded()
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df.set_index("Date")["Temp"]


def make_windows(values: np.ndarray, window_size: int):
    """
    Converts a 1D array into overlapping (X, y) sliding windows:
    X[i] = values[i : i+window_size], y[i] = values[i+window_size]
    """
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i: i + window_size])
        y.append(values[i + window_size])
    return np.array(X), np.array(y)


def get_dataloaders(batch_size: int = 32, window_size: int = WINDOW_SIZE,
                     test_fraction: float = TEST_FRACTION, val_fraction: float = VAL_FRACTION):
    """
    Loads, scales, and windows the temperature series, then returns
    (train_loader, val_loader, test_loader, scaler, raw_series).

    The split is chronological (not shuffled) since this is time series
    data — train on the earliest years, validate/test on the most recent
    years, to avoid leaking future information into training.
    """
    series = load_raw_series()
    values = series.values.astype(np.float32).reshape(-1, 1)

    n_total = len(values)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_test - n_val

    train_raw = values[:n_train]
    val_raw = values[n_train - window_size: n_train + n_val]     # include lookback context
    test_raw = values[n_train + n_val - window_size:]            # include lookback context

    # Fit the scaler ONLY on training data to avoid leakage, then apply to all splits.
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_windows(train_scaled.flatten(), window_size)
    X_val, y_val = make_windows(val_scaled.flatten(), window_size)
    X_test, y_test = make_windows(test_scaled.flatten(), window_size)

    def to_tensor_dataset(X, y):
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, window_size, 1) - 1 feature
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
        return TensorDataset(X_t, y_t)

    train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(to_tensor_dataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, series
