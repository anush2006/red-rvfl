"""
Data utilities for financial time series forecasting.

Handles:
- Dataset loading from Excel files
- Sliding window creation
- Sequential train/val/test splitting (NO shuffle)
- MinMax scaling fitted on TRAINING data only
- Proper inverse transformation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_dataset(path):
    """
    Load financial dataset from Excel file.

    Parameters
    ----------
    path : str
        Path to .xlsx file with Date and Close columns.

    Returns
    -------
    prices : np.ndarray
        1D array of closing prices.
    """
    df = pd.read_excel(path, skiprows=2, engine="openpyxl")
    df.columns = ['Date', 'Close']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    return df['Close'].values


def create_windows(data, window_size):
    """
    Create sliding window features for time series.

    Parameters
    ----------
    data : np.ndarray
        1D array of scaled prices.
    window_size : int
        Number of past time steps to use as features.

    Returns
    -------
    X : np.ndarray, shape (n_samples, window_size, 1)
        Windowed input features.
    y : np.ndarray, shape (n_samples,)
        Target values (next time step).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    X = np.array(X)
    y = np.array(y).ravel()

    # Ensure 3D shape for time series models: (samples, timesteps, features)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.1):
    """
    Sequential split for time series (NO shuffle).

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    train_ratio : float
    val_ratio : float

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
    }


def fit_scaler(X_train, y_train):
    """
    Fit MinMaxScaler on training data only.

    Parameters
    ----------
    X_train : np.ndarray, shape (n, window, 1)
    y_train : np.ndarray, shape (n,)

    Returns
    -------
    scaler : MinMaxScaler
        Fitted scaler (on all training values combined).
    """
    # Combine all training values to get global min/max from train
    all_train_values = np.concatenate([
        X_train.reshape(-1),
        y_train.reshape(-1)
    ]).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_train_values)
    return scaler


def transform_data(X, y, scaler):
    """
    Apply fitted scaler to data.

    Parameters
    ----------
    X : np.ndarray, shape (n, window, 1)
    y : np.ndarray, shape (n,)
    scaler : MinMaxScaler

    Returns
    -------
    X_scaled : np.ndarray, same shape as X
    y_scaled : np.ndarray, same shape as y
    """
    orig_shape = X.shape
    X_flat = X.reshape(-1, 1)
    X_scaled = scaler.transform(X_flat).reshape(orig_shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).ravel()
    return X_scaled, y_scaled


def inverse_transform(y, scaler):
    """
    Inverse transform scaled predictions back to original price space.

    Parameters
    ----------
    y : np.ndarray
    scaler : MinMaxScaler

    Returns
    -------
    y_original : np.ndarray
    """
    return scaler.inverse_transform(y.reshape(-1, 1)).ravel()
