"""
VMD-LSTM model.

Variational Mode Decomposition + LSTM ensemble.
Decomposes the signal into Intrinsic Mode Functions (IMFs),
trains a separate LSTM on each IMF, and sums predictions.

If vmdpy is not installed, falls back to a standard LSTM.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False

from models.lstm_model import LSTMModel, train as lstm_train, predict_model


def decompose_vmd(signal, K=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    Decompose signal using VMD.

    Parameters
    ----------
    signal : np.ndarray, 1D
    K : int
        Number of modes to decompose into.

    Returns
    -------
    imfs : np.ndarray, shape (K, len(signal))
    """
    if not HAS_VMD:
        return None
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return u


def create_windows_1d(data, window_size):
    """Create sliding windows from 1D data."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


class VMDLSTM:
    """VMD-LSTM ensemble model."""

    def __init__(self, window_size, hidden_size=32, num_layers=1, K=5, seed=42):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.K = K
        self.seed = seed
        self.models = []
        self.use_vmd = HAS_VMD

    def fit(self, train_series, val_series=None, epochs=100, batch_size=32):
        """
        Train VMD-LSTM.

        Parameters
        ----------
        train_series : np.ndarray, 1D
            Training time series (already scaled).
        val_series : np.ndarray, 1D, optional
        epochs : int
        batch_size : int
        """
        self.models = []

        if self.use_vmd:
            imfs = decompose_vmd(train_series, K=self.K)
            if imfs is None:
                self.use_vmd = False

        if not self.use_vmd:
            # Fallback: single LSTM on raw series
            X, y = create_windows_1d(train_series, self.window_size)
            X_t = torch.tensor(X.reshape(-1, self.window_size, 1), dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            torch.manual_seed(self.seed)
            model = LSTMModel(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers)
            model = lstm_train(model, X_t, y_t, epochs=epochs, batch_size=batch_size, model_name="VMD-LSTM (Fallback)")
            self.models.append(model)
            return

        for k in range(self.K):
            imf = imfs[k]
            X, y = create_windows_1d(imf, self.window_size)
            X_t = torch.tensor(X.reshape(-1, self.window_size, 1), dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            torch.manual_seed(self.seed + k)
            model = LSTMModel(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers)
            model = lstm_train(model, X_t, y_t, epochs=epochs, batch_size=batch_size, model_name=f"VMD-LSTM IMF{k+1}")
            self.models.append(model)

    def predict(self, test_series):
        """
        Predict using VMD-LSTM ensemble.

        Parameters
        ----------
        test_series : np.ndarray, 1D

        Returns
        -------
        predictions : np.ndarray
        """
        if not self.use_vmd or len(self.models) == 1:
            X, _ = create_windows_1d(test_series, self.window_size)
            X_t = torch.tensor(X.reshape(-1, self.window_size, 1), dtype=torch.float32)
            return predict_model(self.models[0], X_t)

        full_series = test_series
        imfs = decompose_vmd(full_series, K=self.K)
        total_pred = None

        for k in range(self.K):
            X, _ = create_windows_1d(imfs[k], self.window_size)
            X_t = torch.tensor(X.reshape(-1, self.window_size, 1), dtype=torch.float32)
            pred = predict_model(self.models[k], X_t)
            if total_pred is None:
                total_pred = pred
            else:
                total_pred = total_pred + pred

        return total_pred