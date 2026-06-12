"""
EWT-edRVFL model.

Empirical Wavelet Transform + edRVFL ensemble.
Decomposes the signal using EWT, trains edRVFL on each component,
and sums predictions.

If ewtpy is not installed, falls back to a standard edRVFL.
"""

import numpy as np

try:
    import ewtpy
    HAS_EWT = True
except ImportError:
    HAS_EWT = False

from models.edrvfl_model import edRVFL


def decompose_ewt(signal, N=5):
    """Decompose signal using EWT."""
    if not HAS_EWT:
        return None
    try:
        ewt, _, _ = ewtpy.EWT1D(signal, N=N)
        return ewt.T
    except Exception:
        return None


def create_windows_1d(data, window_size):
    """Create sliding windows from 1D data."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


class EWTedRVFL:
    """EWT-edRVFL ensemble model."""

    def __init__(self, window_size, hidden_dim=100, num_layers=3,
                 ridge_alpha=0.1, input_scaling=1.0, N=5, seed=42):
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ridge_alpha = ridge_alpha
        self.input_scaling = input_scaling
        self.N = N
        self.seed = seed
        self.models = []
        self.use_ewt = HAS_EWT

    def fit(self, train_series, y_train_unused=None):
        """Train EWT-edRVFL."""
        self.models = []

        if self.use_ewt:
            components = decompose_ewt(train_series, N=self.N)
            if components is None:
                self.use_ewt = False

        if not self.use_ewt:
            X, y = create_windows_1d(train_series, self.window_size)
            model = edRVFL(
                input_dim=self.window_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                ridge_alpha=self.ridge_alpha,
                input_scaling=self.input_scaling,
                seed=self.seed
            )
            model.fit(X, y)
            self.models.append(model)
            return

        for k in range(components.shape[0]):
            comp = components[k]
            X, y = create_windows_1d(comp, self.window_size)
            model = edRVFL(
                input_dim=self.window_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                ridge_alpha=self.ridge_alpha,
                input_scaling=self.input_scaling,
                seed=self.seed + k
            )
            model.fit(X, y)
            self.models.append(model)

    def predict(self, test_series):
        """Predict using EWT-edRVFL ensemble (sum of component predictions)."""
        if not self.use_ewt or len(self.models) == 1:
            X, _ = create_windows_1d(test_series, self.window_size)
            return self.models[0].predict(X)

        components = decompose_ewt(test_series, N=self.N)
        if components is None:
            X, _ = create_windows_1d(test_series, self.window_size)
            return self.models[0].predict(X)

        total_pred = None
        for k in range(min(components.shape[0], len(self.models))):
            X, _ = create_windows_1d(components[k], self.window_size)
            pred = self.models[k].predict(X)
            if total_pred is None:
                total_pred = pred
            else:
                total_pred = total_pred + pred

        return total_pred