"""
edESN (Ensemble Deep Echo State Network) model.

Features:
  - Reservoir computing with random fixed weights
  - Spectral radius scaling for stability
  - Direct link (input concatenated with reservoir states)
  - Ridge regression readout
"""

import numpy as np
from sklearn.linear_model import Ridge


class edESN:
    def __init__(self, input_dim, reservoir_size=100, spectral_radius=0.9,
                 ridge_alpha=0.1, input_scaling=1.0, seed=42):
        """
        Parameters
        ----------
        input_dim : int
        reservoir_size : int
        spectral_radius : float
            Controls reservoir dynamics stability.
        ridge_alpha : float
        input_scaling : float
        seed : int
        """
        rng = np.random.RandomState(seed)

        # Input-to-reservoir weights
        self.W_in = rng.randn(input_dim, reservoir_size) * input_scaling

        # Reservoir recurrent weights (scaled by spectral radius)
        W_res = rng.randn(reservoir_size, reservoir_size)
        eigenvalues = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res * (spectral_radius / eigenvalues)

        self.bias = rng.randn(reservoir_size) * 0.1
        self.ridge_alpha = ridge_alpha
        self.model = None

    def _reservoir(self, X):
        """Compute reservoir states with direct link."""
        H = np.tanh(X @ self.W_in + self.bias)
        return H

    def fit(self, X, y):
        """Train ridge regression on [reservoir_states | input]."""
        H = self._reservoir(X)
        # Direct link: concatenate reservoir states with original input
        D = np.concatenate([H, X], axis=1)
        self.model = Ridge(alpha=self.ridge_alpha)
        self.model.fit(D, y)

    def predict(self, X):
        """Predict using trained model."""
        H = self._reservoir(X)
        D = np.concatenate([H, X], axis=1)
        return self.model.predict(D)


# Hyperparameter search space
SEARCH_SPACE = {
    'reservoir_size': {'low': 20, 'high': 200},
    'spectral_radius': {'low': 0.1, 'high': 1.0},
    'ridge_alpha': {'low': 1e-6, 'high': 1.0, 'log': True},
    'input_scaling': {'low': 0.01, 'high': 1.0},
}