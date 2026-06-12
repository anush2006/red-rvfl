"""
RVFL (Random Vector Functional Link) model.

Paper-aligned:
  hidden_nodes: [20-200]
  regularization lambda: [0-1]
  input_scaling: [0-1]
"""

import numpy as np
from sklearn.linear_model import Ridge


class RVFL:
    def __init__(self, input_dim, hidden_dim=100, ridge_alpha=0.1, input_scaling=1.0, seed=42):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features (window_size for flattened data).
        hidden_dim : int
            Number of hidden nodes.
        ridge_alpha : float
            Ridge regression regularization parameter.
        input_scaling : float
            Scaling factor for random weights.
        seed : int
            Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        self.W = rng.randn(input_dim, hidden_dim) * input_scaling
        self.b = rng.randn(hidden_dim) * input_scaling
        self.ridge_alpha = ridge_alpha
        self.model = None

    def _hidden(self, X):
        """Compute hidden layer activations with direct link."""
        return np.tanh(X @ self.W + self.b)

    def fit(self, X, y):
        """Fit ridge regression on [hidden | input] features."""
        H = self._hidden(X)
        D = np.concatenate([H, X], axis=1)  # Direct link
        self.model = Ridge(alpha=self.ridge_alpha)
        self.model.fit(D, y)

    def predict(self, X):
        """Predict using trained model."""
        H = self._hidden(X)
        D = np.concatenate([H, X], axis=1)
        return self.model.predict(D)


# Hyperparameter search space
SEARCH_SPACE = {
    'hidden_dim': {'low': 20, 'high': 200},
    'ridge_alpha': {'low': 1e-6, 'high': 1.0, 'log': True},
    'input_scaling': {'low': 0.01, 'high': 1.0},
}