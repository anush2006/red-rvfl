"""
edRVFL (Ensemble Deep Random Vector Functional Link) model.

Paper-aligned:
  hidden_nodes: [20-200]
  layers: [1-12]
  regularization lambda: [0-1]
  input_scaling: [0-1]
  Aggregation: MEDIAN across layers
"""

import numpy as np
from sklearn.linear_model import Ridge


class edRVFL:
    def __init__(self, input_dim, hidden_dim=100, num_layers=3,
                 ridge_alpha=0.1, input_scaling=1.0, seed=42):
        """
        Parameters
        ----------
        input_dim : int
        hidden_dim : int
        num_layers : int
        ridge_alpha : float
        input_scaling : float
        seed : int
        """
        self.num_layers = num_layers
        self.ridge_alpha = ridge_alpha
        self.hidden_dim = hidden_dim

        rng = np.random.RandomState(seed)

        # Layer 0: input_dim -> hidden_dim
        self.W = [rng.randn(input_dim, hidden_dim) * input_scaling]
        self.b = [rng.randn(hidden_dim) * input_scaling]

        # Layers 1+: (input_dim + hidden_dim) -> hidden_dim
        for _ in range(1, num_layers):
            self.W.append(rng.randn(input_dim + hidden_dim, hidden_dim) * input_scaling)
            self.b.append(rng.randn(hidden_dim) * input_scaling)

        self.models = []

    def fit(self, X, y):
        """Train ridge regression at each layer."""
        self.models = []
        H_prev = None

        for i in range(self.num_layers):
            if H_prev is None:
                H = np.tanh(X @ self.W[i] + self.b[i])
            else:
                combined = np.concatenate([H_prev, X], axis=1)
                H = np.tanh(combined @ self.W[i] + self.b[i])

            D = np.concatenate([H, X], axis=1)  # Direct link
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(D, y)
            self.models.append(model)
            H_prev = H

    def predict(self, X):
        """Predict using median aggregation across layers."""
        preds = []
        H_prev = None

        for i in range(self.num_layers):
            if H_prev is None:
                H = np.tanh(X @ self.W[i] + self.b[i])
            else:
                combined = np.concatenate([H_prev, X], axis=1)
                H = np.tanh(combined @ self.W[i] + self.b[i])

            D = np.concatenate([H, X], axis=1)
            preds.append(self.models[i].predict(D))
            H_prev = H

        return np.median(preds, axis=0)


# Hyperparameter search space
SEARCH_SPACE = {
    'hidden_dim': {'low': 20, 'high': 200},
    'num_layers': {'low': 1, 'high': 12},
    'ridge_alpha': {'low': 1e-6, 'high': 1.0, 'log': True},
    'input_scaling': {'low': 0.01, 'high': 1.0},
}