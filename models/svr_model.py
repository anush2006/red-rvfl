"""
SVR baseline model with paper-aligned hyperparameters.

Paper ranges:
  C: [2^-10 to 2^0]
  epsilon: [0.001, 0.01, 0.1]
  gamma: scale, auto
"""

import numpy as np
from sklearn.svm import SVR


def create_model(C=0.1, epsilon=0.01, gamma='scale'):
    """Create SVR model with given hyperparameters."""
    return SVR(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')


def train(X, y, C=0.1, epsilon=0.01, gamma='scale'):
    """Train SVR model. X should be 2D (n_samples, n_features)."""
    model = create_model(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X, y.ravel())
    return model


def predict(model, X):
    """Predict using trained SVR. X should be 2D."""
    return model.predict(X)


# Hyperparameter search space for Optuna
SEARCH_SPACE = {
    'C': {'low': 2**-10, 'high': 2**0, 'log': True},
    'epsilon': [0.001, 0.01, 0.1],
    'gamma': ['scale', 'auto'],
}