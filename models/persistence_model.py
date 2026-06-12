"""Persistence (naive) baseline model.

Predicts the last value in the input window.
"""

import numpy as np


def predict(X):
    """
    Persistence forecast: predict last known value.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, window_size, 1) or (n_samples, window_size)

    Returns
    -------
    predictions : np.ndarray, shape (n_samples,)
    """
    if X.ndim == 3:
        return X[:, -1, 0]
    return X[:, -1]