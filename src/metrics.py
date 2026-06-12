"""
Evaluation metrics for time series forecasting.

All metrics operate on arrays already in original price space
(inverse transform should be applied BEFORE calling these).
"""

import numpy as np


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    # Protect against division by zero
    mask = np.abs(y_true) > 1e-8
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))