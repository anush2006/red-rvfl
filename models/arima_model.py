"""
ARIMA baseline model.

Uses fixed order (1,1,1) as a simple statistical baseline.
ARIMA operates on raw series, not windowed data.
"""

import warnings
from statsmodels.tsa.arima.model import ARIMA


def train(series, order=(1, 1, 1)):
    """
    Fit ARIMA model on training series.

    Parameters
    ----------
    series : np.ndarray, 1D
        Training time series.
    order : tuple
        (p, d, q) order.

    Returns
    -------
    fitted model
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        return model.fit()


def predict(model, steps):
    """
    Forecast future values.

    Parameters
    ----------
    model : fitted ARIMA
    steps : int
        Number of steps to forecast.

    Returns
    -------
    predictions : np.ndarray
    """
    return model.forecast(steps=steps)