"""
RedRVFL Orchestrator — manages multi-layer RandomLSTM pipeline.

Responsibilities:
  - Manage multiple RandomLSTM layers
  - Construct feature matrices per layer
  - Train ridge regression per layer
  - Generate final predictions via MEDIAN aggregation
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.architecture import RandomLSTM, build_feature_matrix, build_layer_input

device = torch.device("cpu")

class RedRVFLOrchestrator:
    """
    Orchestrates the RedRVFL architecture.

    Each layer:
      1. RandomLSTM extracts hidden features (frozen weights)
      2. Feature matrix D = [hidden | flattened_input]
      3. Ridge regression trained on D → y
      4. Next layer input = [current_hidden expanded | original_input]

    Final prediction = MEDIAN across all layer predictions.
    """

    def __init__(self, input_features, hidden_size, num_layers,
                 input_scaling=1.0, seed=42):
        """
        Parameters
        ----------
        input_features : int
            Number of features per time step (1 for univariate).
        hidden_size : int
            LSTM hidden state dimensionality.
        num_layers : int
            Number of stacked RandomLSTM layers.
        input_scaling : float
            Scaling factor for random weights.
        seed : int
            Base seed for reproducibility.
        """
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_scaling = input_scaling
        self.seed = seed

        self.layers = []
        for i in range(num_layers):
            torch.manual_seed(seed + i)
            if i == 0:
                input_size = input_features
            else:
                input_size = input_features + hidden_size

            lstm = RandomLSTM(input_size, hidden_size, input_scaling).to(device)
            self.layers.append(lstm)

    def extract_features(self, X_tensor):
        """
        Extract feature matrices for all layers.

        Parameters
        ----------
        X_tensor : torch.Tensor, shape (n, window, features)

        Returns
        -------
        feature_matrices : list of np.ndarray
            Each element has shape (n, hidden_size + window * features)
        """
        feature_matrices = []
        xtensor_current = X_tensor.to(device)

        for i in range(self.num_layers):
            lstm = self.layers[i]
            hidden = lstm(xtensor_current)
            D = build_feature_matrix(X_tensor, hidden)
            feature_matrices.append(D)

            if i < self.num_layers - 1:
                xtensor_current = build_layer_input(hidden, xtensor_current)

        return feature_matrices

    def fit(self, X_tensor, y, ridge_alpha=0.1):
        """
        Train ridge regression models for each layer.

        Parameters
        ----------
        X_tensor : torch.Tensor, shape (n, window, features)
        y : np.ndarray, shape (n,)
        ridge_alpha : float

        Returns
        -------
        self (for chaining)
        """
        feature_matrices = self.extract_features(X_tensor)
        self.ridge_models = []

        for D in feature_matrices:
            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(D, y)
            self.ridge_models.append(ridge)

        return self

    def predict(self, X_tensor, ridge_models=None):
        """
        Generate predictions using MEDIAN aggregation across layers.

        Parameters
        ----------
        X_tensor : torch.Tensor
        ridge_models : list of Ridge, optional
            If None, uses self.ridge_models from fit().

        Returns
        -------
        predictions : np.ndarray
        """
        if ridge_models is None:
            ridge_models = self.ridge_models

        feature_matrices = self.extract_features(X_tensor)
        predictions = []

        for i, ridge_model in enumerate(ridge_models):
            D = feature_matrices[i]
            pred = ridge_model.predict(D)
            predictions.append(pred)

        predictions = np.array(predictions)
<<<<<<< HEAD
        # MEDIAN aggregation (paper requirement)
        return np.median(predictions, axis=0)
=======
        final_prediction = np.median(predictions, axis=0)

        return final_prediction
>>>>>>> cc49257403b50856bc14abaf963ee5b72c1822ab
