"""
Architecture Module — RedRVFL core components.

Defines the RandomLSTM feature extractor and the construction
of the RVFL feature matrix used by the RedRVFL architecture.

Data Flow:
  1. Input: (batch, window, features)
  2. RandomLSTM (frozen weights) → hidden: (batch, hidden_size)
  3. Feature matrix D = [hidden | flattened_input]
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cpu")

class RandomLSTM(nn.Module):
    """
    Random LSTM feature extractor with frozen weights.

    Weights are randomly initialized and never trained.
    Acts as a fixed nonlinear feature extractor.
    """

    def __init__(self, input_size, hidden_size, input_scaling=1.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        ).to(device)

        # Scale weights by input_scaling
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    param.mul_(input_scaling)

        # Freeze all LSTM parameters
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x):
        x = x.to(device)
        with torch.no_grad():
            _, (h_n, _) = self.lstm(x)

        # h_n shape: (num_layers, batch, hidden_size)
        # Select last layer hidden state
        return h_n[-1]  # shape: (batch, hidden_size)


def flatten_window(X):
    """
    Flatten input time window.

    Input:  (batch, window, features) → Output: numpy (batch, window * features)
    """
    batch_size = X.shape[0]
    xflat = X.reshape(batch_size, -1)
    if isinstance(xflat, torch.Tensor):
        return xflat.detach().cpu().numpy()
    return np.asarray(xflat)


def build_layer_input(previous_hidden, xtensor):
    """
    Construct input for deeper RedRVFL layers.

    Layer l receives: [h_(l-1) expanded across time , X]

    Parameters
    ----------
    previous_hidden : torch.Tensor, (batch, hidden_size)
    xtensor : torch.Tensor, (batch, window, features)

    Returns
    -------
    torch.Tensor, (batch, window, features + hidden_size)
    """
    xtensor = xtensor.to(device)
    previous_hidden = previous_hidden.to(device)
    
    window_size = xtensor.shape[1]
    hidden_expanded = previous_hidden.unsqueeze(1).repeat(1, window_size, 1)
    return torch.cat([xtensor, hidden_expanded], dim=2)


def build_feature_matrix(xtensor, hidden_tensor):
    """
    Build RVFL feature matrix: D = [hidden_features | flattened_input]

    Parameters
    ----------
    xtensor : torch.Tensor, (batch, window, features)
    hidden_tensor : torch.Tensor, (batch, hidden_size)

    Returns
    -------
    D : np.ndarray, (batch, hidden_size + window * features)
    """
    xflat = flatten_window(xtensor)
    if isinstance(hidden_tensor, torch.Tensor):
        hidden_np = hidden_tensor.detach().cpu().numpy()
    else:
        hidden_np = np.asarray(hidden_tensor)
    return np.concatenate([hidden_np, xflat], axis=1)