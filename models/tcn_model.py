"""
TCN (Temporal Convolutional Network) model for time series forecasting.

Paper-aligned:
  filters: [4, 8, 16, 32, 64]
  kernel_size: [1, 2, 3]
  optimizer: Adam, lr=0.001
  epochs: 100, batch_size: 32
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding to prevent future information leakage."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return self.activation(out)


class TCN(nn.Module):
    def __init__(self, input_size=1, num_filters=32, kernel_size=3, num_levels=2):
        super().__init__()
        layers = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_filters
            dilation = 2 ** i
            layers.append(CausalConv1d(in_ch, num_filters, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x shape: (batch, window, features)
        x = x.transpose(1, 2)  # -> (batch, features, window)
        x = self.network(x)    # -> (batch, filters, window)
        x = x[:, :, -1]        # take last time step
        return self.fc(x)


def train(model, X, y, epochs=100, batch_size=32, lr=0.001, X_val=None, y_val=None, model_name="TCN"):
    """Train TCN with mini-batch gradient descent and optional early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(X, y.unsqueeze(1) if y.dim() == 1 else y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        print(f"{model_name} Epoch {epoch+1}/{epochs}")
        model.train()
        for x_batch, y_batch in loader:
            pred = model(x_batch)
            loss = loss_fn(pred.squeeze(), y_batch.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred.squeeze(), y_val.squeeze()).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return model


def predict_model(model, X):
    """Generate predictions."""
    model.eval()
    with torch.no_grad():
        return model(X).squeeze().numpy()


# Hyperparameter search space
SEARCH_SPACE = {
    'num_filters': [4, 8, 16, 32, 64],
    'kernel_size': [1, 2, 3],
}