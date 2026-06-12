"""
GRU model for time series forecasting.

Paper-aligned:
  hidden_units: [4, 8, 16, 32, 64]
  layers: [1, 2, 3]
  activation: tanh (default in PyTorch GRU)
  optimizer: Adam, lr=0.001
  epochs: 100, batch_size: 32
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


def train(model, X, y, epochs=100, batch_size=32, lr=0.001, X_val=None, y_val=None, model_name="GRU"):
    """
    Train GRU with mini-batch gradient descent and optional early stopping.
    """
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
    'hidden_size': [4, 8, 16, 32, 64],
    'num_layers': [1, 2, 3],
}