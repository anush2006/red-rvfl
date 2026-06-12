"""
Hyperparameter tuning module using Optuna.

Each tune_* function uses the VALIDATION set as the objective
and returns the best hyperparameters + best validation score.

Paper-aligned search spaces:
  SVR: C [2^-10, 2^0], epsilon [0.001, 0.01, 0.1]
  LSTM/GRU: hidden [4,8,16,32], layers [1,2,3]
  TCN: filters [4,8,16,32], kernel [1,2,3]
  RVFL: hidden [20-100], lambda [0-1], scaling [0-1]
  edRVFL: + layers [1-6]
  edESN: reservoir [20-100], spectral_radius [0.1-1.0]
  RedRVFL: hidden [20-100], layers [1-6], lambda [0-1], scaling [0-1]

Note: Search spaces have been globally reduced to stabilize performance.
"""

import numpy as np
import optuna
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.metrics import rmse

# enforce CPU
device = torch.device("cpu")


# ============================================================
# SVR
# ============================================================
def tune_svr(X_train, y_train, X_val, y_val, n_trials=50, seed=42, timeout=600):
    """Tune SVR hyperparameters using Optuna."""
    from models.svr_model import train, predict

    def objective(trial):
        C = trial.suggest_float('C', 2**-10, 2**0, log=True)
        epsilon = trial.suggest_categorical('epsilon', [0.001, 0.01, 0.1])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        model = train(X_train, y_train, C=C, epsilon=epsilon, gamma=gamma)
        pred = predict(model, X_val)
        return rmse(y_val, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# LSTM
# ============================================================
def tune_lstm(X_train, y_train, X_val, y_val, n_trials=30, seed=42, timeout=600, epochs=100):
    """Tune LSTM hyperparameters using Optuna."""
    from models.lstm_model import LSTMModel, train, predict_model

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32])
        num_layers = trial.suggest_categorical('num_layers', [1, 2])

        torch.manual_seed(seed)
        model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

        model = train(model, X_t, y_t, epochs=epochs, batch_size=32,
                      X_val=X_v, y_val=y_v, model_name=f"LSTM-Optuna")
        pred = predict_model(model, X_v)
        return rmse(y_v.cpu().numpy(), pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# GRU
# ============================================================
def tune_gru(X_train, y_train, X_val, y_val, n_trials=30, seed=42, timeout=600, epochs=100):
    """Tune GRU hyperparameters using Optuna."""
    from models.gru_model import GRUModel, train, predict_model

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32])
        num_layers = trial.suggest_categorical('num_layers', [1, 2])

        torch.manual_seed(seed)
        model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

        model = train(model, X_t, y_t, epochs=epochs, batch_size=32,
                      X_val=X_v, y_val=y_v, model_name=f"GRU-Optuna")
        pred = predict_model(model, X_v)
        return rmse(y_v.cpu().numpy(), pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# TCN
# ============================================================
def tune_tcn(X_train, y_train, X_val, y_val, n_trials=30, seed=42, timeout=600, epochs=100):
    """Tune TCN hyperparameters using Optuna."""
    from models.tcn_model import TCN, train, predict_model

    def objective(trial):
        num_filters = trial.suggest_categorical('num_filters', [4, 8, 16, 32])
        kernel_size = trial.suggest_categorical('kernel_size', [1, 2, 3])

        torch.manual_seed(seed)
        model = TCN(input_size=1, num_filters=num_filters, kernel_size=kernel_size).to(device)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

        model = train(model, X_t, y_t, epochs=epochs, batch_size=32,
                      X_val=X_v, y_val=y_v, model_name=f"TCN-Optuna")
        pred = predict_model(model, X_v)
        return rmse(y_v.cpu().numpy(), pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# RVFL
# ============================================================
def tune_rvfl(X_train_flat, y_train, X_val_flat, y_val, n_trials=50, seed=42, timeout=600):
    """Tune RVFL hyperparameters using Optuna."""
    from models.rvfl_model import RVFL

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 20, 100)
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.0, 1.0)
        input_scaling = trial.suggest_float('input_scaling', 0.0, 1.0)

        model = RVFL(
            input_dim=X_train_flat.shape[1],
            hidden_dim=hidden_dim,
            ridge_alpha=ridge_alpha,
            input_scaling=input_scaling,
            seed=seed
        )
        model.fit(X_train_flat, y_train)
        pred = model.predict(X_val_flat)
        return rmse(y_val, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# edRVFL
# ============================================================
def tune_edrvfl(X_train_flat, y_train, X_val_flat, y_val, n_trials=50, seed=42, timeout=600):
    """Tune edRVFL hyperparameters using Optuna."""
    from models.edrvfl_model import edRVFL

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 20, 100)
        num_layers = trial.suggest_int('num_layers', 1, 6)
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.0, 1.0)
        input_scaling = trial.suggest_float('input_scaling', 0.0, 1.0)

        model = edRVFL(
            input_dim=X_train_flat.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ridge_alpha=ridge_alpha,
            input_scaling=input_scaling,
            seed=seed
        )
        model.fit(X_train_flat, y_train)
        pred = model.predict(X_val_flat)
        return rmse(y_val, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# edESN
# ============================================================
def tune_edesn(X_train_flat, y_train, X_val_flat, y_val, n_trials=50, seed=42, timeout=600):
    """Tune edESN hyperparameters using Optuna."""
    from models.edesn_model import edESN

    def objective(trial):
        reservoir_size = trial.suggest_int('reservoir_size', 20, 100)
        spectral_radius = trial.suggest_float('spectral_radius', 0.1, 1.0)
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.0, 1.0)
        input_scaling = trial.suggest_float('input_scaling', 0.0, 1.0)

        model = edESN(
            input_dim=X_train_flat.shape[1],
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            ridge_alpha=ridge_alpha,
            input_scaling=input_scaling,
            seed=seed
        )
        model.fit(X_train_flat, y_train)
        pred = model.predict(X_val_flat)
        return rmse(y_val, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


# ============================================================
# RedRVFL
# ============================================================
def tune_redrvfl(X_train, y_train, X_val, y_val, n_trials=50, seed=42, timeout=600):
    """
    Tune RedRVFL hyperparameters using Optuna.
    """
    from src.red_revfl_orchestrator import RedRVFLOrchestrator

    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 20, 100)
        num_layers = trial.suggest_int('num_layers', 1, 6)
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.0, 1.0)
        input_scaling = trial.suggest_float('input_scaling', 0.0, 1.0)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)

        model = RedRVFLOrchestrator(
            input_features=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_scaling=input_scaling,
            seed=seed
        )

        model.fit(X_t, y_train, ridge_alpha=ridge_alpha)
        pred = model.predict(X_v)
        return rmse(y_val, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value
