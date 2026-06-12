"""
run_all_models.py — Main experiment runner for RedRVFL paper replication.

Pipeline:
  1. For each dataset × window_size:
     - Load data, create sliding windows
     - Split: train(70%) → val(10%) → test(20%) — sequential, NO shuffle
     - Fit MinMaxScaler on TRAIN ONLY
     - Tune each model's hyperparameters on validation set (Optuna)
     - Retrain best config on train+val combined
     - Evaluate on test set
  2. Report best window size per dataset
  3. Output paper-style tables (RMSE, MAE, MAPE)

Usage:
    python run_all_models.py
"""

import sys
import os
import warnings
import time
import glob
import importlib
import concurrent.futures

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import random
import torch

# Optuna intentionally imported explicitly to ensure tuning is enabled.
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60)
print("Starting project...")
print("=" * 60)

# ========================
# SETTINGS & FAST MODE
# ========================
FAST_MODE = True

if FAST_MODE:
    print("[INFO] FAST_MODE is enabled! Using optimized hyperparams and timeouts.")
    TRIALS_SIMPLE = 3
    TRIALS_DL = 3
    EPOCHS_DL = 5
    TIMEOUT = 300  # Increased to at least 300s as requested
else:
    TRIALS_SIMPLE = 50
    TRIALS_DL = 20
    EPOCHS_DL = 100
    TIMEOUT = 600

# ========================
# REPRODUCIBILITY & DEVICE
# ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")
print(f"[INFO] Device strictly set to: {device} (CUDA disabled)")

# ========================
# IMPORTS
# ========================
from src.data_utils import load_dataset, create_windows, split_data, fit_scaler, transform_data, inverse_transform
from src.metrics import rmse, mae, mape

# ========================
# CONFIG
# ========================
DATASET_FOLDER = os.path.join(BASE_DIR, "RVFL_Datasets")

DATASETS = [
    "DJI.xlsx", "HSI.xlsx", "KOSPI.xlsx", "LSE.xlsx",
    "NASDAQ.xlsx", "NIFTY50.xlsx", "NYSE.xlsx",
    "RUSSELL2000.xlsx", "SENSEX.xlsx", "SP500.xlsx", "SSE.xlsx"
]

WINDOW_SIZES = [48, 96, 124]


def flatten_3d(X):
    """Flatten 3D (n, window, 1) to 2D (n, window) for ML models."""
    return X.reshape(X.shape[0], -1)


def run_persistence(X_test, y_test_orig, scaler):
    from models.persistence_model import predict
    pred = predict(X_test)
    return inverse_transform(pred, scaler)


def run_arima(prices_scaled, train_end, val_end, n_test, scaler, window_size):
    from models.arima_model import train, predict
    train_series = prices_scaled[:val_end + window_size].flatten()
    model = train(train_series)
    pred = predict(model, n_test)
    pred = pred[:n_test]
    return inverse_transform(pred, scaler)


def run_svr(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler):
    from src.tuning import tune_svr
    from models.svr_model import train, predict
    best_params, _ = tune_svr(X_train_flat, y_train, X_val_flat, y_val, n_trials=TRIALS_SIMPLE, seed=SEED, timeout=TIMEOUT)
    X_full = np.concatenate([X_train_flat, X_val_flat])
    y_full = np.concatenate([y_train, y_val])
    model = train(X_full, y_full, **best_params)
    pred = predict(model, X_test_flat)
    return inverse_transform(pred, scaler)


def run_lstm(X_train, y_train, X_val, y_val, X_test, scaler):
    from src.tuning import tune_lstm
    from models.lstm_model import LSTMModel, train, predict_model
    best_params, _ = tune_lstm(X_train, y_train, X_val, y_val, n_trials=TRIALS_DL, seed=SEED, timeout=TIMEOUT, epochs=EPOCHS_DL)
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    torch.manual_seed(SEED)
    model = LSTMModel(input_size=1, **best_params).to(device)
    X_t = torch.tensor(X_full, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_full, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = train(model, X_t, y_t, epochs=EPOCHS_DL, batch_size=32, model_name="LSTM")
    pred = predict_model(model, X_test_t)
    return inverse_transform(pred, scaler)


def run_gru(X_train, y_train, X_val, y_val, X_test, scaler):
    from src.tuning import tune_gru
    from models.gru_model import GRUModel, train, predict_model
    best_params, _ = tune_gru(X_train, y_train, X_val, y_val, n_trials=TRIALS_DL, seed=SEED, timeout=TIMEOUT, epochs=EPOCHS_DL)
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    torch.manual_seed(SEED)
    model = GRUModel(input_size=1, **best_params).to(device)
    X_t = torch.tensor(X_full, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_full, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = train(model, X_t, y_t, epochs=EPOCHS_DL, batch_size=32, model_name="GRU")
    pred = predict_model(model, X_test_t)
    return inverse_transform(pred, scaler)


def run_tcn(X_train, y_train, X_val, y_val, X_test, scaler):
    from src.tuning import tune_tcn
    from models.tcn_model import TCN, train, predict_model
    best_params, _ = tune_tcn(X_train, y_train, X_val, y_val, n_trials=TRIALS_DL, seed=SEED, timeout=TIMEOUT, epochs=EPOCHS_DL)
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    torch.manual_seed(SEED)
    model = TCN(input_size=1, **best_params).to(device)
    X_t = torch.tensor(X_full, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_full, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = train(model, X_t, y_t, epochs=EPOCHS_DL, batch_size=32, model_name="TCN")
    pred = predict_model(model, X_test_t)
    return inverse_transform(pred, scaler)


def run_rvfl(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler):
    from src.tuning import tune_rvfl
    from models.rvfl_model import RVFL
    best_params, _ = tune_rvfl(X_train_flat, y_train, X_val_flat, y_val, n_trials=TRIALS_SIMPLE, seed=SEED, timeout=TIMEOUT)
    X_full = np.concatenate([X_train_flat, X_val_flat])
    y_full = np.concatenate([y_train, y_val])
    model = RVFL(input_dim=X_full.shape[1], seed=SEED, **best_params)
    model.fit(X_full, y_full)
    pred = model.predict(X_test_flat)
    return inverse_transform(pred, scaler)


def run_edrvfl(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler):
    from src.tuning import tune_edrvfl
    from models.edrvfl_model import edRVFL
    best_params, _ = tune_edrvfl(X_train_flat, y_train, X_val_flat, y_val, n_trials=TRIALS_SIMPLE, seed=SEED, timeout=TIMEOUT)
    X_full = np.concatenate([X_train_flat, X_val_flat])
    y_full = np.concatenate([y_train, y_val])
    model = edRVFL(input_dim=X_full.shape[1], seed=SEED, **best_params)
    model.fit(X_full, y_full)
    pred = model.predict(X_test_flat)
    return inverse_transform(pred, scaler)


def run_edesn(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler):
    from src.tuning import tune_edesn
    from models.edesn_model import edESN
    best_params, _ = tune_edesn(X_train_flat, y_train, X_val_flat, y_val, n_trials=TRIALS_SIMPLE, seed=SEED, timeout=TIMEOUT)
    X_full = np.concatenate([X_train_flat, X_val_flat])
    y_full = np.concatenate([y_train, y_val])
    model = edESN(input_dim=X_full.shape[1], seed=SEED, **best_params)
    model.fit(X_full, y_full)
    pred = model.predict(X_test_flat)
    return inverse_transform(pred, scaler)


def run_vmd_lstm(prices_scaled, train_end, val_end, window_size, scaler):
    from models.vmd_lstm_model import VMDLSTM
    train_val_series = prices_scaled[:val_end + window_size].flatten()
    test_series = prices_scaled[val_end:].flatten()
    model = VMDLSTM(window_size=window_size, hidden_size=16, num_layers=1, seed=SEED)
    model.fit(train_val_series, epochs=EPOCHS_DL, batch_size=32)
    pred = model.predict(test_series)
    return inverse_transform(pred, scaler) if pred is not None and len(pred) > 0 else None


def run_ewtrvfl(prices_scaled, train_end, val_end, window_size, scaler):
    from models.ewtrvfl_model import EWTRVFL
    train_val_series = prices_scaled[:val_end + window_size].flatten()
    test_series = prices_scaled[val_end:].flatten()
    model = EWTRVFL(window_size=window_size, hidden_dim=100, ridge_alpha=0.1, seed=SEED)
    model.fit(train_val_series)
    pred = model.predict(test_series)
    return inverse_transform(pred, scaler) if pred is not None and len(pred) > 0 else None


def run_ewtedrvfl(prices_scaled, train_end, val_end, window_size, scaler):
    from models.ewtedrvfl_model import EWTedRVFL
    train_val_series = prices_scaled[:val_end + window_size].flatten()
    test_series = prices_scaled[val_end:].flatten()
    model = EWTedRVFL(window_size=window_size, hidden_dim=100, num_layers=2, ridge_alpha=0.1, seed=SEED)
    model.fit(train_val_series)
    pred = model.predict(test_series)
    return inverse_transform(pred, scaler) if pred is not None and len(pred) > 0 else None


def run_redrvfl(X_train, y_train, X_val, y_val, X_test, scaler):
    from src.tuning import tune_redrvfl
    from models.redrvfl_model import create_model
    best_params, _ = tune_redrvfl(X_train, y_train, X_val, y_val, n_trials=TRIALS_SIMPLE, seed=SEED, timeout=TIMEOUT)
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    num_layers = best_params['num_layers']
    if FAST_MODE:
        num_layers = min(num_layers, 3) # Reduce layers in FAST_MODE

    model = create_model(
        input_features=1,
        hidden_size=best_params['hidden_size'],
        num_layers=num_layers,
        input_scaling=best_params['input_scaling'],
        seed=SEED
    )
    model.fit(X_full_t, y_full, ridge_alpha=best_params['ridge_alpha'])
    pred = model.predict(X_test_t)
    return inverse_transform(pred, scaler)


def evaluate_model(name, pred_orig, y_test_orig):
    if pred_orig is None:
        return None

    min_len = min(len(pred_orig), len(y_test_orig))
    pred_orig = pred_orig[:min_len]
    y_true = y_test_orig[:min_len]

    r = rmse(y_true, pred_orig)
    m = mae(y_true, pred_orig)
    mp = mape(y_true, pred_orig)

    return {'RMSE': r, 'MAE': m, 'MAPE': mp}


# ========================
# MAIN
# ========================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RedRVFL Paper Replication — Full Experiment")
    print("=" * 60)

    # Dynamic loading of all models from models/ directory
    models_dir = os.path.join(BASE_DIR, "models")
    model_files = glob.glob(os.path.join(models_dir, "*_model.py"))
    
    available_model_bases = [os.path.basename(f).replace('_model.py', '') for f in model_files if os.path.basename(f) != '__init__.py']
    
    print(f"[INFO] Dynamically detected the following models: {available_model_bases}")

    # Collect results
    all_results = []

    for file in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {file}")
        print(f"{'='*60}")

        path = os.path.join(DATASET_FOLDER, file)
        try:
            prices = load_dataset(path)
        except Exception as e:
            print(f"Failed to load dataset {file}: {e}")
            continue

        best_per_model = {base: {'RMSE': None, 'MAE': None, 'MAPE': None, 'window': WINDOW_SIZES[0]} for base in available_model_bases}

        for window_size in WINDOW_SIZES:
            print(f"\n  Window Size: {window_size}")
            print(f"  {'-'*50}")

            if len(prices) < window_size + 50:
                print(f"    Skipping: not enough data for window={window_size}")
                continue

            # Paper requirement: min max scaling, standard window setups
            X_raw, y_raw = create_windows(prices, window_size)

            # Sequential split 70/10/20
            splits = split_data(X_raw, y_raw)
            X_train_raw = splits['X_train']
            y_train_raw = splits['y_train']
            X_val_raw = splits['X_val']
            y_val_raw = splits['y_val']
            X_test_raw = splits['X_test']
            y_test_raw = splits['y_test']

            scaler = fit_scaler(X_train_raw, y_train_raw)

            X_train, y_train = transform_data(X_train_raw, y_train_raw, scaler)
            X_val, y_val = transform_data(X_val_raw, y_val_raw, scaler)
            X_test, y_test = transform_data(X_test_raw, y_test_raw, scaler)

            y_test_orig = y_test_raw

            X_train_flat = flatten_3d(X_train)
            X_val_flat = flatten_3d(X_val)
            X_test_flat = flatten_3d(X_test)

            n = len(X_raw)
            train_end = int(n * 0.7)
            val_end = int(n * 0.8)

            prices_scaled = scaler.transform(prices.reshape(-1, 1)).ravel()

            # Map the dynamically loaded bases to their runners
            for base in available_model_bases:
                def model_runner_proxy():
                    if base == 'persistence':
                        return run_persistence(X_test, y_test_orig, scaler)
                    elif base == 'svr':
                        return run_svr(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler)
                    elif base == 'arima':
                        return run_arima(prices_scaled, train_end, val_end, len(y_test), scaler, window_size)
                    elif base == 'lstm':
                        return run_lstm(X_train, y_train, X_val, y_val, X_test, scaler)
                    elif base == 'gru':
                        return run_gru(X_train, y_train, X_val, y_val, X_test, scaler)
                    elif base == 'tcn':
                        return run_tcn(X_train, y_train, X_val, y_val, X_test, scaler)
                    elif base == 'rvfl':
                        return run_rvfl(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler)
                    elif base == 'edrvfl':
                        return run_edrvfl(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler)
                    elif base == 'edesn':
                        return run_edesn(X_train_flat, y_train, X_val_flat, y_val, X_test_flat, scaler)
                    elif base == 'vmd_lstm':
                        return run_vmd_lstm(prices_scaled, train_end, val_end, window_size, scaler)
                    elif base == 'ewtrvfl':
                        return run_ewtrvfl(prices_scaled, train_end, val_end, window_size, scaler)
                    elif base == 'ewtedrvfl':
                        return run_ewtedrvfl(prices_scaled, train_end, val_end, window_size, scaler)
                    elif base == 'redrvfl':
                        return run_redrvfl(X_train, y_train, X_val, y_val, X_test, scaler)
                    else:
                        raise ValueError("Unknown dynamic model signature")

                print(f"Running {base} on {file}, window={window_size}...")
                start_time = time.time()
                pred_orig = None
                
                # Execute with strict timeout threads avoiding silent freezes
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(model_runner_proxy)
                        pred_orig = future.result(timeout=TIMEOUT + 10)  # Added safety margin
                    
                    if pred_orig is not None:
                        metrics = evaluate_model(base, pred_orig, y_test_orig)
                        elapsed = time.time() - start_time
                        
                        if metrics is not None:
                            print(f"{base} DONE -> RMSE={metrics['RMSE']:.3f} (on {file})")

                            if best_per_model[base]['RMSE'] is None or metrics['RMSE'] < best_per_model[base]['RMSE']:
                                best_per_model[base] = {
                                    **metrics,
                                    'window': window_size
                                }
                        else:
                            print(f"{base} FAILED on {file}: Metrics evaluation returned None")
                    else:
                        print(f"{base} FAILED on {file}: Model returned None")
                            
                except concurrent.futures.TimeoutError:
                    print(f"{base} FAILED on {file}: Timeout exceeded (waited {TIMEOUT+10}s)")
                except Exception as e:
                    print(f"{base} ERROR on {file}: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"{base} finished")

        for model_name, metrics in best_per_model.items():
            all_results.append({
                'Dataset': file,
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'Best_Window': metrics['window']
            })

    results_df = pd.DataFrame(all_results, columns=['Dataset', 'Model', 'RMSE', 'MAE', 'MAPE', 'Best_Window'])
    
    try:
        results_df.to_excel(os.path.join(BASE_DIR, "all_model_results.xlsx"), index=False)
    except Exception as e:
        print(f"Excel saving failed ({e}), saving to CSV instead.")
        results_df.to_csv(os.path.join(BASE_DIR, "all_model_results.csv"), index=False)

    if results_df.empty or 'Model' not in results_df.columns:
        print("No valid results found. Skipping ranking.")
        sys.exit(0)

    model_order = [
        "arima", "persistence", "svr", "tcn", "lstm", "gru",
        "rvfl", "ewtrvfl", "vmd_lstm", "edesn", "edrvfl", "ewtedrvfl", "redrvfl"
    ]

    if 'Model' in results_df.columns:
        available_models = [m for m in model_order if m in results_df['Model'].values]
    else:
        available_models = []

    if not available_models:
        print("No models available for ranking.")
        sys.exit(0)

    print("\n" + "=" * 80)
    print("===== TABLE: RMSE RESULTS =====")
    print("=" * 80)
    rmse_table = results_df.pivot(index="Dataset", columns="Model", values="RMSE")
    rmse_table = rmse_table[[m for m in available_models if m in rmse_table.columns]]
    print(rmse_table.round(3).to_string())
    rmse_table.to_csv(os.path.join(BASE_DIR, "rmse_results.csv"))

    print("\n" + "=" * 80)
    print("===== TABLE: MAE RESULTS =====")
    print("=" * 80)
    mae_table = results_df.pivot(index="Dataset", columns="Model", values="MAE")
    mae_table = mae_table[[m for m in available_models if m in mae_table.columns]]
    print(mae_table.round(3).to_string())
    mae_table.to_csv(os.path.join(BASE_DIR, "mae_results.csv"))

    print("\n" + "=" * 80)
    print("===== TABLE: MAPE RESULTS =====")
    print("=" * 80)
    mape_table = results_df.pivot(index="Dataset", columns="Model", values="MAPE")
    mape_table = mape_table[[m for m in available_models if m in mape_table.columns]]
    print(mape_table.round(4).to_string())
    mape_table.to_csv(os.path.join(BASE_DIR, "mape_results.csv"))

    print("\n" + "=" * 80)
    print("===== MODEL RANKING (by average RMSE) =====")
    print("=" * 80)
    avg_rmse = results_df.groupby('Model')['RMSE'].mean().sort_values()
    for rank, (model, val) in enumerate(avg_rmse.items(), 1):
        marker = " ★" if model == "redrvfl" else ""
        print(f"  {rank:2d}. {model:12s}: {val:.3f}{marker}")

    print(f"\nResults saved to: all_model_results.xlsx (or .csv if fallback)")
    print("Done.")