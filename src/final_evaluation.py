import os
import numpy as np
import pandas as pd

from src.run.data_loader import load_dataset, create_dataset
from src.run.model_runner import train_model, predict
from src.run.evaluator import inverse_scale, evaluate


DATASET_FOLDER = "./RVFL_Datasets"
RESULTS_PATH = "results_mean.csv"   # from 70/10/20 phase


datasets = [
    "DJI.xlsx", "HSI.xlsx", "KOSPI.xlsx", "LSE.xlsx",
    "NASDAQ.xlsx", "NIFTY50.xlsx", "NYSE.xlsx",
    "RUSSELL2000.xlsx", "SENSEX.xlsx", "SP500.xlsx", "SSE.xlsx"
]


def scale(data, x_min, x_max, scaling):
    return ((data - x_min) / (x_max - x_min + 1e-8)) * scaling


# -------------------------------
# LOAD RESULTS (70/10/20)
# -------------------------------
df = pd.read_csv(RESULTS_PATH)

df.columns = [
    "Config ID", "Dataset",
    "Window", "K",
    "Hidden Size", "Num Layers",
    "Ridge Alpha", "Input Scaling",
    "RMSE", "MAE", "MAPE", "Time"
]


# -------------------------------
# DATASET-WISE BEST CONFIG
# -------------------------------
best_per_dataset = (
    df.sort_values("RMSE")
      .groupby("Dataset")
      .first()
      .reset_index()
)

print("\nBest config per dataset:")
print(best_per_dataset[["Dataset", "Config ID", "RMSE"]])


# -------------------------------
# FINAL 80/20 EVALUATION
# -------------------------------
results = []

for _, row in best_per_dataset.iterrows():

    dataset = row["Dataset"]

    cfg = {
        "window": int(row["Window"]),
        "k": int(row["K"]),
        "hidden_size": int(row["Hidden Size"]),
        "num_layers": int(row["Num Layers"]),
        "ridge_alpha": float(row["Ridge Alpha"]),
        "input_scaling": float(row["Input Scaling"])
    }

    print(f"\nRunning {dataset} with Config {int(row['Config ID'])}")

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    path = os.path.join(DATASET_FOLDER, dataset)
    prices = load_dataset(path).reshape(-1, 1)

    n = len(prices)
    split = int(n * 0.8)

    train_prices = prices[:split]
    test_prices  = prices[split:]


    # -------------------------------
    # SCALE (TRAIN ONLY)
    # -------------------------------
    x_min = train_prices.min()
    x_max = train_prices.max()

    train_scaled = scale(train_prices, x_min, x_max, cfg["input_scaling"])
    test_scaled  = scale(test_prices,  x_min, x_max, cfg["input_scaling"])


    # -------------------------------
    # CREATE WINDOWS
    # -------------------------------
    X_train, y_train = create_dataset(train_scaled, cfg["window"], cfg["k"])
    X_test,  y_test  = create_dataset(test_scaled,  cfg["window"], cfg["k"])


    # -------------------------------
    # TRAIN
    # -------------------------------
    model, ridge_models = train_model(X_train, y_train, cfg)


    # -------------------------------
    # TEST
    # -------------------------------
    pred = predict(model, ridge_models, X_test, cfg)


    # -------------------------------
    # INVERSE SCALE
    # -------------------------------
    y_test_orig = inverse_scale(y_test, cfg["input_scaling"], x_min, x_max).flatten()
    pred_orig   = inverse_scale(pred, cfg["input_scaling"], x_min, x_max)


    # -------------------------------
    # METRICS
    # -------------------------------
    rmse, mae, mape = evaluate(y_test_orig, pred_orig)

    print(f"{dataset} | RMSE: {rmse:.3f}")

    results.append([
        dataset,
        int(row["Config ID"]),
        cfg["window"], cfg["k"],
        cfg["hidden_size"], cfg["num_layers"],
        cfg["ridge_alpha"], cfg["input_scaling"],
        rmse, mae, mape
    ])


# -------------------------------
# SAVE FINAL RESULTS
# -------------------------------
columns = [
    "Dataset", "Config ID",
    "Window", "K",
    "Hidden", "Layers",
    "Ridge", "Scaling",
    "RMSE", "MAE", "MAPE"
]

final_df = pd.DataFrame(results, columns=columns)
final_df.to_csv("final_results_datasetwise.csv", index=False)

print("\nSaved → final_results_datasetwise.csv")