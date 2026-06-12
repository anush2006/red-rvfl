import pandas as pd

# -------------------------------
# LOAD DATA
# -------------------------------
FILE_PATH = "results_mean.csv"

df = pd.read_csv(FILE_PATH)

df.columns = [
    "Config ID", "Dataset",
    "Window","K" ,"Hidden Size", "Num Layers",
    "Ridge Alpha", "Input Scaling",
    "RMSE", "MAE", "MAPE", "Training Time"
]

print("\nLoaded data:", df.shape)


# -------------------------------
# SAFETY CHECK (VERY IMPORTANT)
# -------------------------------
# Ensure each config ID has consistent hyperparameters
param_cols = ["Window", "K", "Hidden Size", "Num Layers", "Ridge Alpha", "Input Scaling"]

for col in param_cols:
    check = df.groupby("Config ID")[col].nunique()
    if (check > 1).any():
        raise ValueError(f"Inconsistent config detected in column: {col}")

print("✔ Config consistency verified")


# -------------------------------
# CONFIG-LEVEL AGGREGATION
# -------------------------------
config_perf = (
    df.groupby("Config ID")
    .agg({
        "Window": "first",
        "K": "first",
        "Hidden Size": "first",
        "Num Layers": "first",
        "Ridge Alpha": "first",
        "Input Scaling": "first",
        "RMSE": "mean",
        "MAE": "mean",
        "MAPE": "mean",
        "Training Time": "mean"
    })
    .reset_index()
)

print("\nTotal configs:", len(config_perf))


# -------------------------------
# TOP CONFIGS (ROBUST)
# -------------------------------
TOP_K = 10

top_configs = config_perf.sort_values("RMSE").head(TOP_K).copy()

print("\n==============================")
print("TOP CONFIGS (by AVG RMSE)")
print("==============================")
print(top_configs)


# -------------------------------
# DATASET-LEVEL RESULTS FOR TOP CONFIGS
# -------------------------------
top_config_ids = top_configs["Config ID"].tolist()

top_dataset_results = df[df["Config ID"].isin(top_config_ids)].copy()

print("\n==============================")
print("TOP CONFIGS → PER DATASET RESULTS")
print("==============================")
print(top_dataset_results.head())


# -------------------------------
# HYPERPARAMETER DISTRIBUTION
# -------------------------------
print("\n==============================")
print("TOP CONFIG HYPERPARAMETER DISTRIBUTION")
print("==============================")

for col in param_cols:
    print(f"\n{col}:")
    print(top_configs[col].value_counts())


# -------------------------------
# GLOBAL SENSITIVITY ANALYSIS
# -------------------------------
print("\n==============================")
print("GLOBAL HYPERPARAMETER SENSITIVITY (AVG RMSE)")
print("==============================")

def print_group_analysis(column):
    grouped = df.groupby(column)["RMSE"].mean().sort_values()
    print(f"\n{column}:")
    print(grouped)

for col in param_cols:
    print_group_analysis(col)


# -------------------------------
# INTERACTION EFFECTS
# -------------------------------
print("\n==============================")
print("INTERACTION: Num Layers × Input Scaling")
print("==============================")

interaction = (
    df.groupby(["Num Layers", "Input Scaling"])["RMSE"]
    .mean()
    .unstack()
)

print(interaction)


# -------------------------------
# GOOD vs BAD REGION ANALYSIS
# -------------------------------
print("\n==============================")
print("GOOD vs BAD CONFIG REGIONS")
print("==============================")

good_threshold = config_perf["RMSE"].quantile(0.1)
bad_threshold  = config_perf["RMSE"].quantile(0.9)

good_configs = config_perf[config_perf["RMSE"] <= good_threshold]
bad_configs  = config_perf[config_perf["RMSE"] >= bad_threshold]

print("\nGOOD CONFIG REGION (Top 10%)")
print(good_configs.describe())

print("\nBAD CONFIG REGION (Bottom 10%)")
print(bad_configs.describe())


# -------------------------------
# BEST CONFIG
# -------------------------------
best_config = top_configs.iloc[0]

print("\n==============================")
print("BEST CONFIG")
print("==============================")
print(best_config)


# -------------------------------
# SAVE OUTPUTS (IMPORTANT)
# -------------------------------
top_configs.to_csv("top_10_configs.csv", index=False)
config_perf.to_csv("config_performance.csv", index=False)
top_dataset_results.to_csv("top_configs_dataset_results.csv", index=False)

print("\nSaved:")
print(" - top_10_configs.csv (config-level summary)")
print(" - config_performance.csv (all configs)")
print(" - top_configs_dataset_results.csv (dataset-level results)")