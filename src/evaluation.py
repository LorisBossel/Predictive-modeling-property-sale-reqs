# Multi-District Evaluation for Random Forest & XGBoost Models

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_featured_data(featured_path: str) -> pd.DataFrame:
    """
    Load the feature-engineered dataset already containing lag features and temporal columns.
    """
    df = pd.read_csv(featured_path, encoding="utf-8-sig", low_memory=False)
    df["date_poursuites"] = pd.to_datetime(df["date_poursuites"], errors="coerce")
    print(f"Loaded dataset: {df.shape[0]} rows, {df['district_id'].nunique()} districts")
    return df


def load_trained_models(model_dir="models"):
    """
    Load pre-trained Random Forest and XGBoost models from the models directory.
    """
    rf_path = os.path.join(model_dir, "random_forest.pkl")
    xgb_path = os.path.join(model_dir, "xgboost.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")

    rf = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)
    print("Loaded trained models successfully.")
    return rf, xgb


def aggregate_quarterly(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Aggregate monthly data by quarter and district, maintaining lagged and time features.
    """
    df["quarter"] = df["date_poursuites"].dt.to_period("Q").dt.to_timestamp()

    df_q = (
        df.groupby(["district_id", "quarter"])
          .agg({
              target_col: "sum",
              "taux_de_logements_vacants_en_%": "mean",
              "month_sin": "mean",
              "month_cos": "mean",
              "year_trend": "mean",
              f"{target_col}_lag_3": "mean",
              f"{target_col}_lag_4": "mean",
              f"{target_col}_lag_5": "mean"
          })
          .reset_index()
    )
    return df_q


def prepare_features(df_q: pd.DataFrame, target_col: str):
    """
    Prepare feature matrix X and target vector y.
    """
    y = df_q[target_col]
    X = df_q[[
        f"{target_col}_lag_3",
        f"{target_col}_lag_4",
        f"{target_col}_lag_5",
        "taux_de_logements_vacants_en_%",
        "month_sin",
        "month_cos",
        "year_trend"
    ]].fillna(0)
    return X, y


def evaluate_models_per_district(df_q: pd.DataFrame, rf, xgb, target_col: str, output_path="results/multi_district_metrics.csv"):
    """
    Evaluate both Random Forest and XGBoost models for each district separately.
    Save metrics to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    for district_id in sorted(df_q["district_id"].unique()):
        df_d = df_q[df_q["district_id"] == district_id].copy()
        if len(df_d) < 10:
            continue  # Skip small series

        X, y = prepare_features(df_d, target_col)
        split_index = int(len(df_d) * 0.85)
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        # Predictions
        preds_rf = rf.predict(X_test)
        preds_xgb = xgb.predict(X_test)

        # Compute metrics
        mae_rf = mean_absolute_error(y_test, preds_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))
        mape_rf = np.mean(np.abs((y_test - preds_rf) / y_test)) * 100
        r2_rf = r2_score(y_test, preds_rf)

        mae_xgb = mean_absolute_error(y_test, preds_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
        mape_xgb = np.mean(np.abs((y_test - preds_xgb) / y_test)) * 100
        r2_xgb = r2_score(y_test, preds_xgb)

        results.append({
            "district_id": district_id,
            "MAE_RF": mae_rf, "RMSE_RF": rmse_rf, "MAPE_RF": mape_rf, "R2_RF": r2_rf,
            "MAE_XGB": mae_xgb, "RMSE_XGB": rmse_xgb, "MAPE_XGB": mape_xgb, "R2_XGB": r2_xgb
        })

        print(f"District {district_id} — RF MAE={mae_rf:.2f}, XGB MAE={mae_xgb:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")
    return results_df


def plot_metric_comparison(results_df: pd.DataFrame, output_dir="results"):
    """
    Generate bar charts comparing MAE and RMSE across districts.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_df = results_df.copy()
    plot_df.plot(
        x="district_id", y=["MAE_RF", "MAE_XGB"],
        kind="bar", figsize=(10, 5), color=["skyblue", "salmon"]
    )
    plt.title("MAE Comparison — Random Forest vs XGBoost per District")
    plt.xlabel("District ID")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_comparison_districts.png"))
    plt.close()

    plot_df.plot(
        x="district_id", y=["RMSE_RF", "RMSE_XGB"],
        kind="bar", figsize=(10, 5), color=["skyblue", "salmon"]
    )
    plt.title("RMSE Comparison — Random Forest vs XGBoost per District")
    plt.xlabel("District ID")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_comparison_districts.png"))
    plt.close()

    print("Metric comparison plots saved.")


def run_multi_district_evaluation(featured_path: str, model_dir="models"):
    """
    End-to-end evaluation pipeline:
    1. Load dataset and trained models
    2. Aggregate quarterly data
    3. Evaluate both models across all districts
    4. Generate comparison plots
    """
    df = load_featured_data(featured_path)
    rf, xgb = load_trained_models(model_dir)

    # Detect target column
    possible_targets = [
        "réquisitions_de_vente",
        "réquisitions_de_vente_poursuites",
        "Réquisitions de vente"
    ]
    target_col = next((col for col in possible_targets if col in df.columns), None)
    if target_col is None:
        raise ValueError("Target column not found in dataset.")

    df_q = aggregate_quarterly(df, target_col)
    results_df = evaluate_models_per_district(df_q, rf, xgb, target_col)
    plot_metric_comparison(results_df)

    print("\nEvaluation completed. Results available in /results folder.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")

    run_multi_district_evaluation(featured_path, model_dir)
