# Quarterly Aggregation and Model Training (no evaluation)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def load_featured_data(featured_path: str) -> pd.DataFrame:
    """
    Load the feature-engineered dataset.
    Detect the correct target column automatically.
    """
    df = pd.read_csv(featured_path, encoding="utf-8-sig", low_memory=False)
    print(len(df))
    df["date_poursuites"] = pd.to_datetime(df["date_poursuites"], errors="coerce")

    # Identify target column
    possible_targets = [
        "réquisitions_de_vente",
        "réquisitions_de_vente_poursuites",
        "Réquisitions de vente"
    ]
    target_col = next((col for col in possible_targets if col in df.columns), None)
    if target_col is None:
        raise ValueError(f"Nessuna colonna target trovata. Colonne disponibili: {df.columns.tolist()}")

    print(f"Using target column: {target_col}")
    return df, target_col


def aggregate_quarterly(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Aggregate data quarterly by district and compute averages for features.
    """
    # Convert monthly data to quarterly timestamps
    df["quarter"] = df["date_poursuites"].dt.to_period("Q").dt.to_timestamp()

    # Group by district and quarter
    df_quarterly = (
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

    print(f"Aggregated quarterly dataset: {df_quarterly.shape[0]} rows, {df_quarterly.shape[1]} columns")
    return df_quarterly


def prepare_features(df_quarterly: pd.DataFrame, target_col: str):
    """
    Prepare feature matrix X and target vector y for model training.
    """
    y = df_quarterly[target_col]

    # Select input features
    feature_cols = [
        f"{target_col}_lag_3",
        f"{target_col}_lag_4",
        f"{target_col}_lag_5",
        "taux_de_logements_vacants_en_%",
        "month_sin",
        "month_cos",
        "year_trend"
    ]

    X = df_quarterly[feature_cols].fillna(0)
    return X, y


def split_train_test(X, y, df_quarterly, test_size=0.15):
    """
    Chronological split into training and testing sets.
    """
    df_sorted = df_quarterly.sort_values("quarter")
    split_index = int(len(df_sorted) * (1 - test_size))

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_and_save_models(X_train, y_train, model_dir="models"):
    """
    Train Random Forest and XGBoost models and save them to disk.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    rf_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(rf, rf_path)
    print(f"Random Forest model saved to {rf_path}")

    # Train XGBoost
    xgb = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    xgb_path = os.path.join(model_dir, "xgboost.pkl")
    joblib.dump(xgb, xgb_path)
    print(f"XGBoost model saved to {xgb_path}")

    return rf, xgb


def run_training_pipeline(featured_path: str):
    """
    Full quarterly training pipeline:
    1. Load featured dataset
    2. Aggregate by quarter
    3. Prepare features
    4. Split into train/test
    5. Train and save models
    (No evaluation here)
    """
    # Load data and detect target column
    df, target_col = load_featured_data(featured_path)

    # Aggregate monthly data to quarterly
    df_quarterly = aggregate_quarterly(df, target_col)

    # Prepare features and target
    X, y = prepare_features(df_quarterly, target_col)

    # Chronological train/test split
    X_train, X_test, y_train, y_test = split_train_test(X, y, df_quarterly)

    # Train and persist models
    train_and_save_models(X_train, y_train)


if __name__ == "__main__":
    # Define input path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")

    # Execute training pipeline
    run_training_pipeline(featured_path)
