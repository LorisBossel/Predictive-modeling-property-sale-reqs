"""
Model training module.

Implements training pipeline for Random Forest, XGBoost, and ARIMA models
with temporal validation.
"""


import os
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def load_featured_data(featured_path: str) -> tuple:
    """
    Load the feature-engineered dataset.
    Returns dataframe and detected date column name.
    """
    df = pd.read_csv(featured_path, encoding="utf-8-sig", low_memory=False)

    # Try different possible date column names
    date_cols = ["date", "date_poursuites", "Date"]
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print(f"Loaded dataset: {df.shape[0]} rows, {df['district_id'].nunique()} districts")
    print(f"Using date column: {date_col}")

    return df, date_col

def detect_target_column(df: pd.DataFrame) -> str:
    """
    Automatically detect the target column.
    """
    possible_targets = [
        "requisitions_de_vente",
        "réquisitions_de_vente",
        "RÃ©quisitions de vente"
    ]

    for target in possible_targets:
        if target in df.columns:
            print(f"Using target column: {target}")
            return target

    raise ValueError(f"No target column found. Available: {df.columns.tolist()}")

def create_temporal_splits(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Create temporal split labels if they don't exist.
    Train: 1994-2015
    Validation: 2016-2020
    Test: 2021-2024
    """
    if "split" in df.columns:
        print("Split column already exists")
        return df

    # Extract year from date column
    if "year" not in df.columns:
        df["year"] = df[date_col].dt.year

    # Create split labels
    df["split"] = "train"
    df.loc[df["year"].between(2016, 2020), "split"] = "validation"
    df.loc[df["year"] >= 2021, "split"] = "test"

    print("\nTemporal split distribution:")
    print(df["split"].value_counts().sort_index())

    return df

def prepare_temporal_splits(df: pd.DataFrame, target_col: str):
    """
    Split data according to temporal validation strategy.
    """
    # Identify feature columns (exclude metadata)
    feature_cols = [
        col for col in df.columns
        if "lag" in col or "rolling" in col or col in [
            "month_sin", "month_cos", "year_trend",
            "taux_de_logements_vacants_en_%"
        ]
    ]

    if not feature_cols:
        raise ValueError("No feature columns found in dataset. Did you run feature_engineering.py?")

    print(f"\nFound {len(feature_cols)} feature columns")

    # Split by temporal label
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Prepare X and y for each split
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]

    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col]

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]

    print(f"\nTemporal split:")
    print(f"   Train: {len(X_train)} samples (1994-2015)")
    print(f"   Validation: {len(X_val)} samples (2016-2020)")
    print(f"   Test: {len(X_test)} samples (2021-2024)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def train_random_forest(X_train, y_train, X_val, y_val, model_dir="models"):
    """
    Train Random Forest model for multivariate time series regression.
    Now tracks validation performance during training.
    """
    print("\nTraining Random Forest with validation tracking...")

    # Dictionary to store training history
    history = {
        'n_estimators': [],
        'train_mae': [],
        'val_mae': []
    }

    # Train incrementally to see progression
    max_estimators = 150
    step = 10

    for n_est in range(step, max_estimators + 1, step):
        # Train model with n_est trees
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Calculate MAE on train and validation sets
        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        # Store in history
        history['n_estimators'].append(n_est)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(f"   n_estimators={n_est}: Train MAE={train_mae:.2f}, Val MAE={val_mae:.2f}")

    # Train final model with all estimators
    rf_final = RandomForestRegressor(
        n_estimators=max_estimators,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X_train, y_train)

    # Save model and history
    os.makedirs(model_dir, exist_ok=True)
    rf_path = os.path.join(model_dir, "random_forest.pkl")
    history_path = os.path.join(model_dir, "rf_training_history.pkl")

    joblib.dump(rf_final, rf_path)
    joblib.dump(history, history_path)

    print(f"   Model saved to {rf_path}")
    print(f"   Training history saved to {history_path}")

    return rf_final

def train_xgboost(X_train, y_train, X_val, y_val, model_dir="models"):
    """
    Train XGBoost model for gradient boosting regression.
    XGBoost has native support for tracking training history.
    """
    print("\nTraining XGBoost with validation tracking...")

    # XGBoost can automatically track train and validation metrics
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )

    # Fit with eval_set to track validation performance
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Extract training history
    history = {
        'iteration': list(range(len(xgb.evals_result()['validation_0']['mae']))),
        'train_mae': xgb.evals_result()['validation_0']['mae'],
        'val_mae': xgb.evals_result()['validation_1']['mae']
    }

    # Save model and history
    os.makedirs(model_dir, exist_ok=True)
    xgb_path = os.path.join(model_dir, "xgboost.pkl")
    history_path = os.path.join(model_dir, "xgb_training_history.pkl")

    joblib.dump(xgb, xgb_path)
    joblib.dump(history, history_path)

    print(f"   Model saved to {xgb_path}")
    print(f"   Final Train MAE: {history['train_mae'][-1]:.2f}")
    print(f"   Final Val MAE: {history['val_mae'][-1]:.2f}")
    print(f"   Training history saved to {history_path}")

    return xgb

def train_arima_per_district(df: pd.DataFrame, target_col: str, date_col: str, model_dir="models"):
    """
    Train ARIMA models separately for each district (univariate forecasting).
    ARIMA models are trained on the training period only.
    """
    print("\nTraining ARIMA models per district...")

    os.makedirs(model_dir, exist_ok=True)
    arima_models = {}

    districts = sorted(df["district_id"].unique())

    for district_id in districts:
        df_district = df[df["district_id"] == district_id].sort_values(date_col)

        # Use only training data
        train_data = df_district[df_district["split"] == "train"][target_col]

        if len(train_data) < 20:
            print(f"   Warning: District {district_id}: insufficient training data (n={len(train_data)})")
            continue

        try:
            # Fit ARIMA(1,1,1) as baseline
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            arima_models[district_id] = fitted_model
            print(f"   District {district_id}: ARIMA trained.")

        except Exception as e:
            print(f"   Warning: District {district_id}: ARIMA failed ({str(e)[:50]})")
            continue

    # Save all ARIMA models
    arima_path = os.path.join(model_dir, "arima_models.pkl")
    joblib.dump(arima_models, arima_path)
    print(f"\nTrained ARIMA for {len(arima_models)}/{len(districts)} districts")
    print(f"Models saved to {arima_path}")

    return arima_models

def run_training_pipeline(featured_path: str, model_dir="models"):
    """
    Full training pipeline with temporal validation:
    1. Load featured dataset
    2. Detect target and date columns
    3. Create temporal splits if needed
    4. Prepare train/val/test sets
    5. Train all three models
    6. Save all models
    """
    print("Model training pipeline")

    # Load data
    df, date_col = load_featured_data(featured_path)
    target_col = detect_target_column(df)

    # Create temporal splits if they don't exist
    df = create_temporal_splits(df, date_col)

    # Prepare temporal splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_temporal_splits(df, target_col)

    # Train machine learning models with validation tracking
    rf = train_random_forest(X_train, y_train, X_val, y_val, model_dir)
    xgb = train_xgboost(X_train, y_train, X_val, y_val, model_dir)

    # Train ARIMA models (one per district)
    arima_models = train_arima_per_district(df, target_col, date_col, model_dir)

    print("\nAll models trained.")
    print(f"\nModels saved to: {model_dir}/")
    print("   - random_forest.pkl")
    print("   - rf_training_history.pkl")
    print("   - xgboost.pkl")
    print("   - xgb_training_history.pkl")
    print(f"   - arima_models.pkl ({len(arima_models)} districts)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")

    run_training_pipeline(featured_path, model_dir)
