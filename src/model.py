# Model Training script (with Temporal Validation Train/Val/Test Split)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
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
    
    print(f"\nTemporal Split:")
    print(f"   Train: {len(X_train)} samples (1994-2015)")
    print(f"   Validation: {len(X_val)} samples (2016-2020)")
    print(f"   Test: {len(X_test)} samples (2021-2024)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def train_random_forest(X_train, y_train, model_dir="models"):
    """
    Train Random Forest model for multivariate time series regression.
    """
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    rf_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(rf, rf_path)
    print(f"   Model saved to {rf_path}")
    
    return rf

def train_xgboost(X_train, y_train, model_dir="models"):
    """
    Train XGBoost model for gradient boosting regression.
    """
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    xgb_path = os.path.join(model_dir, "xgboost.pkl")
    joblib.dump(xgb, xgb_path)
    print(f"   Model saved to {xgb_path}")
    
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
            print(f"   District {district_id}: ARIMA trained successfully")
            
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
    5. Train all three models (RF, XGBoost, ARIMA)
    6. Save all models
    """
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    df, date_col = load_featured_data(featured_path)
    target_col = detect_target_column(df)
    
    # Create temporal splits if they don't exist
    df = create_temporal_splits(df, date_col)
    
    # Prepare temporal splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_temporal_splits(df, target_col)
    
    # Train Machine Learning models
    rf = train_random_forest(X_train, y_train, model_dir)
    xgb = train_xgboost(X_train, y_train, model_dir)
    
    # Train ARIMA models (one per district)
    arima_models = train_arima_per_district(df, target_col, date_col, model_dir)
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModels saved to: {model_dir}/")
    print("   - random_forest.pkl")
    print("   - xgboost.pkl")
    print(f"   - arima_models.pkl ({len(arima_models)} districts)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")
    
    run_training_pipeline(featured_path, model_dir)