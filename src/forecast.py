# Future Forecasting (2025-2026) script using best performing model

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    raise ValueError("No target column found.")

def load_best_model(model_dir="models", model_type="xgboost"):
    """
    Load the best performing model (default: XGBoost based on evaluation results).
    """
    model_paths = {
        "random_forest": os.path.join(model_dir, "random_forest.pkl"),
        "xgboost": os.path.join(model_dir, "xgboost.pkl"),
        "arima": os.path.join(model_dir, "arima_models.pkl")
    }
    
    model_path = model_paths.get(model_type)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded {model_type} model from {model_path}")
    return model, model_type

def generate_future_dates(start_year=2025, end_year=2026, freq="MS"):
    """
    Generate monthly dates for the forecast period.
    MS = Month Start
    """
    date_range = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq=freq
    )
    return date_range

def get_last_known_values(df, district_id, target_col, date_col):
    """
    Get the last known values for a district to use as base for forecasting.
    """
    df_d = df[df["district_id"] == district_id].sort_values(date_col)
    
    if len(df_d) == 0:
        return None
    
    last_row = df_d.iloc[-1].copy()
    return last_row

def prepare_forecast_features(df, target_col, date_col, districts, future_dates):
    """
    Prepare features for future forecasting.
    Uses the last known values and create features for the future dates.
    """
    # First, identify which feature columns were used in training
    available_features = [col for col in df.columns if "lag" in col or "rolling" in col]
    print(f"Available features in dataset: {len(available_features)}")
    
    forecast_records = []
    
    for district_id in districts:
        last_row = get_last_known_values(df, district_id, target_col, date_col)
        
        if last_row is None:
            continue
        
        for future_date in future_dates:
            # Extract temporal features
            month = future_date.month
            year = future_date.year
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Year trend (assuming year column exists or extract from date)
            year_min = df["year"].min() if "year" in df.columns else df[date_col].dt.year.min()
            year_trend = year - year_min
            
            # Prepare feature dictionary
            features = {}
            
            # Add ALL lag features that exist in the dataset
            for col in df.columns:
                if "lag" in col and target_col in col:
                    features[col] = last_row.get(col, 0)
            
            # Add ALL rolling features that exist in the dataset
            for col in df.columns:
                if "rolling" in col and target_col in col:
                    features[col] = last_row.get(col, 0)
            
            # Add vacancy rate if available
            if "taux_de_logements_vacants_en_%" in df.columns:
                features["taux_de_logements_vacants_en_%"] = last_row.get("taux_de_logements_vacants_en_%", 0)
            
            # Add temporal features
            features["month_sin"] = month_sin
            features["month_cos"] = month_cos
            features["year_trend"] = year_trend
            
            # Add metadata (useful for output)
            features["district_id"] = district_id
            features["date"] = future_date
            features["year"] = year
            features["month"] = month
            
            forecast_records.append(features)
    
    forecast_df = pd.DataFrame(forecast_records)
    
    # Ensure column order matches training data
    feature_cols_ordered = [
        col for col in df.columns 
        if col in forecast_df.columns and (
            "lag" in col or "rolling" in col or col in [
                "taux_de_logements_vacants_en_%", "month_sin", "month_cos", "year_trend"
            ]
        )
    ]
    
    # Reorder columns: metadata first, then features in correct order
    metadata_cols = ["district_id", "date", "year", "month"]
    forecast_df = forecast_df[metadata_cols + feature_cols_ordered]
    
    print(f"Prepared {len(forecast_df)} forecast records with {len(feature_cols_ordered)} features")
    
    return forecast_df

def generate_forecasts(model, model_type, forecast_df, target_col):
    """
    Generate predictions using the trained model.
    Ensures feature order matches training.
    """
    # Identify feature columns (exclude metadata)
    feature_cols = [
        col for col in forecast_df.columns
        if col not in ["district_id", "date", "year", "month"]
    ]
    
    if not feature_cols:
        raise ValueError("No feature columns found in forecast dataframe")
    
    print(f"Using {len(feature_cols)} features for prediction")
    print(f"Features: {feature_cols[:5]}...")  # Print first 5 for verification
    
    X_future = forecast_df[feature_cols].fillna(0)
    
    if model_type == "arima":
        # ARIMA requires special handling (per-district forecasting)
        predictions = []
        for district_id in forecast_df["district_id"].unique():
            if district_id in model:
                arima_model = model[district_id]
                n_steps = len(forecast_df[forecast_df["district_id"] == district_id])
                try:
                    forecast = arima_model.forecast(steps=n_steps)
                    predictions.extend(forecast.tolist())
                except:
                    # Fallback to zero or mean
                    predictions.extend([0] * n_steps)
            else:
                predictions.extend([0] * len(forecast_df[forecast_df["district_id"] == district_id]))
        
        forecast_df["prediction"] = predictions
    else:
        # ML models (RF, XGBoost)
        forecast_df["prediction"] = model.predict(X_future)
    
    return forecast_df

def aggregate_forecasts(forecast_df, groupby="year"):
    """
    Aggregate forecasts by year or quarter.
    """
    if groupby == "year":
        summary = forecast_df.groupby(["district_id", "year"])["prediction"].sum().reset_index()
    elif groupby == "quarter":
        forecast_df["quarter"] = pd.PeriodIndex(forecast_df["date"], freq="Q")
        summary = forecast_df.groupby(["district_id", "quarter"])["prediction"].sum().reset_index()
    else:
        summary = forecast_df
    
    return summary

def plot_forecast_summary(forecast_df, output_dir="results"):
    """
    Create visualization of annual forecasts by district.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    annual_summary = aggregate_forecasts(forecast_df, groupby="year")
    
    # Pivot for plotting
    pivot_df = annual_summary.pivot(index="district_id", columns="year", values="prediction")
    
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", ax=plt.gca(), width=0.8, color=["skyblue", "salmon"])
    plt.title("Annual Forecasts by District (2025-2026)", fontsize=14, fontweight="bold")
    plt.xlabel("District ID", fontsize=12)
    plt.ylabel("Predicted Requisitions", fontsize=12)
    plt.legend(title="Year", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "forecast_2025_2026_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Forecast plot saved to {plot_path}")

def plot_monthly_forecast(forecast_df, sample_districts=5, output_dir="results"):
    """
    Plot monthly forecasts for a sample of districts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    districts = sorted(forecast_df["district_id"].unique())[:sample_districts]
    
    if len(districts) == 0:
        print("No districts to plot")
        return
    
    fig, axes = plt.subplots(len(districts), 1, figsize=(12, 3 * len(districts)))
    if len(districts) == 1:
        axes = [axes]
    
    for idx, district_id in enumerate(districts):
        df_d = forecast_df[forecast_df["district_id"] == district_id]
        
        axes[idx].plot(df_d["date"], df_d["prediction"], marker='o', linewidth=2, color='steelblue')
        axes[idx].set_title(f"District {district_id} - Monthly Forecast", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Date", fontsize=10)
        axes[idx].set_ylabel("Predicted Requisitions", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    monthly_plot_path = os.path.join(output_dir, "forecast_monthly_sample.png")
    plt.savefig(monthly_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Monthly forecast plot saved to {monthly_plot_path}")

def save_forecasts(forecast_df, output_dir="results"):
    """
    Save forecasts to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "forecast_2025_2026.csv")
    forecast_df[["district_id", "date", "year", "month", "prediction"]].to_csv(
        output_path, index=False
    )
    
    print(f"Forecasts saved to {output_path}")

def print_forecast_summary(forecast_df):
    """
    Print summary statistics of forecasts.
    """

    print("forecast summary performance results")    
    # Annual summary
    annual = aggregate_forecasts(forecast_df, groupby="year")
    print("\nTotal Predicted Requisitions by Year:")
    for year in sorted(annual["year"].unique()):
        total = annual[annual["year"] == year]["prediction"].sum()
        print(f"  {year}: {total:.0f}")
    
    # District summary
    print("\nAverage Monthly Prediction by District:")
    district_avg = forecast_df.groupby("district_id")["prediction"].mean().sort_values(ascending=False)
    for district_id, avg_pred in district_avg.items():
        print(f"  District {district_id}: {avg_pred:.2f}")
    
    # Overall stats
    print(f"\nOverall Statistics:")
    print(f"  Mean monthly prediction: {forecast_df['prediction'].mean():.2f}")
    print(f"  Std monthly prediction: {forecast_df['prediction'].std():.2f}")
    print(f"  Min monthly prediction: {forecast_df['prediction'].min():.2f}")
    print(f"  Max monthly prediction: {forecast_df['prediction'].max():.2f}")

def run_forecast_pipeline(
    featured_path: str,
    model_dir="models",
    output_dir="results",
    model_type="xgboost"
):
    """
    Full forecasting process:
    1. Load data and best model
    2. Generate future dates (2025-2026)
    3. Prepare forecast features
    4. Generate predictions
    5. Create visualizations
    6. Save results
    """
    print("Forecasting process (2025-2026)")  

    # Load data
    df, date_col = load_featured_data(featured_path)
    target_col = detect_target_column(df)
    
    # Load best model
    model, model_type = load_best_model(model_dir, model_type)
    
    # Generate future dates
    future_dates = generate_future_dates(start_year=2025, end_year=2026)
    print(f"\nForecasting period: {future_dates[0].date()} to {future_dates[-1].date()}")
    print(f"Total forecast points: {len(future_dates)} months")
    
    # Prepare features
    districts = sorted(df["district_id"].unique())
    print(f"Forecasting for {len(districts)} districts")
    
    forecast_df = prepare_forecast_features(df, target_col, date_col, districts, future_dates)
    
    # Generate predictions
    print(f"\nGenerating forecasts using {model_type}...")
    forecast_df = generate_forecasts(model, model_type, forecast_df, target_col)
    
    # Print summary
    print_forecast_summary(forecast_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_forecast_summary(forecast_df, output_dir)
    plot_monthly_forecast(forecast_df, min(5, len(districts)), output_dir)
    
    # Save results
    save_forecasts(forecast_df, output_dir)
    
    print(" Completed.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "results")
    
    run_forecast_pipeline(
        featured_path,
        model_dir,
        output_dir,
        model_type="xgboost" #xgboost perform better
    )