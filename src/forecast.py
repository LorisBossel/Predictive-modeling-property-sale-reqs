# Future Forecasting (2025-2026) script using best performing model with rolling forecast

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

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
    Load the best performing model.
    (XGBoost for the moment)
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
    frequency monthly start
    """
    date_range = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq=freq
    )
    return date_range

def get_historical_values(df: pd.DataFrame, district_id: int, target_col: str, 
                          date_col: str, n_values: int = 5) -> List[float]:
    """
    Get the last n historical values for a district.
    Returns list of values in chronological order (oldest to newest).
    """
    df_district = df[df["district_id"] == district_id].sort_values(date_col)
    
    if len(df_district) == 0:
        return [0] * n_values
    
    # Get last n values
    recent_values = df_district[target_col].tail(n_values).tolist()
    
    # Fill with zeros if not enough historical data
    while len(recent_values) < n_values:
        recent_values.insert(0, 0)
    
    return recent_values

def get_last_known_values(df: pd.DataFrame, district_id: int, target_col: str, date_col: str) -> pd.Series:
    """
    Get the last known row for a district.
    """
    df_d = df[df["district_id"] == district_id].sort_values(date_col)
    
    if len(df_d) == 0:
        return None
    
    last_row = df_d.iloc[-1].copy()
    return last_row

def prepare_forecast_features_rolling(
    df: pd.DataFrame, 
    target_col: str, 
    date_col: str, 
    districts: List[int], 
    future_dates: pd.DatetimeIndex,
    model,
    model_type: str
) -> pd.DataFrame:
    """
    Future forecasting with rolling updtes.
    update lag features with the new predictions for the next month.
    """
    # Get year_min for year_trend calculation
    year_min = df["year"].min() if "year" in df.columns else df[date_col].dt.year.min()
    
    # Identify feature columns used in training
    feature_cols = [
        col for col in df.columns 
        if "lag" in col or "rolling" in col or col in [
            "taux_de_logements_vacants_en_%", "month_sin", "month_cos", "year_trend"
        ]
    ]
    
    print(f"Using {len(feature_cols)} features")
    
    # Dictionary to store prediction history for each district
    prediction_history: Dict[int, List[float]] = {}
    
    # Initialize with historical values
    for district_id in districts:
        historical = get_historical_values(df, district_id, target_col, date_col, n_values=5)
        prediction_history[district_id] = historical
        
    # Store all forecast records
    all_forecasts = []
    
    # Process month by month
    print(f"Processing {len(future_dates)} months:")
    
    for month_idx, future_date in enumerate(future_dates):
        month_forecasts = []
        
        # Extract temporal features for this month
        month = future_date.month
        year = future_date.year
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        year_trend = year - year_min
        
        # Prepare features for all districts for this month
        X_month = []
        district_order = []
        
        for district_id in districts:
            last_row = get_last_known_values(df, district_id, target_col, date_col)
            
            if last_row is None:
                continue
            
            # Get current prediction history for this district
            history = prediction_history[district_id]
            
            # Build features dictionary
            features = {}
            
            # Lag features: use most recent values from prediction history
            if f"{target_col}_lag_3" in feature_cols:
                features[f"{target_col}_lag_3"] = history[-3] if len(history) >= 3 else 0
            
            if f"{target_col}_lag_4" in feature_cols:
                features[f"{target_col}_lag_4"] = history[-4] if len(history) >= 4 else 0
            
            if f"{target_col}_lag_5" in feature_cols:
                features[f"{target_col}_lag_5"] = history[-5] if len(history) >= 5 else 0
            
            # Rolling features: use last known rolling values
            for col in feature_cols:
                if "rolling" in col and col not in features:
                    features[col] = last_row.get(col, 0)
            
            # Vacancy rate: use last known value
            if "taux_de_logements_vacants_en_%" in feature_cols:
                features["taux_de_logements_vacants_en_%"] = last_row.get(
                    "taux_de_logements_vacants_en_%", 0
                )
            
            # Temporal features
            features["month_sin"] = month_sin
            features["month_cos"] = month_cos
            features["year_trend"] = year_trend
            
            feature_vector = [features.get(col, 0) for col in feature_cols]
            X_month.append(feature_vector)
            district_order.append(district_id)
            
            # Store metadata for output
            month_forecasts.append({
                "district_id": district_id,
                "date": future_date,
                "year": year,
                "month": month,
            })
        
        # Make predictions for this month (all districts at once)
        if len(X_month) > 0:
            X_month_df = pd.DataFrame(X_month, columns=feature_cols)
            
            if model_type == "arima":
                # ARIMA: predict per district
                predictions = []
                for district_id in district_order:
                    if district_id in model:
                        try:
                            pred = model[district_id].forecast(steps=1)[0]
                            predictions.append(pred)
                        except:
                            predictions.append(0)
                    else:
                        predictions.append(0)
            else:
                # ML models: batch prediction
                predictions = model.predict(X_month_df)
            
            # Update prediction history and store results
            for i, (district_id, prediction) in enumerate(zip(district_order, predictions)):
                # Add prediction to history (for next month's lag features)
                prediction_history[district_id].append(float(prediction))
                
                # Keep only last 5 values (for lag_5)
                if len(prediction_history[district_id]) > 5:
                    prediction_history[district_id].pop(0)
                
                # Add prediction to output
                month_forecasts[i]["prediction"] = float(prediction)
            
            all_forecasts.extend(month_forecasts)
        
        # Progress indicator
        if (month_idx + 1) % 6 == 0 or month_idx == len(future_dates) - 1:
            print(f"  Processed {month_idx + 1}/{len(future_dates)} months")
    
    forecast_df = pd.DataFrame(all_forecasts)
    
    print(f"Generated {len(forecast_df)} forecasts")
    print("Lag features updated dynamically based on predictions")
    
    return forecast_df

def aggregate_forecasts(forecast_df: pd.DataFrame, groupby="year") -> pd.DataFrame:
    """
    Aggregate forecasts by year (also possible by quarter).
    """
    if groupby == "year":
        summary = forecast_df.groupby(["district_id", "year"])["prediction"].sum().reset_index()
    elif groupby == "quarter":
        forecast_df["quarter"] = pd.PeriodIndex(forecast_df["date"], freq="Q")
        summary = forecast_df.groupby(["district_id", "quarter"])["prediction"].sum().reset_index()
    else:
        summary = forecast_df
    
    return summary

def plot_forecast_summary(forecast_df: pd.DataFrame, output_dir="results"):
    """
    Create visualization of forecasts by district.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    annual_summary = aggregate_forecasts(forecast_df, groupby="year")
    
    pivot_df = annual_summary.pivot(index="district_id", columns="year", values="prediction")
    
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", ax=plt.gca(), width=0.8, color=["skyblue", "salmon"])
    plt.title("Annual Forecasts by District (2025-2026) - Rolling Forecast", 
              fontsize=14, fontweight="bold")
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

def plot_monthly_forecast(forecast_df: pd.DataFrame, sample_districts=5, 
                         output_dir="results"):
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
        df_d = forecast_df[forecast_df["district_id"] == district_id].sort_values("date")
        
        axes[idx].plot(df_d["date"], df_d["prediction"], 
                      marker='o', linewidth=2, color='steelblue', markersize=5)
        axes[idx].set_title(f"District {district_id} - Monthly Rolling Forecast", 
                          fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Date", fontsize=10)
        axes[idx].set_ylabel("Predicted Requisitions", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvline(pd.Timestamp('2025-12-31'), color='red', 
                         linestyle='--', alpha=0.5, label='2025/2026')
        axes[idx].legend()
    
    plt.tight_layout()
    
    monthly_plot_path = os.path.join(output_dir, "forecast_monthly_sample.png")
    plt.savefig(monthly_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Monthly forecast plot saved to {monthly_plot_path}")

def save_forecasts(forecast_df: pd.DataFrame, output_dir="results"):
    """
    Save forecasts to CSV, to have trace of values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "forecast_2025_2026.csv")
    forecast_df[["district_id", "date", "year", "month", "prediction"]].to_csv(
        output_path, index=False
    )
    
    print(f"Forecasts saved to {output_path}")

def print_forecast_summary(forecast_df: pd.DataFrame):
    """
    Print summary statistics of forecasts.
    """
    print("\nForecast summary:")
    
    # Annual summary
    annual = aggregate_forecasts(forecast_df, groupby="year")
    print("\nTotal predicted requisitions by year:")
    for year in sorted(annual["year"].unique()):
        total = annual[annual["year"] == year]["prediction"].sum()
        print(f"  {year}: {total:.0f}")
    
    # Year over year change
    years = sorted(annual["year"].unique())
    if len(years) == 2:
        total_2025 = annual[annual["year"] == years[0]]["prediction"].sum()
        total_2026 = annual[annual["year"] == years[1]]["prediction"].sum()
        yoy_change = ((total_2026 - total_2025) / total_2025) * 100
        print(f"  Year-over-year change: {yoy_change:+.1f}%")
    
    # District summary
    print("\nAverage monthly prediction by district:")
    district_avg = forecast_df.groupby("district_id")["prediction"].mean().sort_values(
        ascending=False
    )
    for district_id, avg_pred in district_avg.items():
        pct_of_total = (avg_pred / district_avg.sum()) * 100
        print(f"  District {district_id}: {avg_pred:.2f}")
    
    # Overall stats
    print(f"\nOverall statistics:")
    print(f"  Mean monthly prediction: {forecast_df['prediction'].mean():.2f}")
    print(f"  Median monthly prediction: {forecast_df['prediction'].median():.2f}")
    print(f"  Min monthly prediction: {forecast_df['prediction'].min():.2f}")
    print(f"  Max monthly prediction: {forecast_df['prediction'].max():.2f}")
    
    # Coefficient of variation
    cv = (forecast_df['prediction'].std() / forecast_df['prediction'].mean()) * 100
    print(f"  Coefficient of variation: {cv:.1f}%")

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
    4. Generate predictions month by month
    5. Compare with historical data
    """
    print("Rolling forecast pipeline (2025-2026):")
    
    # Load data
    df, date_col = load_featured_data(featured_path)
    target_col = detect_target_column(df)
    
    # Load best model
    model, model_type = load_best_model(model_dir, model_type)
    
    # Generate future dates
    future_dates = generate_future_dates(start_year=2025, end_year=2026)
    print(f"Forecasting period: {future_dates[0].date()} to {future_dates[-1].date()}")
    print(f"Total forecast points: {len(future_dates)} months")
    
    # Get districts
    districts = sorted(df["district_id"].unique())
    print(f"Forecasting for {len(districts)} districts")
    
    # Generate forecasts
    print(f"Generating forecasts using {model_type}:")
    forecast_df = prepare_forecast_features_rolling(
        df, target_col, date_col, districts, future_dates, model, model_type
    )
    
    # Print summary
    print_forecast_summary(forecast_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_forecast_summary(forecast_df, output_dir)
    plot_monthly_forecast(forecast_df, min(5, len(districts)), output_dir)
    
    save_forecasts(forecast_df, output_dir)
    
    print("\nForecast completed.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "results")
    
    run_forecast_pipeline(
        featured_path,
        model_dir,
        output_dir,
        model_type="xgboost"
    )