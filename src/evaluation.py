# Models Evaluation script (with Temporal Validation) 

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading
def load_dataset(path):
    """Load dataset and detect date column."""
    # Read CSV file with proper encoding
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    
    # Try to find the date column (different possible names)
    for col in ["date_poursuites"]:
        if col in df.columns:
            # Convert to datetime format
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return df, col
    
    # If no date column found, raise error
    raise ValueError(f"No date column found in: {df.columns.tolist()}")

def ensure_split_column(df, date_col):
    """Create split column (train/validation/test)."""
    # Check if split column already exists
    if "split" in df.columns:
        return df
    
    print("Creating temporal splits...")
    
    # Extract year from date 
    if "year" not in df.columns:
        df["year"] = df[date_col].dt.year
    
    # Create temporal splits based on year ranges
    df["split"] = "train"  # Default: everything is train
    df.loc[df["year"].between(2016, 2020), "split"] = "validation"  # 2016-2020: validation
    df.loc[df["year"] >= 2021, "split"] = "test"  # 2021+: test
    
    # Show how many rows in each split
    print("Split counts:", df["split"].value_counts().to_dict())
    return df

def detect_target(df):
    """Detect target column from common names."""
    # List of possible target column names (different encodings)
    targets = ["réquisitions_de_vente", "requisitions_de_vente", 
               "réquisitions_de_vente", "RÃ©quisitions de vente"]
    
    # Try each name until one is found
    for target in targets:
        if target in df.columns:
            return target
    
    # If none found, raise error
    raise ValueError(f"Target not found in: {df.columns.tolist()}")

def load_models(model_dir):
    """Load all trained models."""
    # Define paths to the 3 model files
    paths = {
        "rf": os.path.join(model_dir, "random_forest.pkl"),
        "xgb": os.path.join(model_dir, "xgboost.pkl"),
        "arima": os.path.join(model_dir, "arima_models.pkl")
    }
    
    # Check if all model files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} model not found: {path}")
    
    # Load all models and return as dictionary
    return {
        "rf": joblib.load(paths["rf"]),
        "xgb": joblib.load(paths["xgb"]),
        "arima": joblib.load(paths["arima"])
    }

# Prepare features to perform evaluatio using the models
def get_feature_columns(df):
    """Extract feature column names."""
    # Get all columns that are features
    return [col for col in df.columns 
            if "lag" in col or "rolling" in col or 
            col in ["month_sin", "month_cos", "year_trend", 
                    "taux_de_logements_vacants_en_%"]]

def prepare_split_data(df, target_col, split_name):
    """Prepare X, y for a specific split."""
    # Filter data for this split (train/validation/test)
    df_split = df[df["split"] == split_name].copy()
    
    # Get list of feature columns
    feature_cols = get_feature_columns(df)
    
    # Create feature matrix X (fill missing values with 0)
    X = df_split[feature_cols].fillna(0)
    
    # Create target vector y
    y = df_split[target_col]
    
    return X, y, df_split

def predict_ml_model(model, X):
    """Generate predictions from ML model."""
    # Use model to predict (works for RF and XGBoost)
    return model.predict(X)

def predict_arima(arima_models, df_split, target_col, date_col):
    """Generate ARIMA predictions with fallback."""
    predictions = []
    
    # Loop through each district
    for district_id in sorted(df_split["district_id"].unique()):
        # Get data for this district
        df_d = df_split[df_split["district_id"] == district_id]
        n_steps = len(df_d)
        
        # Check if we have an ARIMA model for this district
        if district_id in arima_models:
            try:
                # Sort by date
                df_d = df_d.sort_values(date_col)
                # Generate forecast
                forecast = arima_models[district_id].forecast(steps=n_steps)
                predictions.extend(forecast.tolist())
                continue
            except:
                pass
        
        # Fallback: if ARIMA fails or doesn't exist, use mean value
        predictions.extend([df_d[target_col].mean()] * n_steps)
    
    return np.array(predictions)

def compute_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    # Mean Absolute Error (average prediction error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error (penalizes large errors)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error (error in %)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    # R-squared (how much variance is explained)
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def evaluate_models(df, models, target_col, date_col):
    """Evaluate all models on validation and test sets."""
    results = []
    
    # Evaluate on both validation and test sets
    for split in ["validation", "test"]:
        # Prepare data for this split
        X, y, df_split = prepare_split_data(df, target_col, split)
        
        # Skip if split is empty
        if len(X) == 0:
            print(f"WARNING: {split} set is empty")
            continue
        
        # Evaluate Random Forest
        preds_rf = predict_ml_model(models["rf"], X)
        metrics_rf = compute_metrics(y, preds_rf)
        results.append({"Model": "Random Forest", "Split": split, **metrics_rf})
        
        # Evaluate XGBoost
        preds_xgb = predict_ml_model(models["xgb"], X)
        metrics_xgb = compute_metrics(y, preds_xgb)
        results.append({"Model": "XGBoost", "Split": split, **metrics_xgb})
        
        # Evaluate ARIMA
        preds_arima = predict_arima(models["arima"], df_split, target_col, date_col)
        metrics_arima = compute_metrics(y, preds_arima)
        results.append({"Model": "ARIMA", "Split": split, **metrics_arima})
        
        # Print results for this split
        print(f"{split.upper()}: RF={metrics_rf['MAE']:.1f}, "
              f"XGB={metrics_xgb['MAE']:.1f}, ARIMA={metrics_arima['MAE']:.1f}")
    
    # Convert results list to DataFrame
    return pd.DataFrame(results)

# Visualize of the results with help of 2 plots
def plot_comparison(results_df, output_dir):
    """Create comparison plots for all metrics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Create 4 plots (one for each metric)
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Create bar chart for each metric
    for idx, metric in enumerate(metrics):
        # Reshape data for plotting (models as rows, splits as columns)
        pivot = results_df.pivot(index="Model", columns="Split", values=metric)
        
        # Create bar chart
        pivot.plot(kind="bar", ax=axes[idx], color=["skyblue", "salmon"])
        axes[idx].set_title(f"{metric} Comparison")
        axes[idx].set_ylabel(metric)
        axes[idx].grid(alpha=0.3, axis='y')
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    # Save plot to file
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

def plot_val_vs_test(results_df, output_dir):
    """Plot validation vs test MAE."""
    # Get validation and test data separately
    val = results_df[results_df["Split"] == "validation"][["Model", "MAE"]]
    test = results_df[results_df["Split"] == "test"][["Model", "MAE"]]
    
    # Skip if either is empty
    if len(val) == 0 or len(test) == 0:
        return
    
    # Merge validation and test MAE for each model
    merged = pd.merge(val, test, on="Model", suffixes=("_Val", "_Test"))
    
    # Create scatter plot
    plt.figure(figsize=(7, 6))
    for _, row in merged.iterrows():
        # Plot point for each model
        plt.scatter(row["MAE_Val"], row["MAE_Test"], s=150)
        # Add model name as label
        plt.text(row["MAE_Val"], row["MAE_Test"], row["Model"], 
                fontsize=9, ha='center', va='bottom')
    
    # Add diagonal line (perfect agreement between val and test)
    max_val = max(merged["MAE_Val"].max(), merged["MAE_Test"].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    plt.xlabel("Validation MAE")
    plt.ylabel("Test MAE")
    plt.title("Validation vs Test Performance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Save plot to file
    plt.savefig(os.path.join(output_dir, "val_vs_test.png"), dpi=150)
    plt.close()

# Write a csv with the results
def save_results(results_df, output_dir):
    """Save results in CSV file."""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV file
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

def print_summary(results_df):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    # Print full results table
    print(results_df.to_string(index=False))
    
    # Find and print best model (lowest MAE on validation)
    val = results_df[results_df["Split"] == "validation"]
    if len(val) > 0:
        best_idx = val["MAE"].idxmin()
        best = val.loc[best_idx]
        print(f"\nBest Model: {best['Model']} (Val MAE: {best['MAE']:.2f})")

#  Full evaluation process
def run_evaluation(featured_path, model_dir="models", output_dir="results"):
    """Execute full evaluation pipeline."""
    # Step 1: Load and prepare data
    print("Loading data...")
    df, date_col = load_dataset(featured_path)
    df = ensure_split_column(df, date_col)
    target_col = detect_target(df)
    
    print(f"Dataset: {len(df)} rows, {df['district_id'].nunique()} districts")
    print(f"Target: {target_col}")
    
    # Step 2: Load trained models
    print("\nLoading models...")
    models = load_models(model_dir)
    print(f"Loaded: RF, XGBoost, ARIMA ({len(models['arima'])} districts)")
    
    # Step 3: Evaluate all models
    print("\nEvaluating models...")
    results_df = evaluate_models(df, models, target_col, date_col)
    
    # Check if evaluation succeeded
    if len(results_df) == 0:
        print("ERROR: No results generated")
        return
    
    # Step 4: Create visualizations
    print("\nGenerating visualizations...")
    plot_comparison(results_df, output_dir)
    plot_val_vs_test(results_df, output_dir)
    
    # Step 5: Save results
    print("\nSaving results...")
    save_results(results_df, output_dir)
    
    # Step 6: Print summary
    print_summary(results_df)
    print(f"\nResults saved to: {output_dir}/")

# Run the script
if __name__ == "__main__":
    # Set up paths 
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "results")
    
    # Execute the evaluation
    run_evaluation(featured_path, model_dir, output_dir)