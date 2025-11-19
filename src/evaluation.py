# Models Evaluation script (with Temporal Validation)
"""
Model evaluation module.

Computes performance metrics (MAE, RMSE, R²) and generates comparison
visualizations across validation and test sets.
"""


import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_dataset(path):
    """Load dataset and detect date column."""
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
    df["split"] = "train"
    df.loc[df["year"].between(2016, 2020), "split"] = "validation"
    df.loc[df["year"] >= 2021, "split"] = "test"

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

def load_training_history(model_dir):
    """Load training history for RF and XGBoost."""
    histories = {}

    # Try to load RF history
    rf_history_path = os.path.join(model_dir, "rf_training_history.pkl")
    if os.path.exists(rf_history_path):
        histories['rf'] = joblib.load(rf_history_path)
        print("   Loaded Random Forest training history")
    else:
        print("   Warning: RF training history not found")

    # Try to load XGBoost history
    xgb_history_path = os.path.join(model_dir, "xgb_training_history.pkl")
    if os.path.exists(xgb_history_path):
        histories['xgb'] = joblib.load(xgb_history_path)
        print("   Loaded XGBoost training history")
    else:
        print("   Warning: XGBoost training history not found")

    return histories

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
            print(f"Warning: {split} set is empty")
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

def plot_training_curves(histories, output_dir):
    """
    Plot training curves for Random Forest and XGBoost.
    Shows train vs validation MAE over training iterations.
    """
    if not histories:
        print("   No training history found. Skipping training curves plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Determine how many subplots we need
    n_models = len(histories)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))

    # If only one model, axes is not a list
    if n_models == 1:
        axes = [axes]

    colors = {'train': '#3498db', 'val': '#e74c3c'}

    for idx, (model_name, history) in enumerate(histories.items()):
        ax = axes[idx]

        if model_name == 'rf':
            # Random Forest plot
            x_values = history['n_estimators']
            title = "Random Forest Training Progress"
            xlabel = "Number of Trees (n_estimators)"
        else:
            # XGBoost plot
            x_values = history['iteration']
            title = "XGBoost Training Progress"
            xlabel = "Iteration (Boosting Rounds)"

        # Plot train and validation curves
        ax.plot(x_values, history['train_mae'],
                label='Train MAE', color=colors['train'],
                linewidth=2, marker='o', markersize=4)
        ax.plot(x_values, history['val_mae'],
                label='Validation MAE', color=colors['val'],
                linewidth=2, marker='s', markersize=4)

        # Styling
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate final values
        final_train = history['train_mae'][-1]
        final_val = history['val_mae'][-1]
        textstr = f"Final Train MAE: {final_train:.2f}\nFinal Val MAE: {final_val:.2f}"
        ax.text(0.98, 0.98, textstr,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Training curves saved to {plot_path}")

def save_results(results_df, output_dir):
    """Save results in CSV file."""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV file
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

def print_summary(results_df):
    """Print evaluation summary."""
    print("\nEvaluation summary")

    # Print full results table
    print(results_df.to_string(index=False))

    # Find and print best model (lowest MAE on validation)
    val = results_df[results_df["Split"] == "validation"]
    if len(val) > 0:
        best_idx = val["MAE"].idxmin()
        best = val.loc[best_idx]
        print(f"\nBest model: {best['Model']} (Val MAE: {best['MAE']:.2f})")

def run_evaluation(featured_path, model_dir="models", output_dir="results"):
    """Execute full evaluation pipeline."""
    # Load and prepare data
    df, date_col = load_dataset(featured_path)
    df = ensure_split_column(df, date_col)
    target_col = detect_target(df)

    print(f"Dataset: {len(df)} rows, {df['district_id'].nunique()} districts")
    print(f"Target: {target_col}")

    # Load trained models
    print("\nLoading models:")
    models = load_models(model_dir)
    print(f"Loaded: RF, XGBoost, ARIMA ({len(models['arima'])} districts)")

    # Load training histories
    print("\nLoading training histories:")
    histories = load_training_history(model_dir)

    # Evaluate all models
    print("\nEvaluating models:")
    results_df = evaluate_models(df, models, target_col, date_col)

    # Check if evaluation OK
    if len(results_df) == 0:
        print("Error: No results generated")
        return

    # Create visualizations
    print("\nGenerating visualizations:")
    plot_comparison(results_df, output_dir)
    plot_val_vs_test(results_df, output_dir)

    # Training curves plot for see curve of train
    if histories:
        plot_training_curves(histories, output_dir)

    # Save results
    print("\nSaving results:")
    save_results(results_df, output_dir)

    print_summary(results_df)
    print(f"\nResults saved to: {output_dir}/")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    featured_path = os.path.join(base_dir, "data", "featured", "featured_dataset.csv")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "results")

    run_evaluation(featured_path, model_dir, output_dir)
