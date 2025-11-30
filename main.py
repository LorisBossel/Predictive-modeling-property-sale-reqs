"""
Main Script, pipeline of Property Sale Requisitions Forecasting in Canton Fribourg.

Entry point for the ML pipeline:
1. Data preprocessing (preprocessing.py)
2. Feature engineering (feature_engineering.py)
3. Model training (model.py)
4. Model evaluation (evaluation.py)
5. Rolling forecasts (forecast.py)
"""
import os
import argparse

from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model import run_training_pipeline
from src.evaluation import run_evaluation
from src.forecast import run_forecast_pipeline

def main():
    """
    Run the complete ML pipeline.
    """
    print("Property Sale Requisitions Forecasting:\n")

    parser = argparse.ArgumentParser(
        description="Property Sale Requisitions Forecasting Pipeline"
    )
    args = parser.parse_args()

    # Define paths
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")

    # Raw data paths
    poursuites_path = os.path.join(data_dir, "raw", "06_02_conjoncture_poursuites.csv")
    vacants_path = os.path.join(data_dir, "raw", "09_03_log_vacants_taux_des_1975.csv")

    # Processed data paths
    processed_path = os.path.join(data_dir, "processed", "processed_dataset.csv")
    featured_path = os.path.join(data_dir, "featured", "featured_dataset.csv")

    # score symbols to identify steps during pipeline execution.

    print("\nSTEP 1: Data preprocessing-----------------------------------\n")
    preprocess_data(poursuites_path, vacants_path, processed_path)

    print("\nSTEP 2: Feature engineering----------------------------------\n")
    feature_engineering(processed_path, featured_path)

    print("\nSTEP 3: Model training---------------------------------------\n")
    run_training_pipeline(featured_path, model_dir)

    print("\nSTEP 4: Model evaluation-------------------------------------\n")
    run_evaluation(featured_path, model_dir, results_dir)

    print("\nSTEP 5: Demo forecast generation (2025-2026)-----------------\n")
    run_forecast_pipeline(featured_path, model_dir, results_dir)

    print("\nPipeline complete!")

if __name__ == "__main__":
    main()