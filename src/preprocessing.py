# Preprocessing script

import pandas as pd
import os
from typing import Tuple


def load_raw_data(poursuites_path: str, vacants_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV files for both datasets using ';' as separator and utf-8-sig encoding.
    """
    df_poursuites = pd.read_csv(poursuites_path, sep=";", encoding="utf-8-sig", low_memory=False)
    df_vacants = pd.read_csv(vacants_path, sep=";", encoding="utf-8-sig", low_memory=False)

    print("Poursuites columns:", df_poursuites.columns.tolist())
    print("Vacants columns:", df_vacants.columns.tolist())

    return df_poursuites, df_vacants

def clean_poursuites(df_poursuites: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the 'Poursuites' dataset.
    """
    # Normalize column names
    df_poursuites.columns = (
        df_poursuites.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    # Rename the key column for consistency
    df_poursuites = df_poursuites.rename(columns={"district_id": "district_id", "district_id": "district_id"})

    # Keep only relevant columns
    columns_to_keep = [
        "date",
        "district_id",
        "district",
        "bezirk",
        "réquisitions_de_poursuite",
        "réquisitions_de_continuer_la_poursuite",
        "réquisitions_de_vente",
        "actes_de_défaut_de_biens"
    ]
    df_poursuites = df_poursuites[columns_to_keep]

    # Convert Date column to datetime
    df_poursuites["date"] = pd.to_datetime(df_poursuites["date"], errors="coerce")

    # Extract year and month
    df_poursuites["year"] = df_poursuites["date"].dt.year
    df_poursuites["month"] = df_poursuites["date"].dt.month

    # Drop rows with missing dates
    df_poursuites = df_poursuites.dropna(subset=["date"])

    return df_poursuites

def clean_vacants(df_vacants: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the 'Vacants' dataset.
    """
    # Normalize column names
    df_vacants.columns = (
        df_vacants.columns.str.strip().str.lower().str.replace(" ", "_")
    )

    # Keep only relevant columns
    columns_to_keep = [
        "année",
        "district_id",
        "district",
        "bezirk",
        "commune_id_hist.",
        "commune_id",
        "commune",
        "parc_de_logements_au_31_décembre",
        "nombre_de_logements_vacants",
        "taux_de_logements_vacants_en_%"
    ]
    df_vacants = df_vacants[columns_to_keep]

    # Convert the year column to datetime (January 1st)
    df_vacants["date"] = pd.to_datetime(df_vacants["année"].astype(str) + "-01-01", errors="coerce")

    # Extract year
    df_vacants["year"] = df_vacants["date"].dt.year

    # Drop rows with missing values
    df_vacants = df_vacants.dropna(subset=["district_id", "année"])

    return df_vacants

def merge_datasets(df_poursuites: pd.DataFrame, df_vacants: pd.DataFrame) -> pd.DataFrame:
    """
    Merge both datasets on district_id and year.
    """
    df_merged = pd.merge(
        df_poursuites,
        df_vacants,
        on=["district_id", "year"],
        how="left",
        suffixes=("_poursuites", "_vacants")
    )
    return df_merged

def preprocess_data(poursuites_path: str, vacants_path: str, output_path: str = "data/processed/merged_dataset.csv") -> pd.DataFrame:
    """
    Full preprocessing pipeline combining loading, cleaning, and merging.
    The final dataset is also saved to the specified output path.
    """
    df_poursuites, df_vacants = load_raw_data(poursuites_path, vacants_path)
    df_poursuites = clean_poursuites(df_poursuites)
    df_vacants = clean_vacants(df_vacants)
    df_merged = merge_datasets(df_poursuites, df_vacants)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Processed data saved to: {output_path}")

    return df_merged


if __name__ == "__main__":
    # Define relative paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")

    path_poursuites = os.path.join(data_dir, "raw", "06_02_conjoncture_poursuites.csv")
    path_vacants = os.path.join(data_dir, "raw", "09_03_log_vacants_taux_des_1975.csv")
    output_path = os.path.join(data_dir, "processed", "processed_dataset.csv")

    df_final = preprocess_data(path_poursuites, path_vacants, output_path)
    print("Final dataset rows:", df_final.head(n=5))
