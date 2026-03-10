"""Data loading and basic validation utilities.

Separates raw I/O from all ML logic so the rest of the pipeline
can assume a clean, validated DataFrame.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    BASELINE_NUMERIC_FEATURES,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load raw apartment data from a CSV or Excel file.

    Args:
        filepath: Path to the raw data file. Defaults to RAW_DATA_FILE
                  defined in config.py.

    Returns:
        Raw, unmodified DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if filepath is None:
        filepath = RAW_DATA_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            "Place your dataset in data/raw/ and update RAW_DATA_FILE in config.py."
        )

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Use .csv or .xlsx.")

    print(f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} columns from '{filepath.name}'")
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Assert that all required columns are present in the DataFrame.

    Raises:
        ValueError: With a clear message listing missing columns.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal, safe cleaning steps to the raw DataFrame.

    Steps:
    1. Drop exact duplicate rows.
    2. Drop rows where the target column is missing.
    3. Drop rows where the price is non-positive (target leakage guard).

    TODO: Add domain-specific filters once you know your dataset:
          - e.g. df = df[df["area"] > 10]          to remove implausible areas
          - e.g. df = df[df["rooms"].between(1, 20)] to clip room outliers
          - e.g. df = df[df[TARGET_COLUMN] < 20_000]  to remove extreme outliers

    Args:
        df: Raw DataFrame from load_raw_data().

    Returns:
        Cleaned DataFrame with reset index.
    """
    n_raw = len(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    n_clean = len(df)
    print(
        f"[data_loader] Cleaned: {n_raw:,} → {n_clean:,} rows "
        f"({n_raw - n_clean:,} removed)"
    )
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke test: load and clean, then print a summary
    df_raw = load_raw_data()
    validate_columns(df_raw, required_columns=[TARGET_COLUMN] + BASELINE_NUMERIC_FEATURES)
    df_clean = basic_clean(df_raw)
    print(df_clean.describe())
