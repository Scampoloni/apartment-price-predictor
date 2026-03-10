"""Feature engineering functions for the apartment price predictor.

Design rules:
- Every function accepts a DataFrame, creates new column(s), and returns a copy.
- Functions never modify the original DataFrame in-place.
- If a source column is missing, the derived feature defaults to 0 / NaN
  so training and inference never break on partial input.
- engineer_baseline_features() — identity transform, used in Iteration 1.
- engineer_all_features()       — full pipeline, used in Iteration 2.
"""

import re
from typing import List

import numpy as np
import pandas as pd

from src.config import DESCRIPTION_COLUMN, LOCATION_COLUMN


# ── Helper ─────────────────────────────────────────────────────────────────────

def _flag_from_keywords(series: pd.Series, keywords: List[str]) -> pd.Series:
    """Return a binary int Series: 1 if any keyword appears in the text.

    The search is case-insensitive and applied to the lowercased text.
    Missing values are treated as empty strings (no match → 0).
    """
    pattern = "|".join(re.escape(kw) for kw in keywords)
    return (
        series.fillna("")
        .str.lower()
        .str.contains(pattern, regex=True)
        .astype(int)
    )


# ── Numeric derived features ───────────────────────────────────────────────────

def add_rooms_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Add rooms_per_m2 = rooms / area.

    Captures "room density": a studio of 30 m² has a very different density
    than a 4-room flat of 30 m².  Helps the model distinguish these cases.

    TODO: Verify that your dataset uses 'rooms' and 'area' as column names.
    """
    df = df.copy()
    if "rooms" in df.columns and "area" in df.columns:
        df["rooms_per_m2"] = df["rooms"] / df["area"].replace(0, np.nan)
    else:
        df["rooms_per_m2"] = np.nan
    return df


# ── Text-based binary flags ────────────────────────────────────────────────────

def add_furnished_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_furnished = 1 if the description mentions furnished / möbliert.

    Furnished apartments command a rent premium of ~15–30 % in Switzerland.
    """
    df = df.copy()
    if DESCRIPTION_COLUMN in df.columns:
        keywords = [
            "furnished", "möbliert", "moebliert", "meublé",
            "inkl. möbel", "inkl. moebel",
        ]
        df["is_furnished"] = _flag_from_keywords(df[DESCRIPTION_COLUMN], keywords)
    else:
        df["is_furnished"] = 0
    return df


def add_temporary_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_temporary = 1 if the listing is advertised as short-term / befristet.

    Temporary rentals (Zwischenmiete, Untermiete) typically have lower or
    artificially constrained prices.
    """
    df = df.copy()
    if DESCRIPTION_COLUMN in df.columns:
        keywords = [
            "temporary", "befristet", "zwischenmiete", "untermiete",
            "on demand", "auf anfrage", "zeitlich beschränkt",
        ]
        df["is_temporary"] = _flag_from_keywords(df[DESCRIPTION_COLUMN], keywords)
    else:
        df["is_temporary"] = 0
    return df


def add_balcony_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_balcony = 1 if the description mentions a balcony or terrace.

    Outdoor space is a well-known value driver in Swiss apartment markets.
    """
    df = df.copy()
    if DESCRIPTION_COLUMN in df.columns:
        keywords = ["balkon", "balcony", "terrasse", "terrace", "loggia", "sitzplatz"]
        df["has_balcony"] = _flag_from_keywords(df[DESCRIPTION_COLUMN], keywords)
    else:
        df["has_balcony"] = 0
    return df


def add_luxurious_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_luxurious = 1 if the description contains luxury-segment keywords.

    Targets the upper price tail; prevents average models from under-pricing
    high-end listings.
    """
    df = df.copy()
    if DESCRIPTION_COLUMN in df.columns:
        keywords = [
            "luxury", "luxus", "luxuriös", "exklusiv", "exclusive",
            "premium", "penthouse", "hochwertig", "first class",
            "designer", "villa",
        ]
        df["is_luxurious"] = _flag_from_keywords(df[DESCRIPTION_COLUMN], keywords)
    else:
        df["is_luxurious"] = 0
    return df


# ── Location-based features ────────────────────────────────────────────────────

def add_zurich_district_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_zurich_city = 1 if the apartment is in the city of Zurich.

    Zurich city rents are significantly above the cantonal average.
    This is a coarse binary proxy; replace with a district-level mapping
    if geodata or a district column is available.

    TODO: Consider replacing with a quantitative feature such as:
          - average municipality rent (join from external table), or
          - km distance from Zurich HB (requires coordinates).
    """
    df = df.copy()
    if LOCATION_COLUMN in df.columns:
        df["is_zurich_city"] = (
            df[LOCATION_COLUMN]
            .fillna("")
            .str.lower()
            .str.contains(r"\bzürich\b|\bzurich\b", regex=True)
            .astype(int)
        )
    else:
        df["is_zurich_city"] = 0
    return df


# ── Composed pipelines ─────────────────────────────────────────────────────────

def engineer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the DataFrame unchanged — Iteration 1 uses raw columns only."""
    return df.copy()


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps for Iteration 2.

    Order matters: numeric features must be computed before the
    preprocessor sees the DataFrame.
    """
    df = add_rooms_per_m2(df)
    df = add_furnished_flag(df)
    df = add_temporary_flag(df)
    df = add_balcony_flag(df)
    df = add_luxurious_flag(df)
    df = add_zurich_district_flag(df)
    return df


# ── Runtime feature-list resolver ─────────────────────────────────────────────

def get_feature_lists(
    df_engineered: pd.DataFrame,
    iteration: int,
) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, categorical, binary) feature lists for the given iteration.

    Called *after* feature engineering so engineered columns (rooms_per_m2,
    flags, …) are already present in df_engineered.  Only includes columns
    that actually exist in the DataFrame — missing optionals are silently
    dropped with a warning so the pipeline never crashes on partial data.

    Args:
        df_engineered: DataFrame after the relevant engineer_*_features() call.
        iteration:     1 (baseline) or 2 (improved).

    Returns:
        Tuple of three lists: (numeric_features, categorical_features, binary_features).

    Raises:
        ValueError: If no numeric features are found (rooms/area both missing).
    """
    available = set(df_engineered.columns)

    def _pick(candidates: list[str], label: str) -> list[str]:
        found = [c for c in candidates if c in available]
        dropped = [c for c in candidates if c not in available]
        if dropped:
            print(f"[features] {label}: column(s) {dropped} not found — skipped")
        return found

    if iteration == 1:
        numeric     = _pick(["rooms", "area"], "numeric (iter 1)")
        categorical = _pick(["municipality"], "categorical (iter 1)")
        binary      = []
    else:
        numeric     = _pick(["rooms", "area", "rooms_per_m2"], "numeric (iter 2)")
        categorical = _pick(["municipality"], "categorical (iter 2)")
        binary      = _pick(
            ["is_furnished", "is_temporary", "has_balcony", "is_luxurious", "is_zurich_city"],
            "binary (iter 2)",
        )

    if not numeric:
        raise ValueError(
            "No usable numeric features found. "
            "Ensure 'rooms' and/or 'area' columns exist after standardize_columns()."
        )

    print(
        f"[features] Iter {iteration} — "
        f"numeric={numeric}  categorical={categorical}  binary={binary}"
    )
    return numeric, categorical, binary


if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    sample = pd.DataFrame({
        "rooms": [2.5, 4.0, 1.5],
        "area": [60, 110, 30],
        "municipality": ["Zürich", "Winterthur", "Uster"],
        "descriptionraw": [
            "Schöne möblierte Wohnung mit Balkon",
            "Luxuriöses Penthouse, hochwertig",
            "Befristete Untermiete, klein",
        ],
    })
    result = engineer_all_features(sample)
    print(result[["rooms", "area", "rooms_per_m2",
                  "is_furnished", "is_temporary", "has_balcony",
                  "is_luxurious", "is_zurich_city"]])
    numeric, categorical, binary = get_feature_lists(result, iteration=2)
    print(f"\nResolved features: num={numeric}  cat={categorical}  bin={binary}")
