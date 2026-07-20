"""Inference module — loads saved model artifacts and predicts rent.

The app imports *only* this module.  It never retrains or touches raw data.
Artifacts are loaded once and cached for the lifetime of the process.

Usage:
    from price_estimator.src.predict import predict_price
    result = predict_price(rooms=3.5, area=80, municipality="Zürich")
    print(result["predicted_price_chf"])
"""

import json
from functools import lru_cache
from typing import Optional

import joblib
import pandas as pd

from price_estimator.src.config import (
    DESCRIPTION_COLUMN,
    FEATURE_NAMES_ARTIFACT,
    LOCATION_COLUMN,
    MODEL_ARTIFACT,
    MODEL_METADATA_ARTIFACT,
)
from price_estimator.src.features import engineer_all_features


# ── Artifact loading (cached) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_pipeline():
    """Load and cache the trained sklearn Pipeline from disk.

    Raises:
        FileNotFoundError: If models/pipeline.joblib does not exist.
    """
    if not MODEL_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_ARTIFACT}\n"
            "Run training first:\n"
            "  python -m price_estimator.src.train --iteration 2"
        )
    return joblib.load(MODEL_ARTIFACT)


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    """Load and cache the list of feature names saved during training."""
    if not FEATURE_NAMES_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Feature names artifact not found: {FEATURE_NAMES_ARTIFACT}\n"
            "Run training first:\n"
            "  python -m price_estimator.src.train --iteration 2"
        )
    with open(FEATURE_NAMES_ARTIFACT) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_metadata() -> dict:
    """Load and cache the model metadata (metrics, feature list) saved during training."""
    if not MODEL_METADATA_ARTIFACT.exists():
        return {}  # metadata is optional — app still works without it
    with open(MODEL_METADATA_ARTIFACT) as f:
        return json.load(f)


def assess_municipality(municipality: Optional[str]) -> dict[str, object]:
    """Describe whether a municipality was represented during training."""
    value = (municipality or "").strip()
    known = set(_load_metadata().get("known_municipalities", []))
    if not value:
        return {
            "is_known": False,
            "warning": (
                "No municipality was supplied. Location-specific information "
                "cannot be used."
            ),
        }
    if known and value not in known:
        return {
            "is_known": False,
            "warning": (
                f"'{value}' was not represented in training. The one-hot "
                "encoder handles it safely, but the estimate has weaker "
                "geographic support."
            ),
        }
    return {"is_known": True, "warning": ""}


# ── Public prediction function ────────────────────────────────────────────────

def predict_price(
    rooms: float,
    area: float,
    municipality: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Predict the monthly rental price for an apartment.

    Applies the same feature engineering used during training so that
    the preprocessing pipeline always receives the expected columns.

    Args:
        rooms:        Number of rooms (e.g. 3.5).
        area:         Living area in m² (e.g. 80).
        municipality: Municipality name string, optional.
                      Used to derive the is_zurich_city flag.
        description:  Free-text listing description, optional.
                      Used to derive furnished / balcony / luxury flags.

    Returns:
        dict:
            "predicted_price_chf" (float) — estimated monthly rent in CHF.
            "model_note" (str)            — brief annotation about the estimate.
    """
    if not 0.5 <= float(rooms) <= 15:
        raise ValueError("rooms must be between 0.5 and 15.")
    if not 10 <= float(area) <= 500:
        raise ValueError("area must be between 10 and 500 m².")

    pipeline = _load_pipeline()
    feature_names = _load_feature_names()

    # Build a single-row DataFrame with all raw inputs.
    # Any column not provided falls back to a safe default so that
    # feature engineering can still run without error.
    row = {
        "rooms":           rooms,
        "area":            area,
        LOCATION_COLUMN:   municipality or "",
        DESCRIPTION_COLUMN: description or "",
    }
    df = pd.DataFrame([row])

    # Apply the same feature engineering pipeline used during training.
    df = engineer_all_features(df)

    # Ensure every expected feature column exists; fill with 0 if absent.
    # This makes inference robust to new / missing columns at runtime.
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_names]

    predicted = float(pipeline.predict(X)[0])
    predicted = max(0.0, predicted)  # rent must be non-negative

    # Build a human-readable model note from saved metadata
    meta = _load_metadata()
    municipality_support = assess_municipality(municipality)
    if meta:
        geographic = meta.get("geographic_evaluation") or {}
        model_note = (
            f"Model: {meta.get('model_name', 'unknown')}  |"
            f"  random-holdout RMSE ≈ CHF {meta.get('holdout_rmse', '?'):,}"
        )
        if geographic:
            model_note += (
                "  |  municipality-holdout RMSE ≈ "
                f"CHF {geographic.get('rmse', '?'):,}"
            )
        model_note += (
            ". Indicative estimate only; the small canton-only dataset does "
            "not support production valuation."
        )
    else:
        model_note = "Estimate from trained scikit-learn pipeline. For indicative purposes only."

    return {
        "predicted_price_chf": round(predicted, 2),
        "model_note": model_note,
        "municipality_known": municipality_support["is_known"],
        "municipality_warning": municipality_support["warning"],
    }


if __name__ == "__main__":
    # Quick smoke test — requires a trained model artifact
    test_cases = [
        {"rooms": 3.5, "area": 80,  "municipality": "Zürich",     "description": "Balkon, hell"},
        {"rooms": 2.5, "area": 60,  "municipality": "Winterthur", "description": "möbliert"},
        {"rooms": 5.5, "area": 140, "municipality": "Zürich",     "description": "Luxus Penthouse"},
    ]
    print("\n── Smoke-test predictions ───────────────────────────────────────")
    for tc in test_cases:
        result = predict_price(**tc)
        print(
            f"  {tc['rooms']}r / {tc['area']}m² / {tc['municipality']:<12} "
            f"→  CHF {result['predicted_price_chf']:,.0f}"
        )
    print("─────────────────────────────────────────────────────────────────\n")
