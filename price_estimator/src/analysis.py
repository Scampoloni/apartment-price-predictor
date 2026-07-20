"""Leakage-safe holdout evaluation and residual analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from price_estimator.src.evaluate import mae, r2_score, rmse

MIN_SUBGROUP_SIZE = 20
FREQUENT_MUNICIPALITY_THRESHOLD = 10


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return the three regression metrics used throughout the suite."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def geographic_holdout_indices(
    groups: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by municipality, keeping every held-out group out of training."""
    safe_groups = groups.fillna("<missing>").astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(safe_groups, groups=safe_groups))

    train_groups = set(safe_groups.iloc[train_idx])
    test_groups = set(safe_groups.iloc[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Municipality leakage detected: {sorted(overlap)}")
    return train_idx, test_idx


def evaluate_geographic_holdout(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    group_column: str = "municipality",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Fit a fresh pipeline on municipalities disjoint from the test set."""
    if group_column not in X.columns:
        raise ValueError(
            f"Geographic evaluation requires the '{group_column}' feature."
        )

    train_idx, test_idx = geographic_holdout_indices(
        X[group_column],
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    held_out = sorted(X_test[group_column].fillna("<missing>").astype(str).unique())

    result: dict[str, Any] = {
        **regression_metrics(y_test, predictions),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_municipalities": int(X_train[group_column].nunique(dropna=False)),
        "n_test_municipalities": int(X_test[group_column].nunique(dropna=False)),
        "held_out_municipalities": held_out,
        "municipality_overlap": [],
        "splitter": "GroupShuffleSplit",
        "test_group_fraction": test_size,
        "random_state": random_state,
    }

    residuals = X_test.copy()
    residuals["actual_price_chf"] = y_test.to_numpy()
    residuals["predicted_price_chf"] = predictions
    residuals["residual_chf"] = y_test.to_numpy() - predictions
    residuals["absolute_error_chf"] = np.abs(residuals["residual_chf"])
    return fitted, result, residuals


def evaluate_geographic_group_kfold(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    group_column: str = "municipality",
    n_splits: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create out-of-fold predictions with municipality-disjoint folds.

    Every observation is evaluated once, and its municipality is absent from
    that fold's training data. Global metrics are calculated over the combined
    out-of-fold predictions.
    """
    if group_column not in X.columns:
        raise ValueError(
            f"Geographic evaluation requires the '{group_column}' feature."
        )
    groups = X[group_column].fillna("<missing>").astype(str)
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.full(len(X), np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(
                f"Municipality leakage in fold {fold}: {sorted(overlap)}"
            )

        fitted = clone(pipeline)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_predictions = fitted.predict(X.iloc[test_idx])
        predictions[test_idx] = fold_predictions
        fold_records.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_municipalities": int(len(train_groups)),
                "n_test_municipalities": int(len(test_groups)),
                "held_out_municipalities": sorted(test_groups),
                "municipality_overlap": [],
                **regression_metrics(y.iloc[test_idx], fold_predictions),
            }
        )

    if np.isnan(predictions).any():
        raise AssertionError("Some rows did not receive a group-fold prediction.")

    result: dict[str, Any] = {
        **regression_metrics(y, predictions),
        "n_observations": int(len(X)),
        "n_municipalities": int(groups.nunique()),
        "splitter": "GroupKFold",
        "n_splits": n_splits,
        "all_folds_municipality_disjoint": True,
        "folds": fold_records,
    }
    residuals = X.copy()
    residuals["actual_price_chf"] = y.to_numpy()
    residuals["predicted_price_chf"] = predictions
    residuals["residual_chf"] = y.to_numpy() - predictions
    residuals["absolute_error_chf"] = np.abs(residuals["residual_chf"])
    return result, residuals


def _metric_row(
    dimension: str,
    group: str,
    mask: pd.Series,
    residuals: pd.DataFrame,
    *,
    min_samples: int,
) -> dict[str, Any]:
    subset = residuals.loc[mask]
    n = len(subset)
    warning = (
        f"Low sample size (n={n}); interpret cautiously."
        if n < min_samples
        else ""
    )
    if n == 0:
        return {
            "dimension": dimension,
            "group": group,
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "warning": warning or "No observations.",
        }

    metrics = regression_metrics(
        subset["actual_price_chf"],
        subset["predicted_price_chf"].to_numpy(),
    )
    return {
        "dimension": dimension,
        "group": group,
        "n": n,
        **metrics,
        "warning": warning,
    }


def build_error_analysis(
    residuals: pd.DataFrame,
    training_municipality_counts: pd.Series,
    *,
    min_samples: int = MIN_SUBGROUP_SIZE,
    frequent_threshold: int = FREQUENT_MUNICIPALITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate requested subgroup metrics and largest residual examples."""
    required = {
        "actual_price_chf",
        "predicted_price_chf",
        "municipality",
        "is_zurich_city",
        "is_furnished",
    }
    missing = required - set(residuals.columns)
    if missing:
        raise ValueError(f"Residual frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []

    price = residuals["actual_price_chf"]
    price_bands = {
        "< CHF 2,000": price < 2000,
        "CHF 2,000–2,999": (price >= 2000) & (price < 3000),
        "≥ CHF 3,000": price >= 3000,
    }
    for label, mask in price_bands.items():
        rows.append(
            _metric_row("rent_price_band", label, mask, residuals, min_samples=min_samples)
        )

    for label, value in (("Zurich city", 1), ("Rest of canton", 0)):
        rows.append(
            _metric_row(
                "geography",
                label,
                residuals["is_zurich_city"] == value,
                residuals,
                min_samples=min_samples,
            )
        )

    test_frequency = (
        residuals["municipality"]
        .map(training_municipality_counts)
        .fillna(0)
        .astype(int)
    )
    rows.append(
        _metric_row(
            "municipality_frequency",
            f"Frequent in training (≥ {frequent_threshold})",
            test_frequency >= frequent_threshold,
            residuals,
            min_samples=min_samples,
        )
    )
    rows.append(
        _metric_row(
            "municipality_frequency",
            f"Sparse/unseen in training (< {frequent_threshold})",
            test_frequency < frequent_threshold,
            residuals,
            min_samples=min_samples,
        )
    )

    for label, value in (("Furnished flag", 1), ("No furnished flag", 0)):
        rows.append(
            _metric_row(
                "furnishing",
                label,
                residuals["is_furnished"] == value,
                residuals,
                min_samples=min_samples,
            )
        )

    subgroup_metrics = pd.DataFrame(rows)
    largest = (
        residuals.sort_values("absolute_error_chf", ascending=False)
        .head(10)
        .loc[
            :,
            [
                "rooms",
                "area",
                "municipality",
                "is_furnished",
                "actual_price_chf",
                "predicted_price_chf",
                "residual_chf",
                "absolute_error_chf",
            ],
        ]
        .reset_index(drop=True)
    )
    return subgroup_metrics, largest


def save_evaluation_artifacts(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    subgroup_metrics: pd.DataFrame,
    largest_residuals: pd.DataFrame,
) -> None:
    """Persist aggregate evidence without publishing the underlying listings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    subgroup_metrics.to_csv(output_dir / "error_analysis.csv", index=False)
    largest_residuals.round(1).to_csv(
        output_dir / "largest_residuals.csv",
        index=False,
    )
