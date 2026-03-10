"""Regression evaluation utilities and result persistence.

Provides:
- Standalone metric functions (RMSE, MAE, R²)
- cross_validate_model()   — k-fold CV with all three metrics
- evaluate_on_holdout()    — final test-set evaluation
- save_cv_results()        — append results to CSV in results/tables/
- print_model_comparison() — formatted terminal table
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CV_RESULTS_FILE, RANDOM_STATE, TABLES_DIR


# ── Metric functions ───────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error — primary metric for this project."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error — easier to interpret in CHF."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination R²."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    model_name: str = "model",
) -> dict[str, Any]:
    """Run stratified k-fold cross-validation and return a result dictionary.

    Uses scikit-learn's cross_val_score with a KFold splitter so that
    the random seed is controlled by RANDOM_STATE.

    Args:
        pipeline:   Fitted or unfitted sklearn Pipeline.
        X:          Feature DataFrame (training split only).
        y:          Target Series (training split only).
        cv:         Number of folds.
        model_name: Label stored in the result dict.

    Returns:
        dict with keys: model_name, cv_rmse_mean, cv_rmse_std,
                        cv_mae_mean, cv_mae_std, cv_r2_mean, cv_r2_std.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    neg_mse = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse)

    neg_mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
    mae_scores = -neg_mae

    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")

    return {
        "model_name": model_name,
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std":  float(rmse_scores.std()),
        "cv_mae_mean":  float(mae_scores.mean()),
        "cv_mae_std":   float(mae_scores.std()),
        "cv_r2_mean":   float(r2_scores.mean()),
        "cv_r2_std":    float(r2_scores.std()),
    }


# ── Holdout evaluation ─────────────────────────────────────────────────────────

def evaluate_on_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a *fitted* pipeline on the held-out test set.

    Args:
        pipeline: Already-fitted sklearn Pipeline.
        X_test:   Test features.
        y_test:   True test labels.

    Returns:
        dict with rmse, mae, r2.
    """
    y_pred = pipeline.predict(X_test)
    return {
        "rmse": rmse(y_test, y_pred),
        "mae":  mae(y_test, y_pred),
        "r2":   r2_score(y_test, y_pred),
    }


# ── Persistence ────────────────────────────────────────────────────────────────

def save_cv_results(
    cv_records: list[dict[str, Any]],
    iteration: int,
) -> None:
    """Append cross-validation records to the cumulative CSV.

    If the file already exists, old rows for the same iteration number
    are replaced so re-running training doesn't accumulate stale rows.

    Args:
        cv_records: List of result dicts from cross_validate_model().
        iteration:  Iteration number (1 or 2) to tag each row.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(cv_records)
    df_new.insert(0, "iteration", iteration)

    if CV_RESULTS_FILE.exists():
        df_existing = pd.read_csv(CV_RESULTS_FILE)
        # Remove any existing rows for this iteration before appending
        df_existing = df_existing[df_existing["iteration"] != iteration]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(CV_RESULTS_FILE, index=False)
    print(f"[evaluate] CV results saved → {CV_RESULTS_FILE}")


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_model_comparison(cv_results_path: Path | None = None) -> None:
    """Print a formatted comparison table of all cross-validation results."""
    if cv_results_path is None:
        cv_results_path = CV_RESULTS_FILE

    if not cv_results_path.exists():
        print("[evaluate] No CV results file found. Run training first.")
        return

    df = pd.read_csv(cv_results_path)
    cols = ["iteration", "model_name", "cv_rmse_mean", "cv_rmse_std",
            "cv_mae_mean", "cv_r2_mean"]
    cols = [c for c in cols if c in df.columns]

    print("\n── Model Comparison ──────────────────────────────────────────────────")
    print(
        df[cols]
        .sort_values(["iteration", "cv_rmse_mean"])
        .to_string(index=False, float_format="{:.2f}".format)
    )
    print("─────────────────────────────────────────────────────────────────────\n")


def save_model_comparison_csv(enriched_cv_records: list[dict[str, Any]]) -> None:
    """Persist all model results (CV + optional holdout) to model_comparison.csv.

    Replaces existing rows for the same iteration so re-running is idempotent.
    The file accumulates results across both iterations, making it easy to
    produce the academic comparison table directly from CSV.

    Args:
        enriched_cv_records: CV result dicts from cross_validate_model(),
            each enriched with keys: iteration, is_best,
            holdout_rmse, holdout_mae, holdout_r2 (None for non-best rows).
    """
    from src.config import MODEL_COMPARISON_FILE  # avoid circular imports at module load
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(enriched_cv_records)
    iteration = df_new["iteration"].iloc[0]

    if MODEL_COMPARISON_FILE.exists():
        df_existing = pd.read_csv(MODEL_COMPARISON_FILE)
        df_existing = df_existing[df_existing["iteration"] != iteration]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Reorder columns for readability
    preferred_order = [
        "iteration", "model_name", "is_best",
        "cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_mae_std",
        "cv_r2_mean", "cv_r2_std",
        "holdout_rmse", "holdout_mae", "holdout_r2",
    ]
    cols = [c for c in preferred_order if c in df_combined.columns]
    df_combined = df_combined[cols + [c for c in df_combined.columns if c not in cols]]

    df_combined.to_csv(MODEL_COMPARISON_FILE, index=False)
    print(f"[evaluate] Model comparison saved → {MODEL_COMPARISON_FILE}")


def save_iterations_csv(iteration_record: dict[str, Any]) -> None:
    """Persist a one-row summary of the best model per iteration.

    Creates results/tables/iterations.csv — one row per iteration,
    overwriting any previous row for the same iteration number.

    Args:
        iteration_record: dict with keys:
            iteration, best_model, n_features, features,
            cv_rmse, holdout_rmse, holdout_mae, holdout_r2.
    """
    from src.config import ITERATIONS_FILE  # avoid circular imports at module load
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([iteration_record])
    iteration = iteration_record["iteration"]

    if ITERATIONS_FILE.exists():
        df_existing = pd.read_csv(ITERATIONS_FILE)
        df_existing = df_existing[df_existing["iteration"] != iteration]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.sort_values("iteration").to_csv(ITERATIONS_FILE, index=False)
    print(f"[evaluate] Iteration summary saved → {ITERATIONS_FILE}")


if __name__ == "__main__":
    print_model_comparison()
