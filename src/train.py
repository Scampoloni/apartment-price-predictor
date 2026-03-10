"""Training entry point for the apartment price predictor.

Supports two modeling iterations, each configurable via CLI:

  python -m src.train --iteration 1   # Baseline: LinearRegression + RandomForest
  python -m src.train --iteration 2   # Improved: RandomForest + MLPRegressor

Each iteration:
1. Loads and cleans the raw data.
2. Applies feature engineering matching the iteration's config.
3. Performs k-fold cross-validation on all candidate models.
4. Evaluates the best model on a held-out test split.
5. Saves an iteration summary to results/tables/.
6. If iteration == 2, saves the final fitted pipeline artifact to models/.
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config import (
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
    CV_FOLDS,
    FEATURE_NAMES_ARTIFACT,
    IMPROVED_BINARY_FEATURES,
    IMPROVED_CATEGORICAL_FEATURES,
    IMPROVED_NUMERIC_FEATURES,
    ITERATION_SUMMARY_FILE,
    MODEL_ARTIFACT,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import basic_clean, load_raw_data
from src.evaluate import (
    cross_validate_model,
    evaluate_on_holdout,
    print_model_comparison,
    save_cv_results,
)
from src.features import engineer_all_features, engineer_baseline_features
from src.preprocessing import build_preprocessor


# ── Hyperparameter definitions ─────────────────────────────────────────────────
# Keep all hyperparameters explicit here for easy documentation and comparison.

ITER1_MODELS: dict = {
    "LinearRegression": LinearRegression(),
    "RandomForest_v1": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

ITER2_MODELS: dict = {
    "RandomForest_v2": RandomForestRegressor(
        # TODO: Replace with best params from GridSearchCV / RandomizedSearchCV
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "MLPRegressor": MLPRegressor(
        # NOTE: MLPRegressor is sensitive to feature scaling.
        #       Scaling is enabled via scale_numeric=True in ITER2_CONFIG.
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    ),
}


# ── Per-iteration configuration ────────────────────────────────────────────────

ITER1_CONFIG: dict = {
    "numeric":      BASELINE_NUMERIC_FEATURES,
    "categorical":  BASELINE_CATEGORICAL_FEATURES,
    "binary":       [],
    "scale":        False,   # tree models don't need scaling
    "engineer_fn":  engineer_baseline_features,
}

ITER2_CONFIG: dict = {
    "numeric":      IMPROVED_NUMERIC_FEATURES,
    "categorical":  IMPROVED_CATEGORICAL_FEATURES,
    "binary":       IMPROVED_BINARY_FEATURES,
    "scale":        True,    # MLPRegressor benefits from scaled input
    "engineer_fn":  engineer_all_features,
}


# ── Core training logic ────────────────────────────────────────────────────────

def train_iteration(iteration: int) -> None:
    """Run a full training cycle for the given iteration number.

    Args:
        iteration: 1 (baseline) or 2 (improved).
    """
    print(f"\n{'=' * 65}")
    print(f"  ITERATION {iteration}  —  Apartment Price Predictor")
    print(f"{'=' * 65}\n")

    config = ITER1_CONFIG if iteration == 1 else ITER2_CONFIG
    models = ITER1_MODELS if iteration == 1 else ITER2_MODELS

    # ── 1. Load and clean data ─────────────────────────────────────────────────
    df_raw = load_raw_data()
    df = basic_clean(df_raw)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    df = config["engineer_fn"](df)

    all_features: list[str] = (
        config["numeric"] + config["categorical"] + config["binary"]
    )

    # Drop rows still missing required columns after engineering
    df = df.dropna(subset=all_features + [TARGET_COLUMN])
    print(f"[train] Dataset after engineering: {len(df):,} rows, {len(all_features)} features")
    print(f"[train] Feature list: {all_features}")

    X = df[all_features]
    y = df[TARGET_COLUMN]

    # ── 3. Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[train] Split: {len(X_train):,} train / {len(X_test):,} test")

    # ── 4. Build shared preprocessor ──────────────────────────────────────────
    preprocessor = build_preprocessor(
        numeric_features=config["numeric"],
        categorical_features=config["categorical"],
        binary_features=config["binary"] or None,
        scale_numeric=config["scale"],
    )

    # ── 5. Cross-validate all candidate models ─────────────────────────────────
    cv_records = []
    best_cv_rmse = float("inf")
    best_name: str = ""
    best_pipeline: Pipeline | None = None

    for name, estimator in models.items():
        print(f"\n[train] Cross-validating: {name} …")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        cv_record = cross_validate_model(
            pipeline, X_train, y_train, cv=CV_FOLDS, model_name=name
        )
        cv_records.append(cv_record)

        print(
            f"         CV RMSE : {cv_record['cv_rmse_mean']:.0f} "
            f"(± {cv_record['cv_rmse_std']:.0f})"
        )
        print(f"         CV MAE  : {cv_record['cv_mae_mean']:.0f}")
        print(f"         CV R²   : {cv_record['cv_r2_mean']:.4f}")

        if cv_record["cv_rmse_mean"] < best_cv_rmse:
            best_cv_rmse = cv_record["cv_rmse_mean"]
            best_name = name
            best_pipeline = pipeline

    save_cv_results(cv_records, iteration=iteration)
    print_model_comparison()

    # ── 6. Final fit + holdout evaluation of best model ───────────────────────
    print(f"[train] Best model: '{best_name}'  (CV RMSE = {best_cv_rmse:.0f})")
    print("[train] Fitting best model on full training set …")
    assert best_pipeline is not None
    best_pipeline.fit(X_train, y_train)

    holdout = evaluate_on_holdout(best_pipeline, X_test, y_test)
    print(f"\n[train] Holdout evaluation:")
    print(f"         RMSE : {holdout['rmse']:.0f} CHF")
    print(f"         MAE  : {holdout['mae']:.0f} CHF")
    print(f"         R²   : {holdout['r2']:.4f}")

    # ── 7. Persist iteration summary ──────────────────────────────────────────
    summary = {
        "iteration":    iteration,
        "best_model":   best_name,
        "features":     all_features,
        "cv_rmse_mean": best_cv_rmse,
        "holdout":      holdout,
    }
    ITERATION_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summaries: list = []
    if ITERATION_SUMMARY_FILE.exists():
        with open(ITERATION_SUMMARY_FILE) as f:
            summaries = json.load(f)
    # Replace any existing entry for this iteration
    summaries = [s for s in summaries if s.get("iteration") != iteration]
    summaries.append(summary)
    with open(ITERATION_SUMMARY_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[train] Iteration summary saved → {ITERATION_SUMMARY_FILE}")

    # ── 8. Save final model artifact (only after iteration 2) ─────────────────
    if iteration == 2:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, MODEL_ARTIFACT)
        with open(FEATURE_NAMES_ARTIFACT, "w") as f:
            json.dump(all_features, f, indent=2)
        print(f"[train] Pipeline  saved → {MODEL_ARTIFACT}")
        print(f"[train] Feature names  → {FEATURE_NAMES_ARTIFACT}")
    else:
        print("\n[train] Tip: Run --iteration 2 to train the improved model and save artifacts.")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train apartment price prediction model (iteration 1 or 2)."
    )
    parser.add_argument(
        "--iteration",
        type=int,
        choices=[1, 2],
        default=2,
        help="Modeling iteration to run. 1=baseline, 2=improved (default: 2).",
    )
    args = parser.parse_args()
    train_iteration(args.iteration)
