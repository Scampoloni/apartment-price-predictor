"""Training entry point for the apartment price predictor.

Supports two modeling iterations, each selectable via CLI:

  python -m src.train --iteration 1   # Baseline: LinearRegression + RandomForest
  python -m src.train --iteration 2   # Improved:  RandomForest + MLPRegressor

Each iteration:
1. Loads and cleans data/raw/apartments.csv.
2. Detects available columns and renames aliases to canonical names.
3. Applies feature engineering matching the iteration.
4. Resolves the actual feature list dynamically (handles missing optionals).
5. k-fold cross-validation across all candidate models.
6. Final holdout evaluation of the best model.
7. Saves results to results/tables/model_comparison.csv and iterations.csv.
8. Iteration 2 also saves the fitted pipeline artifact to models/.
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FEATURE_NAMES_ARTIFACT,
    MODEL_ARTIFACT,
    MODEL_METADATA_ARTIFACT,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import basic_clean, load_raw_data, standardize_columns
from src.evaluate import (
    cross_validate_model,
    evaluate_on_holdout,
    print_model_comparison,
    save_cv_results,
    save_iterations_csv,
    save_model_comparison_csv,
)
from src.features import (
    engineer_all_features,
    engineer_baseline_features,
    get_feature_lists,
)
from src.preprocessing import build_preprocessor


# ── Explicit hyperparameter dictionaries ──────────────────────────────────────
# Keep these in one place for easy documentation and future tuning.

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
        # NOTE: Replace with GridSearchCV / RandomizedSearchCV results once you
        #       have trained on real data and want to tune further.
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "MLPRegressor": MLPRegressor(
        # NOTE: scale_numeric=True is set for Iteration 2 to ensure all inputs
        #       are standardised before they reach the neural network.
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


# ── Core training function ─────────────────────────────────────────────────────

def train_iteration(iteration: int) -> None:
    """Run a complete training cycle for one iteration.

    Args:
        iteration: 1 (baseline) or 2 (improved/final).
    """
    print(f"\n{'=' * 65}")
    print(f"  ITERATION {iteration}  —  Apartment Price Predictor")
    print(f"{'=' * 65}\n")

    engineer_fn = engineer_baseline_features if iteration == 1 else engineer_all_features
    models = ITER1_MODELS if iteration == 1 else ITER2_MODELS
    scale_numeric = (iteration == 2)  # MLPRegressor needs scaled inputs

    # ── 1. Load, standardise, clean ───────────────────────────────────────────
    df_raw = load_raw_data()
    df = standardize_columns(df_raw)
    df = basic_clean(df)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    df = engineer_fn(df)

    # ── 3. Resolve feature lists based on what columns actually exist ──────────
    numeric_features, categorical_features, binary_features = get_feature_lists(df, iteration)
    all_features = numeric_features + categorical_features + binary_features

    # Drop rows where required numeric features are still missing (e.g. area=NaN)
    df = df.dropna(subset=numeric_features + [TARGET_COLUMN])
    print(f"[train] {len(df):,} rows usable  |  {len(all_features)} features: {all_features}")

    X = df[all_features]
    y = df[TARGET_COLUMN]

    # ── 4. Train / test split (fixed seed for reproducibility) ────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[train] Split: {len(X_train):,} train  /  {len(X_test):,} test")

    # Preprocessor template — cloned fresh for every model pipeline below.
    preprocessor_template = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features or None,
        scale_numeric=scale_numeric,
    )

    # ── 5. Cross-validate all candidate models ─────────────────────────────────
    cv_records: list[dict] = []
    best_cv_rmse = float("inf")
    best_name = ""
    best_estimator = None  # store the sklearn estimator, not the pipeline

    for name, estimator in models.items():
        print(f"\n[train] Cross-validating: {name} …")
        # Clone both steps so models are completely independent
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor_template)),
            ("model", clone(estimator)),
        ])
        cv_record = cross_validate_model(
            pipeline, X_train, y_train, cv=CV_FOLDS, model_name=name
        )
        cv_records.append(cv_record)
        print(
            f"         CV RMSE : {cv_record['cv_rmse_mean']:.0f} "
            f"± {cv_record['cv_rmse_std']:.0f} CHF"
        )
        print(f"         CV MAE  : {cv_record['cv_mae_mean']:.0f} CHF")
        print(f"         CV R²   : {cv_record['cv_r2_mean']:.4f}")

        if cv_record["cv_rmse_mean"] < best_cv_rmse:
            best_cv_rmse = cv_record["cv_rmse_mean"]
            best_name = name
            best_estimator = estimator

    save_cv_results(cv_records, iteration=iteration)
    print_model_comparison()

    # ── 6. Final fit on full training set + holdout evaluation ────────────────
    print(f"[train] Best model: '{best_name}'  (CV RMSE = {best_cv_rmse:.0f} CHF)")
    print("[train] Fitting best model on full training set …")
    best_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor_template)),
        ("model", clone(best_estimator)),
    ])
    best_pipeline.fit(X_train, y_train)

    holdout = evaluate_on_holdout(best_pipeline, X_test, y_test)
    print(f"\n[train] Holdout results:")
    print(f"         RMSE : {holdout['rmse']:.0f} CHF")
    print(f"         MAE  : {holdout['mae']:.0f} CHF")
    print(f"         R²   : {holdout['r2']:.4f}")

    # ── 7. Save model_comparison.csv  (all models, both iterations) ───────────
    for record in cv_records:
        record["iteration"] = iteration
        record["is_best"] = record["model_name"] == best_name
        if record["is_best"]:
            record.update({
                "holdout_rmse": round(holdout["rmse"], 1),
                "holdout_mae":  round(holdout["mae"], 1),
                "holdout_r2":   round(holdout["r2"], 4),
            })
        else:
            record["holdout_rmse"] = None
            record["holdout_mae"]  = None
            record["holdout_r2"]   = None

    save_model_comparison_csv(cv_records)

    # ── 8. Save iterations.csv  (one row per iteration) ───────────────────────
    save_iterations_csv({
        "iteration":    iteration,
        "best_model":   best_name,
        "n_features":   len(all_features),
        "features":     ", ".join(all_features),
        "cv_rmse":      round(best_cv_rmse, 1),
        "holdout_rmse": round(holdout["rmse"], 1),
        "holdout_mae":  round(holdout["mae"], 1),
        "holdout_r2":   round(holdout["r2"], 4),
    })

    # ── 9. Save model artifacts (only for Iteration 2 → used by app.py) ───────
    if iteration == 2:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, MODEL_ARTIFACT)

        with open(FEATURE_NAMES_ARTIFACT, "w") as f:
            json.dump(all_features, f, indent=2)

        metadata = {
            "iteration":    iteration,
            "model_name":   best_name,
            "cv_rmse_mean": round(best_cv_rmse, 1),
            "holdout_rmse": round(holdout["rmse"], 1),
            "holdout_mae":  round(holdout["mae"], 1),
            "holdout_r2":   round(holdout["r2"], 4),
            "n_features":   len(all_features),
            "features":     all_features,
        }
        with open(MODEL_METADATA_ARTIFACT, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[train] Artifacts saved:")
        print(f"         Pipeline : {MODEL_ARTIFACT}")
        print(f"         Features : {FEATURE_NAMES_ARTIFACT}")
        print(f"         Metadata : {MODEL_METADATA_ARTIFACT}")
    else:
        print(
            "\n[train] Iteration 1 complete. "
            "Run --iteration 2 to train the improved model and save inference artifacts."
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train apartment price prediction model."
    )
    parser.add_argument(
        "--iteration",
        type=int,
        choices=[1, 2],
        default=2,
        help="Modeling iteration: 1=baseline, 2=improved (default: 2).",
    )
    args = parser.parse_args()
    train_iteration(args.iteration)

