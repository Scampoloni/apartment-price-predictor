"""Central configuration for the apartment price predictor.

All paths, column names, feature lists, and hyperparameter defaults
live here so every other module imports from one place.
"""

from pathlib import Path

# ── Project root (two levels up from this file) ────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset file — place your CSV at data/raw/apartments.csv
# If your filename differs, update this path.
RAW_DATA_FILE = RAW_DATA_DIR / "apartments.csv"

# ── Model / artifact paths ─────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACT = MODELS_DIR / "pipeline.joblib"
FEATURE_NAMES_ARTIFACT = MODELS_DIR / "feature_names.json"
MODEL_METADATA_ARTIFACT = MODELS_DIR / "metadata.json"  # stores metrics + feature list

# ── Results paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CV_RESULTS_FILE = TABLES_DIR / "cv_results.csv"
ITERATION_SUMMARY_FILE = TABLES_DIR / "iteration_summary.json"
MODEL_COMPARISON_FILE = TABLES_DIR / "model_comparison.csv"   # all models, all iterations
ITERATIONS_FILE = TABLES_DIR / "iterations.csv"               # one row per iteration

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20       # 80/20 train-test split
CV_FOLDS = 5           # k-fold cross-validation

# ── Target variable ────────────────────────────────────────────────────────────
# TODO: Set to the exact column name for monthly rent in your dataset.
#       Common names: "price", "rent", "miete", "bruttomiete"
TARGET_COLUMN = "price"

# ── Baseline feature set (Iteration 1) ────────────────────────────────────────
# TODO: Adjust to the actual column names present in your dataset.
#       Start with the most obviously predictive numeric columns.
BASELINE_NUMERIC_FEATURES: list[str] = [
    "rooms",  # number of rooms  (e.g. 3.5)
    "area",   # living area in m²
]

BASELINE_CATEGORICAL_FEATURES: list[str] = [
    # TODO: Uncomment and adjust once you know your categorical columns.
    # "municipality",   # municipality name string
    # "canton",         # should be "ZH" for all rows in this dataset
]

# ── Improved feature set (Iteration 2) ────────────────────────────────────────
# Extends the baseline with engineered features from src/features.py
IMPROVED_NUMERIC_FEATURES: list[str] = BASELINE_NUMERIC_FEATURES + [
    "rooms_per_m2",  # engineered: rooms / area — captures room density
]

IMPROVED_CATEGORICAL_FEATURES: list[str] = BASELINE_CATEGORICAL_FEATURES + [
    # TODO: Add higher-cardinality location columns if available, e.g.
    # "district_category",   # binned district (low/mid/high price zone)
    # "municipality",        # move here from baseline if cardinality is manageable
]

IMPROVED_BINARY_FEATURES: list[str] = [
    "is_furnished",  # extracted from description text
    "is_temporary",  # befristet / Zwischenmiete
    "has_balcony",   # balkon / terrasse
    "is_luxurious",  # luxury / exklusiv / penthouse keywords
    "is_zurich_city",  # 1 if municipality == Zürich city
]

# ── Text column ────────────────────────────────────────────────────────────────
# TODO: Set to the raw description column name used for flag extraction.
#       Common names: "descriptionraw", "description", "text", "beschreibung"
DESCRIPTION_COLUMN = "descriptionraw"

# ── Address / location column ──────────────────────────────────────────────────
# TODO: Set to the column that holds the municipality or location name.
#       Common names: "municipality", "gemeinde", "address", "location"
LOCATION_COLUMN = "municipality"

# ── Column auto-detection alias lists ─────────────────────────────────────────
# data_loader.standardize_columns() searches these lists in order and renames
# the first matching alias to the canonical name. Add your column name here if
# it does not match any of the existing candidates.
CANDIDATE_TARGET_COLS: list[str] = [
    "price", "rent", "miete", "preis", "bruttomiete", "nettomiete",
]
CANDIDATE_ROOMS_COLS: list[str] = [
    "rooms", "zimmer", "anzahl_zimmer", "numberOfRooms", "number_of_rooms",
    "zimmeranzahl",
]
CANDIDATE_AREA_COLS: list[str] = [
    "area", "flaeche", "wohnflaeche", "livingArea", "sqm", "m2",
    "groesse", "wohnungsgroesse", "quadratmeter",
]
CANDIDATE_LOCATION_COLS: list[str] = [
    "municipality", "gemeinde", "city", "ort", "location",
    "address", "standort", "lokalitaet",
]
CANDIDATE_DESCRIPTION_COLS: list[str] = [
    "descriptionraw", "description", "text", "beschreibung",
    "freitext", "inserattext", "beschr",
]
