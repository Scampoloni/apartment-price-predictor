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

# Primary dataset file for this project.
# Place the file at this exact path; no renaming needed.
RAW_DATA_FILE = RAW_DATA_DIR / "original_apartment_data_analytics_hs24_with_lat_lon.csv"

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
# Monthly gross rental price in CHF.  Aliases detected by data_loader are
# listed in CANDIDATE_TARGET_COLS below.
TARGET_COLUMN = "price"

# ── Static reference feature lists (documentation only) ────────────────────────
# These lists describe the intended feature sets for each iteration.
# They are NOT imported by train.py; the runtime list is resolved dynamically
# by src.features.get_feature_lists() which checks what columns actually exist.

# Iteration 1 — baseline (raw columns only, no engineering)
BASELINE_NUMERIC_FEATURES: list[str] = [
    "rooms",  # number of rooms (e.g. 3.5)
    "area",   # living area in m²
]
BASELINE_CATEGORICAL_FEATURES: list[str] = [
    "municipality",  # municipality name string (included when column is found)
]

# Iteration 2 — improved (adds engineered features from src/features.py)
IMPROVED_NUMERIC_FEATURES: list[str] = BASELINE_NUMERIC_FEATURES + [
    "rooms_per_m2",  # engineered: rooms / area — captures room density
]

IMPROVED_CATEGORICAL_FEATURES: list[str] = [
    "municipality",  # municipality name string (included when column is found)
]

IMPROVED_BINARY_FEATURES: list[str] = [
    "is_furnished",  # extracted from description text
    "is_temporary",  # befristet / Zwischenmiete
    "has_balcony",   # balkon / terrasse
    "is_luxurious",  # luxury / exklusiv / penthouse keywords
    "is_zurich_city",  # 1 if municipality == Zürich city
]

# ── Text column ────────────────────────────────────────────────────────────────
# Free-text listing description — used only for flag extraction, never as a
# direct model input.  Aliases auto-detected by CANDIDATE_DESCRIPTION_COLS.
DESCRIPTION_COLUMN = "descriptionraw"

# ── Address / location column ──────────────────────────────────────────────────
# Municipality / location name column.  Aliases auto-detected by CANDIDATE_LOCATION_COLS.
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
    "municipality", "gemeinde",
    "bfs_name",   # official BFS municipality name (e.g. "Rüti (ZH)")
    "town",       # display town name   (e.g. "Rüti ZH")
    "city", "ort", "location", "standort", "lokalitaet",
    "address",    # last: full street address — only used as fallback
]
CANDIDATE_DESCRIPTION_COLS: list[str] = [
    "descriptionraw",
    "description_raw",  # variant used in HS24 dataset
    "description", "text", "beschreibung",
    "freitext", "inserattext", "beschr",
]

# ── Passthrough columns (available in the HS24 dataset, not used by the model) ─
# lat, lon  — WGS-84 coordinates (potential future feature: distance to Zurich HB)
# x, y      — Swiss LV95 coordinates
# pop_dens  — population density (potential future feature)
# tax_income — median taxable income per municipality (potential future feature)
# These columns pass through data_loader unchanged and are available for EDA.
