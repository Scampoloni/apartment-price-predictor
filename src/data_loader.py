"""Data loading, column standardisation, and basic cleaning.

Separates raw I/O from all ML logic so the rest of the pipeline
can assume a clean, validated DataFrame with canonical column names.

Canonical names used throughout the codebase:
    price         — monthly rent in CHF (target)
    rooms         — number of rooms
    area          — living area in m²
    municipality  — location / municipality name  (optional)
    descriptionraw — free-text listing description (optional)
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    CANDIDATE_AREA_COLS,
    CANDIDATE_DESCRIPTION_COLS,
    CANDIDATE_LOCATION_COLS,
    CANDIDATE_ROOMS_COLS,
    CANDIDATE_TARGET_COLS,
    DESCRIPTION_COLUMN,
    LOCATION_COLUMN,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load raw apartment data from CSV or Excel with auto-detection.

    For CSV files, tries common encodings (utf-8, latin-1, cp1252) and
    separators (comma, semicolon, tab) automatically so you don't need to
    pre-configure anything.

    Args:
        filepath: Path to the raw data file. Defaults to RAW_DATA_FILE.

    Returns:
        Raw, unmodified DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at filepath.
        ValueError: If the file format is unsupported.
    """
    if filepath is None:
        filepath = RAW_DATA_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {filepath}\n"
            "→ Place your CSV at  data/raw/apartments.csv\n"
            "→ Or update RAW_DATA_FILE in src/config.py."
        )

    suffix = filepath.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
        print(f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} cols from '{filepath.name}'")
        return df

    if suffix != ".csv":
        raise ValueError(f"Unsupported format '{suffix}'. Use .csv or .xlsx.")

    # Try encodings × separators; return on first valid parse (>1 column).
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    print(
                        f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} cols "
                        f"from '{filepath.name}'  (encoding={encoding}, sep={repr(sep)})"
                    )
                    return df
            except Exception:
                continue

    # Last-resort: let pandas auto-detect
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} cols (auto-detect mode)")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect aliased column names and rename them to canonical names.

    Searches CANDIDATE_*_COLS lists defined in config.py (first match wins)
    and renames discovered columns to the canonical names that every other
    module expects.  Reports each decision so you can verify the mapping.

    Required (training will fail if absent after detection):
        price, rooms, area

    Optional (features default to 0 / empty string if absent):
        municipality, descriptionraw
    """
    df = df.copy()
    # Normalise column names: strip whitespace
    df.columns = df.columns.str.strip()

    alias_map: dict[str, list[str]] = {
        TARGET_COLUMN:      CANDIDATE_TARGET_COLS,
        "rooms":            CANDIDATE_ROOMS_COLS,
        "area":             CANDIDATE_AREA_COLS,
        LOCATION_COLUMN:    CANDIDATE_LOCATION_COLS,
        DESCRIPTION_COLUMN: CANDIDATE_DESCRIPTION_COLS,
    }
    required = {TARGET_COLUMN, "rooms", "area"}

    required_found:   list[str] = []
    required_missing: list[str] = []
    optional_found:   list[str] = []
    optional_missing: list[str] = []

    print("[data_loader] ── Column Mapping Report ────────────────────────────────")
    for canonical, candidates in alias_map.items():
        is_req = canonical in required
        bucket_ok  = required_found   if is_req else optional_found
        bucket_bad = required_missing if is_req else optional_missing
        if canonical in df.columns:
            print(f"  ✓  {canonical:<22} found directly")
            bucket_ok.append(canonical)
            continue
        matched = False
        for alias in candidates:
            if alias in df.columns:
                df = df.rename(columns={alias: canonical})
                print(f"  ✓  {canonical:<22} ← renamed from '{alias}'")
                bucket_ok.append(canonical)
                matched = True
                break
        if not matched:
            tag = "✗  REQUIRED" if is_req else "○  optional"
            print(f"  {tag:<13} {canonical:<22} (tried: {candidates})")
            bucket_bad.append(canonical)

    print()
    req_ok  = ", ".join(required_found)  or "—"
    req_bad = ", ".join(required_missing) or "—"
    opt_ok  = ", ".join(optional_found)  or "—"
    opt_bad = ", ".join(optional_missing) or "—"
    print(f"  Required  found   : {req_ok}")
    print(f"  Required  missing : {req_bad}")
    print(f"  Optional  found   : {opt_ok}")
    print(f"  Optional  missing : {opt_bad}")
    print("[data_loader] ─────────────────────────────────────────────────────────")

    if required_missing:
        col_list = ", ".join(f"'{c}'" for c in required_missing)
        hints = "\n".join(
            f"    {c}: tried {alias_map[c]}" for c in required_missing
        )
        raise ValueError(
            f"\n\nRequired column(s) not found: {col_list}\n\n"
            f"Candidates tried:\n{hints}\n\n"
            f"→ Add your column name to the matching CANDIDATE_*_COLS list in src/config.py.\n"
            f"→ Columns present in your CSV: {sorted(df.columns.tolist())}"
        )

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply safe, minimal cleaning steps to the standardised DataFrame.

    Steps applied:
    1. Strip leading/trailing whitespace from all string columns.
    2. Drop exact duplicate rows.
    3. Parse target column as numeric (handles Swiss apostrophe formatting).
    4. Drop rows where target is missing or non-positive.
    5. Parse rooms/area as numeric.
    6. Apply domain-specific bounds for Swiss apartments.

    NOTE: The price bounds (200–25 000 CHF) and area/room caps below are
    sensible defaults for canton-of-Zurich rental data. Adjust if your
    dataset has different characteristics.
    """
    n_raw = len(df)

    # Strip string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])

    # Handle Swiss price formatting, e.g. "2'500" or "2.500"
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN]
        .astype(str)
        .str.replace("'", "", regex=False)
        .str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    # Numeric coercion for key columns (in case they were parsed as strings)
    for col in ("rooms", "area"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Domain bounds — adjust if needed for your dataset
    if "area" in df.columns:
        df = df[df["area"].isna() | df["area"].between(10, 600)]
    if "rooms" in df.columns:
        df = df[df["rooms"].isna() | df["rooms"].between(0.5, 25)]
    df = df[df[TARGET_COLUMN].between(200, 25_000)]   # CHF/month

    n_clean = len(df)
    print(
        f"[data_loader] Cleaned: {n_raw:,} → {n_clean:,} rows "
        f"({n_raw - n_clean:,} removed by filters)"
    )
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df_raw = load_raw_data()
    df = standardize_columns(df_raw)
    df = basic_clean(df)
    print(f"\nTarget statistics (CHF/month):\n{df[TARGET_COLUMN].describe().round(0)}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
