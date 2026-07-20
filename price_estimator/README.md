# Apartment-rent regression

Leakage-safe rent-estimation application for the canton of Zurich.

## Data contract

Place the authorized local dataset at:

```text
data/raw/original_apartment_data_analytics_hs24_with_lat_lon.csv
```

The file is ignored by Git. The audited local copy has 819 raw rows and 93
municipalities; 817 rows remain after validity filters.

Required columns are detected through aliases for target rent, rooms, and area.
Municipality and listing description are optional but expected for the full
feature set.

## Evaluation

```bash
python -m price_estimator.src.train --iteration 2
```

This preserves the fixed random 80/20 holdout and adds five-fold municipality
`GroupKFold`. In every geographic fold, all rows for each test municipality are
absent from training. Fitted imputers, scalers, and one-hot categories live
inside the scikit-learn pipeline.

The current aggregate evidence is in `results/price_estimator/`:

- random holdout: RMSE CHF 840, MAE CHF 504, R² 0.563;
- municipality GroupKFold: RMSE CHF 1,104, MAE CHF 657, R² 0.235.

The weaker grouped result is the primary warning for unseen-location use.

## Inference

```bash
python -m price_estimator.app
```

`predict_price()` validates numeric ranges, applies the same row-wise feature
engineering, loads the fitted pipeline lazily, handles unseen municipalities
through `OneHotEncoder(handle_unknown="ignore")`, and returns an explicit
support warning. Estimates are indicative only.
