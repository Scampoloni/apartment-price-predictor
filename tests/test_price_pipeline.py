import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from price_estimator.src.analysis import (
    evaluate_geographic_group_kfold,
    geographic_holdout_indices,
)
from price_estimator.src.features import engineer_all_features
from price_estimator.src.preprocessing import build_preprocessor


def _fixture() -> pd.DataFrame:
    municipalities = ["Zürich", "Winterthur", "Uster", "Dietikon"]
    rows = []
    for municipality_index, municipality in enumerate(municipalities):
        for index in range(6):
            rooms = 1.5 + (index % 4)
            area = 35 + index * 12 + municipality_index * 3
            rows.append(
                {
                    "rooms": rooms,
                    "area": area,
                    "municipality": municipality,
                    "descriptionraw": "Balkon" if index % 2 else "",
                    "price": 900 + area * 14 + municipality_index * 50,
                }
            )
    return pd.DataFrame(rows)


def test_small_fixture_pipeline_handles_unseen_municipality():
    data = engineer_all_features(_fixture())
    features = [
        "rooms",
        "area",
        "rooms_per_m2",
        "municipality",
        "is_furnished",
        "is_temporary",
        "has_balcony",
        "is_luxurious",
        "is_zurich_city",
    ]
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features=["rooms", "area", "rooms_per_m2"],
                    categorical_features=["municipality"],
                    binary_features=[
                        "is_furnished",
                        "is_temporary",
                        "has_balcony",
                        "is_luxurious",
                        "is_zurich_city",
                    ],
                ),
            ),
            (
                "model",
                RandomForestRegressor(n_estimators=8, random_state=42),
            ),
        ]
    )
    pipeline.fit(data[features], data["price"])
    unseen = engineer_all_features(
        pd.DataFrame(
            [
                {
                    "rooms": 3.5,
                    "area": 82,
                    "municipality": "Unseen municipality",
                    "descriptionraw": "",
                }
            ]
        )
    )
    prediction = pipeline.predict(unseen[features])
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_feature_engineering_does_not_use_target():
    first = _fixture()
    second = first.copy()
    second["price"] = second["price"] * 10
    feature_columns = [
        "rooms_per_m2",
        "is_furnished",
        "is_temporary",
        "has_balcony",
        "is_luxurious",
        "is_zurich_city",
    ]
    pd.testing.assert_frame_equal(
        engineer_all_features(first)[feature_columns],
        engineer_all_features(second)[feature_columns],
    )


def test_geographic_split_has_no_municipality_overlap():
    data = _fixture()
    train_index, test_index = geographic_holdout_indices(
        data["municipality"],
        test_size=0.25,
        random_state=42,
    )
    train_groups = set(data.iloc[train_index]["municipality"])
    test_groups = set(data.iloc[test_index]["municipality"])
    assert train_groups.isdisjoint(test_groups)


def test_group_kfold_keeps_every_fold_municipality_disjoint():
    data = engineer_all_features(_fixture())
    features = [
        "rooms",
        "area",
        "rooms_per_m2",
        "municipality",
        "is_furnished",
        "is_temporary",
        "has_balcony",
        "is_luxurious",
        "is_zurich_city",
    ]
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    ["rooms", "area", "rooms_per_m2"],
                    ["municipality"],
                    [
                        "is_furnished",
                        "is_temporary",
                        "has_balcony",
                        "is_luxurious",
                        "is_zurich_city",
                    ],
                ),
            ),
            ("model", RandomForestRegressor(n_estimators=4, random_state=42)),
        ]
    )
    result, residuals = evaluate_geographic_group_kfold(
        pipeline,
        data[features],
        data["price"],
        n_splits=4,
    )
    assert result["all_folds_municipality_disjoint"] is True
    assert all(not fold["municipality_overlap"] for fold in result["folds"])
    assert len(residuals) == len(data)
