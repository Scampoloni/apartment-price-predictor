import pytest

from conversational_agent.core import (
    ApartmentQuery,
    municipality_support,
    parse_apartment_query,
    parse_json_response,
    validate_price_free_explanation,
)


def test_valid_structured_extraction():
    query = parse_apartment_query(
        '{"rooms": 3.5, "area_m2": 85, "municipality": "Winterthur", '
        '"description": "mit Balkon"}'
    )
    assert query == ApartmentQuery(3.5, 85.0, "Winterthur", "mit Balkon")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[]",
        '{"rooms": 3.5',
    ],
)
def test_malformed_output_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_json_response(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"rooms": 3.5, "municipality": "Uster"}',
        '{"area_m2": 80, "municipality": "Uster"}',
        '{"rooms": 3.5, "area_m2": 80}',
        '{"rooms": null, "area_m2": 80, "municipality": "Uster"}',
    ],
)
def test_missing_fields_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_apartment_query(raw)


def test_price_fields_are_rejected():
    with pytest.raises(ValueError, match="Price fields are forbidden"):
        parse_apartment_query(
            '{"rooms": 3.5, "area_m2": 80, "municipality": "Uster", '
            '"price": 2500}'
        )


def test_unknown_municipality_gets_explicit_warning():
    known, warning = municipality_support("Atlantis", ["Zürich", "Uster"])
    assert known is False
    assert "not represented" in warning


def test_explanation_cannot_contain_numeric_price():
    assert (
        validate_price_free_explanation(
            '{"answer": "Die Schätzung berücksichtigt Fläche und Lage."}'
        )
        == "Die Schätzung berücksichtigt Fläche und Lage."
    )
    with pytest.raises(ValueError):
        validate_price_free_explanation(
            '{"answer": "Die Schätzung beträgt CHF 2500."}'
        )
