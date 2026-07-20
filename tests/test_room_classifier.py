import pytest
import numpy as np

from room_classifier.core import LABELS, parse_vision_response
from room_classifier.evaluate import calculate_metrics


def test_room_labels_are_stable():
    assert len(LABELS) == 8
    assert len(set(LABELS)) == 8


def test_vision_json_validation():
    assert parse_vision_response(
        '{"label": "kitchen", "confidence": 0.8}'
    ) == {"kitchen": 0.8}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"label": "garage", "confidence": 0.8}',
        '{"label": "kitchen", "confidence": 2}',
        '{"label": "kitchen"}',
    ],
)
def test_bad_vision_output_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_vision_response(raw)


def test_metrics_disclose_missing_test_classes():
    references = np.asarray([3, 4, 5, 6, 7])
    predictions = references.copy()
    summary, per_class, matrix = calculate_metrics(references, predictions)
    assert summary["class_coverage"] == "5/8"
    assert summary["missing_test_classes"] == [
        "bathroom",
        "bedroom",
        "children's room",
    ]
    assert per_class.loc[0, "evaluable"] == False  # noqa: E712
    assert matrix.shape == (8, 8)
