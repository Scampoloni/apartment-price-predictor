"""Pure helpers shared by the room-classifier app and smoke tests."""

from __future__ import annotations

import json
import re
from typing import Any

LABELS = [
    "bathroom",
    "bedroom",
    "children's room",
    "corridor",
    "dining room",
    "kitchen",
    "living room",
    "nursery",
]

DATASET_LABELS = [
    "bathroom",
    "bedroom",
    "children_room",
    "corridor",
    "dining_room",
    "kitchen",
    "livingroom",
    "nursery",
]

DATASET_TO_DISPLAY = dict(zip(DATASET_LABELS, LABELS))
CLIP_PROMPTS = [f"a photograph of an apartment {label}" for label in LABELS]


def parse_vision_response(raw: str) -> dict[str, float]:
    """Validate the optional GPT-4o qualitative classification response."""
    cleaned = (raw or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        result: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Vision model returned invalid JSON.") from exc

    label = result.get("label")
    if label not in LABELS:
        raise ValueError(f"Vision model returned unsupported label: {label!r}")
    try:
        confidence = float(result["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Vision response needs a numeric confidence.") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("Vision confidence must be between zero and one.")
    return {label: round(confidence, 4)}
