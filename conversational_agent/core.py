"""Validation and orchestration primitives for conversational rent input."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Mapping

FORBIDDEN_PRICE_KEYS = {
    "price",
    "rent",
    "monthly_rent",
    "predicted_price",
    "predicted_price_chf",
}


@dataclass(frozen=True)
class ApartmentQuery:
    """Validated fields the language model is allowed to extract."""

    rooms: float
    area_m2: float
    municipality: str
    description: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ApartmentQuery":
        if not isinstance(value, Mapping):
            raise ValueError("The extracted JSON must be an object.")

        forbidden = FORBIDDEN_PRICE_KEYS & {str(key).lower() for key in value}
        if forbidden:
            raise ValueError(
                "Price fields are forbidden in LLM output: "
                + ", ".join(sorted(forbidden))
            )

        required = ("rooms", "area_m2", "municipality")
        missing = [key for key in required if key not in value or value[key] is None]
        if missing:
            raise ValueError("Missing required fields: " + ", ".join(missing))

        try:
            rooms = float(value["rooms"])
            area_m2 = float(value["area_m2"])
        except (TypeError, ValueError) as exc:
            raise ValueError("rooms and area_m2 must be numeric.") from exc

        municipality = str(value["municipality"]).strip()
        description = str(value.get("description") or "").strip()
        if not 0.5 <= rooms <= 15:
            raise ValueError("rooms must be between 0.5 and 15.")
        if not 10 <= area_m2 <= 500:
            raise ValueError("area_m2 must be between 10 and 500.")
        if not municipality:
            raise ValueError("municipality must not be empty.")

        return cls(
            rooms=rooms,
            area_m2=area_m2,
            municipality=municipality,
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a single JSON object, accepting an optional Markdown fence."""
    cleaned = (raw or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned:
        raise ValueError("The LLM returned an empty response.")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("The extracted JSON must be an object.")
    return parsed


def parse_apartment_query(raw: str) -> ApartmentQuery:
    """Parse and validate the language model's structured extraction."""
    return ApartmentQuery.from_mapping(parse_json_response(raw))


def validate_price_free_explanation(raw: str) -> str:
    """Reject explanations containing numbers or currency-like price text."""
    parsed = parse_json_response(raw)
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Explanation JSON must contain a non-empty 'answer'.")
    answer = answer.strip()
    if re.search(r"\d|CHF|Fr\.?|\$|€", answer, flags=re.IGNORECASE):
        raise ValueError(
            "The LLM explanation must not contain numbers or currency values."
        )
    return answer


def municipality_support(
    municipality: str,
    known_municipalities: list[str],
) -> tuple[bool, str]:
    """Return explicit support information for known and unseen locations."""
    known = set(known_municipalities)
    if municipality in known:
        return True, ""
    return (
        False,
        (
            f"'{municipality}' was not represented in the training sample. "
            "The encoder accepts unseen municipalities, but geographic "
            "generalisation is less reliable."
        ),
    )
