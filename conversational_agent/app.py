"""Conversational structured-input frontend for the shared rent model."""

from __future__ import annotations

import json
import os
from typing import Callable

import gradio as gr
from anthropic import Anthropic

from conversational_agent.core import (
    ApartmentQuery,
    municipality_support,
    parse_apartment_query,
    validate_price_free_explanation,
)
from price_estimator.src.predict import _load_metadata, predict_price

EXTRACTION_SYSTEM_PROMPT = """
Extract apartment search preferences from German free text.
Return exactly one JSON object with these keys:
- rooms: number or null
- area_m2: number or null
- municipality: string or null
- description: only listing attributes explicitly stated by the user, or ""
Never infer missing values. Never output, estimate, copy, or add a price field.
""".strip()

EXPLANATION_SYSTEM_PROMPT = """
Write a short German explanation of which apartment attributes a statistical
rent model can use and which important limitations remain. Return JSON with
exactly one string field named "answer". Do not mention any number, price,
currency, range, or calculation. The numeric estimate is rendered separately
by deterministic application code.
""".strip()


def call_llm_json(system_prompt: str, user_prompt: str) -> str:
    """Call the configured Claude model and return its text response."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    response = Anthropic(api_key=api_key).messages.create(
        model=model_name,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=250,
    )
    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if not text:
        raise ValueError("Claude returned an empty response.")
    return text


def extract_preferences(user_text: str) -> ApartmentQuery:
    raw = call_llm_json(
        EXTRACTION_SYSTEM_PROMPT,
        f"Extract only explicitly stated fields from:\n\n{user_text}",
    )
    return parse_apartment_query(raw)


def generate_price_free_explanation(query: ApartmentQuery) -> str:
    """Generate qualitative prose without exposing the model's price output."""
    raw = call_llm_json(
        EXPLANATION_SYSTEM_PROMPT,
        "Validated apartment attributes:\n"
        + json.dumps(query.to_dict(), ensure_ascii=False),
    )
    return validate_price_free_explanation(raw)


def _deterministic_uncertainty_note(
    prediction_result: dict,
    support_warning: str,
) -> str:
    metadata = _load_metadata()
    random_rmse = metadata.get("holdout_rmse")
    geographic = metadata.get("geographic_evaluation") or {}
    parts = [
        "Research estimate only: the model was trained on a small sample from "
        "the canton of Zurich and is not suitable for production valuation."
    ]
    if random_rmse is not None:
        parts.append(f"Random-holdout RMSE: about CHF {random_rmse:,.0f}.")
    if geographic.get("rmse") is not None:
        parts.append(
            "Municipality-holdout RMSE: about "
            f"CHF {geographic['rmse']:,.0f}; this harder test is the better "
            "warning for new locations."
        )
    warning = support_warning or prediction_result.get("municipality_warning", "")
    if warning:
        parts.append(warning)
    return " ".join(parts)


def run_pipeline(
    user_text: str,
    *,
    extractor: Callable[[str], ApartmentQuery] = extract_preferences,
    explainer: Callable[[ApartmentQuery], str] = generate_price_free_explanation,
) -> tuple[dict, float | None, str]:
    """Extract, validate, predict, then explain without giving price to the LLM."""
    if not user_text or not user_text.strip():
        return {}, None, "Bitte einen Wohnungswunsch eingeben."
    try:
        query = extractor(user_text)
    except Exception as exc:  # noqa: BLE001
        return {}, None, f"Extraktion fehlgeschlagen: {exc}"

    metadata = _load_metadata()
    _, support_warning = municipality_support(
        query.municipality,
        metadata.get("known_municipalities", []),
    )
    try:
        prediction_result = predict_price(
            rooms=query.rooms,
            area=query.area_m2,
            municipality=query.municipality,
            description=query.description,
        )
    except Exception as exc:  # noqa: BLE001
        return query.to_dict(), None, f"Vorhersage fehlgeschlagen: {exc}"

    try:
        explanation = explainer(query)
    except Exception:
        explanation = (
            "Die Schätzung nutzt die validierten Wohnungsangaben. Zustand, "
            "Mikrolage und aktuelle Marktänderungen bleiben unberücksichtigt."
        )

    note = _deterministic_uncertainty_note(prediction_result, support_warning)
    return (
        query.to_dict(),
        float(prediction_result["predicted_price_chf"]),
        f"{explanation}\n\n{note}",
    )


with gr.Blocks(title="Zurich Apartment AI Suite — Conversational Agent") as demo:
    gr.Markdown(
        """
        # Zurich Apartment AI Suite — Conversational Agent

        Beschreibe Zimmer, Fläche, Zürcher Gemeinde und optionale
        Inseratmerkmale. Das LLM strukturiert nur die Eingabe; ausschliesslich
        das Regressionsmodell erzeugt die numerische Mietschätzung.
        """
    )
    user_text = gr.Textbox(
        label="Wohnungswunsch (Deutsch)",
        lines=4,
        placeholder="Zum Beispiel: dreieinhalb Zimmer, achtzig Quadratmeter in Uster",
    )
    submit = gr.Button("Miete schätzen", variant="primary")
    extracted = gr.JSON(label="Validierte Modellparameter")
    price = gr.Number(label="Modellschätzung Monatsmiete (CHF)")
    response = gr.Textbox(label="Erklärung und Unsicherheit", lines=8)
    submit.click(
        fn=run_pipeline,
        inputs=[user_text],
        outputs=[extracted, price, response],
    )


if __name__ == "__main__":
    demo.launch()
