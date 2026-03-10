"""Gradio web application — Apartment Price Predictor, Canton of Zurich.

Loads the saved model artifact once at startup.
Never retrains; only calls src.predict.predict_price().

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    Push the entire repository to a Space with SDK = gradio.
    The Space will run   python app.py   automatically.
"""

import gradio as gr

from src.predict import predict_price


# ── Example inputs (shown below the interface) ────────────────────────────────
EXAMPLES = [
    [3.5,  80,  "Zürich",      "Helle Wohnung mit Balkon, ruhige Lage"],
    [2.5,  60,  "Zürich",      "Möbliertes Studio, zentral, befristet"],
    [4.5, 110,  "Winterthur",  "Geräumige Familienwohnung, gute Anbindung"],
    [5.5, 140,  "Zürich",      "Luxuriöses Penthouse, exklusive Ausstattung, Terrasse"],
    [1.5,  35,  "Uster",       "Kleines Zimmer, Untermiete, befristet"],
]


# ── Prediction handler ─────────────────────────────────────────────────────────

def predict_fn(
    rooms: float,
    area: float,
    municipality: str,
    description: str,
) -> str:
    """Gradio handler: build inputs, call predict_price(), format output."""
    try:
        result = predict_price(
            rooms=rooms,
            area=area,
            municipality=municipality.strip() or None,
            description=description.strip() or None,
        )
        price = result["predicted_price_chf"]
        note = result["model_note"]
        return (
            f"## Estimated Monthly Rent\n\n"
            f"# CHF {price:,.0f}\n\n"
            f"---\n_{note}_"
        )
    except FileNotFoundError as exc:
        return (
            "⚠️ **Model not found.**\n\n"
            "Please train the model first:\n"
            "```\npython -m src.train --iteration 2\n```\n\n"
            f"Details: `{exc}`"
        )
    except Exception as exc:  # noqa: BLE001
        return f"❌ **Prediction error:** {exc}"


# ── Gradio Blocks UI ───────────────────────────────────────────────────────────

with gr.Blocks(title="Apartment Price Predictor — Canton of Zurich") as demo:

    gr.Markdown("""
    # 🏠 Apartment Price Predictor — Canton of Zurich
    Estimate the **monthly rental price** for an apartment in the canton of Zurich.
    Enter the key apartment details below and click **Predict Rent**.

    > Model trained on rental listings using a scikit-learn pipeline
    > (Random Forest + feature engineering). Results are indicative only.
    """)

    with gr.Row():

        # ── Left column: inputs ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Apartment Details")

            rooms_input = gr.Number(
                label="Number of Rooms",
                value=3.5,
                minimum=0.5,
                maximum=20.0,
                step=0.5,
                info="Swiss convention: includes living room (e.g. 3.5-Zimmer)",
            )
            area_input = gr.Number(
                label="Living Area (m²)",
                value=80,
                minimum=10,
                maximum=500,
                step=5,
            )
            municipality_input = gr.Textbox(
                label="Municipality (optional)",
                placeholder="e.g. Zürich, Winterthur, Uster, Küsnacht",
                info="Helps the model apply location-based adjustments.",
            )
            description_input = gr.Textbox(
                label="Description keywords (optional)",
                placeholder="e.g. möbliert, Balkon, Terrasse, Luxus, befristet …",
                lines=3,
                info="Paste or type relevant keywords from the listing. "
                     "The model extracts furnished / balcony / luxury flags.",
            )
            predict_btn = gr.Button("Predict Rent", variant="primary", size="lg")

        # ── Right column: output ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Prediction")
            output = gr.Markdown(
                value="_Enter apartment details and click **Predict Rent**._"
            )

    # ── Wire up button ─────────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict_fn,
        inputs=[rooms_input, area_input, municipality_input, description_input],
        outputs=output,
    )

    # ── Example inputs ─────────────────────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLES,
        inputs=[rooms_input, area_input, municipality_input, description_input],
        label="Example Apartments",
    )

    gr.Markdown("""
    ---
    **Disclaimer:** Predictions are statistical estimates based on historical
    rental data for the canton of Zurich. They do not constitute a binding
    offer and may deviate significantly from actual market prices.
    """)


if __name__ == "__main__":
    demo.launch()
