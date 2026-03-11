"""Gradio web application — Apartment Price Predictor, Canton of Zurich.

Loads the saved model artifact once at startup.
Never retrains; only calls src.predict.predict_price().

Run locally:
    python app.py          →  http://localhost:7860

Deploy to Hugging Face Spaces:
    Push the whole repo (including models/) to a Gradio Space.
    The Space automatically runs   python app.py.
"""

import gradio as gr

from src.predict import predict_price, _load_metadata

# ── Known municipalities in canton of Zurich ──────────────────────────────────
# Sorted roughly by population for a useful default ordering.
# Add or remove entries to match your dataset's municipality values.
ZH_MUNICIPALITIES = [
    "",                      # blank = unknown / unspecified
    "Zürich",
    "Winterthur",
    "Uster",
    "Dietikon",
    "Dübendorf",
    "Kloten",
    "Wetzikon",
    "Bülach",
    "Regensdorf",
    "Illnau-Effretikon",
    "Schlieren",
    "Adliswil",
    "Horgen",
    "Thalwil",
    "Küsnacht",
    "Meilen",
    "Herrliberg",
    "Erlenbach (ZH)",
    "Zollikon",
    "Kilchberg",
    "Rüschlikon",
    "Männedorf",
    "Rapperswil-Jona",
    "Gossau (ZH)",
    "Andelfingen",
    "Andere / Other",
]

# ── Example inputs displayed below the interface ──────────────────────────────
EXAMPLES = [
    [3.5,  80,  "Zürich",      "Helle Wohnung, Balkon, zentrale Lage"],
    [2.5,  58,  "Zürich",      "Möbliertes Studio, zentral, befristet"],
    [4.5, 110,  "Winterthur",  "Geräumige Familienwohnung, gute Anbindung"],
    [5.5, 145,  "Zürich",      "Luxuriöses Penthouse, exklusive Ausstattung, grosse Terrasse"],
    [1.5,  32,  "Uster",       "Kleines Zimmer, Untermiete, befristet"],
    [3.5,  90,  "Thalwil",     "Ruhige Lage am See, Balkon, sonnig"],
]


# ── Prediction handler ─────────────────────────────────────────────────────────

def predict_fn(
    rooms: float,
    area: float,
    municipality: str,
    description: str,
) -> tuple[str, str]:
    """Gradio handler: call predict_price() and return (price_md, note_md)."""
    try:
        result = predict_price(
            rooms=float(rooms),
            area=float(area),
            municipality=municipality.strip() if municipality else None,
            description=description.strip() if description else None,
        )
        price = result["predicted_price_chf"]
        note = result["model_note"]

        price_md = (
            f"<div style='text-align:center; padding:20px;'>"
            f"<p style='font-size:1rem; color:#666; margin:0;'>Estimated monthly rent</p>"
            f"<p style='font-size:2.8rem; font-weight:700; color:#1f6feb; margin:4px 0;'>"
            f"CHF {price:,.0f}</p>"
            f"<p style='font-size:0.85rem; color:#999;'>per month (incl. ancillary costs)</p>"
            f"</div>"
        )
        note_md = f"ℹ️ {note}"
        return price_md, note_md

    except FileNotFoundError:
        error_md = (
            "<div style='text-align:center; padding:20px; color:#d73a49;'>"
            "<b>⚠️ Model not found</b><br>"
            "Train the model first:<br>"
            "<code>python -m src.train --iteration 2</code>"
            "</div>"
        )
        return error_md, ""
    except Exception as exc:  # noqa: BLE001
        return f"<div style='color:red;'>❌ Error: {exc}</div>", ""


# ── Build a model-info snippet for the sidebar ────────────────────────────────

def _model_info_md() -> str:
    try:
        meta = _load_metadata()
    except Exception:
        meta = {}
    if not meta:
        return "_Train the model to see performance metrics here._"
    model_name = meta.get("selected_model") or meta.get("model_name", "–")
    iteration  = meta.get("selected_iteration") or meta.get("iteration", "–")
    return (
        f"**Model:** {model_name}  \n"
        f"**Iteration:** {iteration}  \n"
        f"**CV RMSE:** CHF {meta.get('cv_rmse_mean', '–'):,}  \n"
        f"**Holdout RMSE:** CHF {meta.get('holdout_rmse', '–'):,}  \n"
        f"**Holdout MAE:** CHF {meta.get('holdout_mae', '–'):,}  \n"
        f"**R²:** {meta.get('holdout_r2', '–')}  \n"
        f"**Features ({meta.get('n_features', '–')}):** "
        f"{', '.join(meta.get('features', [])) or '–'}"
    )


# ── Gradio Blocks UI ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="Apartment Price Predictor — Canton of Zurich",
) as demo:

    gr.Markdown("""
    # 🏠 Apartment Price Predictor — Canton of Zurich
    Predict the **estimated monthly rent** for an apartment in the canton of Zurich.
    Fill in the details below, then click **Predict Rent**.

    > Powered by a scikit-learn regression pipeline trained on real rental listings.
    > Predictions are statistical estimates — not binding offers.
    """)

    with gr.Row():

        # ── Left column: inputs ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📋 Apartment Details")

            rooms_input = gr.Slider(
                label="Number of Rooms",
                minimum=0.5, maximum=12.0, step=0.5, value=3.5,
                info="Swiss convention: living room counts as a room (e.g. 3.5-Zimmer)",
            )
            area_input = gr.Slider(
                label="Living Area (m²)",
                minimum=15, maximum=400, step=5, value=80,
            )
            municipality_input = gr.Dropdown(
                label="Municipality",
                choices=ZH_MUNICIPALITIES,
                value="",
                allow_custom_value=True,   # user can type any name
                info="Select from the list or type a municipality name.",
            )
            description_input = gr.Textbox(
                label="Listing Keywords (optional)",
                placeholder="e.g. möbliert, Balkon, Terrasse, Luxus, befristet, Untermiete …",
                lines=3,
                info=(
                    "Paste keywords from the listing description. "
                    "The model extracts furnished / balcony / luxury / temporary flags."
                ),
            )
            predict_btn = gr.Button("🔍 Predict Rent", variant="primary", size="lg")

        # ── Right column: output ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 💰 Prediction")
            price_output = gr.HTML(
                value="<div style='text-align:center; padding:40px; color:#888;'>"
                      "Enter apartment details and click <b>Predict Rent</b>.</div>"
            )
            note_output = gr.Markdown()

            with gr.Accordion("📊 Model Performance", open=False):
                gr.Markdown(_model_info_md())

    # ── Wire up ────────────────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict_fn,
        inputs=[rooms_input, area_input, municipality_input, description_input],
        outputs=[price_output, note_output],
    )

    # ── Examples ───────────────────────────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLES,
        inputs=[rooms_input, area_input, municipality_input, description_input],
        label="Example Apartments",
        examples_per_page=6,
    )

    gr.Markdown("""
    ---
    **Disclaimer:** Predictions are statistical estimates based on historical rental data
    for the canton of Zurich. They do not constitute a binding offer and may deviate
    significantly from actual market conditions.
    """)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
