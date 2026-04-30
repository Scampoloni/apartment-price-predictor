import json
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd
from openai import OpenAI

MODEL_PATH = "random_forest_regression.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df_bfs = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs["tax_income"] = (
    df_bfs["tax_income"].astype(str).str.replace("'", "", regex=False).astype(float)
)

town_to_row = {str(row["bfs_name"]).lower(): row for _, row in df_bfs.iterrows()}
valid_towns = list(df_bfs["bfs_name"].sort_values().unique())


def match_town(user_town: str):
    if not user_town or not user_town.strip():
        return None
    key = user_town.strip().lower()
    if key in town_to_row:
        return str(town_to_row[key]["bfs_name"])
    for canonical in valid_towns:
        if key in canonical.lower():
            return canonical
    return None


def call_llm_json(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(raw: str, required_keys: tuple) -> dict:
    cleaned = (raw or "").strip().strip("```").lstrip("json").strip()
    if not cleaned:
        raise ValueError("Das LLM hat eine leere Antwort zurückgegeben.")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Das LLM hat kein gültiges JSON zurückgegeben: {cleaned[:300]}") from exc
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f"JSON fehlen Pflichtfelder: {', '.join(missing)}")
    return parsed


def extract_preferences(user_text: str) -> dict:
    system_prompt = (
        "Du bist ein Assistent, der Wohnungswünsche aus deutschem Text extrahiert. "
        "Antworte ausschliesslich mit einem JSON-Objekt ohne Markdown. "
        "Pflichtfelder: rooms (float), area_m2 (float), town (string). "
        "Falls ein Wert nicht genannt wird, setze null."
    )
    user_prompt = f"Extrahiere die Wohnungsparameter aus folgendem Text:\n\n{user_text}"
    raw = call_llm_json(system_prompt, user_prompt)
    return parse_json_response(raw, required_keys=("rooms", "area_m2", "town"))


def predict_apartment_price(rooms: float, area_m2: float, town: str) -> float:
    row = town_to_row.get(town.lower())
    if row is None:
        raise ValueError(f"Gemeinde '{town}' wurde nicht in den BFS-Daten gefunden.")
    features = np.array([[
        rooms,
        area_m2,
        float(row["pop"]),
        float(row["pop_dens"]),
        float(row["frg_pct"]),
        float(row["emp"]),
        float(row["tax_income"]),
    ]])
    return float(model.predict(features)[0])


def generate_explanation(preferences: dict, prediction: float) -> str:
    system_prompt = (
        "Du bist ein freundlicher Immobilienassistent. "
        "Erkläre die Mietpreisschätzung kurz auf Deutsch. "
        "Antworte ausschliesslich mit einem JSON-Objekt: {\"answer\": \"...\"}. "
        "Erwähne eine Unsicherheit des Modells. Berechne keinen eigenen Preis."
    )
    user_prompt = (
        f"Wohnungsparameter: {json.dumps(preferences, ensure_ascii=False)}\n"
        f"Modellschätzung: {prediction:.0f} CHF/Monat\n"
        "Erkläre das Ergebnis in 2-3 Sätzen."
    )
    raw = call_llm_json(system_prompt, user_prompt)
    parsed = parse_json_response(raw, required_keys=("answer",))
    return parsed["answer"]


def run_pipeline(user_text: str):
    if not user_text or not user_text.strip():
        return {}, None, "Bitte einen Wohnungswunsch eingeben."
    try:
        prefs = extract_preferences(user_text)
    except Exception as e:
        return {}, None, f"Fehler bei der Extraktion: {e}"

    rooms = prefs.get("rooms")
    area_m2 = prefs.get("area_m2")
    town_raw = prefs.get("town")

    if not rooms or not area_m2 or not town_raw:
        return prefs, None, "Bitte Zimmeranzahl, Fläche und Ort angeben."

    matched = match_town(str(town_raw))
    if not matched:
        return prefs, None, f"Ort '{town_raw}' wurde nicht gefunden. Bitte einen Schweizer Ortsnamen angeben."

    prefs["town_matched"] = matched

    try:
        prediction = predict_apartment_price(rooms, area_m2, matched)
    except Exception as e:
        return prefs, None, f"Fehler bei der Vorhersage: {e}"

    try:
        explanation = generate_explanation(
            {"rooms": rooms, "area_m2": area_m2, "town": matched},
            prediction,
        )
    except Exception as e:
        return prefs, prediction, f"Vorhersage: {prediction:.0f} CHF (Erklärung fehlgeschlagen: {e})"

    return prefs, prediction, explanation


with gr.Blocks(title="Apartment Predictor") as demo:
    gr.Markdown(
        """
        # Apartment Predictor — Wohnungsmiete schätzen
        Beschreibe deinen Wohnungswunsch auf Deutsch.
        Das Modell extrahiert Zimmer, Fläche und Ort und schätzt die monatliche Miete.

        **Beispiel:** *Ich suche eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur.*
        """
    )

    user_text = gr.Textbox(
        label="Wohnungswunsch (Deutsch)",
        lines=4,
        placeholder="Beschreibe Zimmer, Fläche und Ort auf Deutsch...",
    )
    submit = gr.Button("Miete schätzen", variant="primary")

    extracted = gr.JSON(label="Extrahierte Parameter")
    price = gr.Number(label="Geschätzte Monatsmiete (CHF)")
    response = gr.Textbox(label="Erklärung", lines=6)

    submit.click(
        fn=run_pipeline,
        inputs=[user_text],
        outputs=[extracted, price, response],
    )

demo.launch()
