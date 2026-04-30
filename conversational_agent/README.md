---
title: Apartment Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Apartment Predictor — Wohnungsmiete schätzen

Diese App kombiniert ein vortrainiertes Random-Forest-Regressionsmodell mit GPT-4o-mini, um aus einem deutschen Freitext-Wohnungswunsch eine monatliche Mietpreisschätzung zu erstellen.

## Workflow

1. Nutzer beschreibt Wohnungswunsch auf Deutsch
2. LLM extrahiert `rooms`, `area_m2`, `town` als JSON
3. Regressionsmodell schätzt Monatsmiete (CHF) anhand von 7 Features inkl. BFS-Gemeindedaten
4. LLM erklärt das Ergebnis auf Deutsch mit Unsicherheitshinweis

## Beispiel-Eingabe

> *Ich suche eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur.*

## Benötigte Secrets

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, Standard: `gpt-4o-mini`)
