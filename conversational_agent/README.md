---
title: Zurich Apartment AI - Conversational Agent
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Conversational structured-input agent

This application converts German apartment requests into validated fields for
the suite's shared Zurich rent model.

The LLM may extract only:

- `rooms`
- `area_m2`
- `municipality`
- optional explicitly stated listing attributes in `description`

It never predicts, edits, or explains a numeric price. Application code rejects
price fields in extraction output, and the explanation call never receives the
regression result. Numeric rent is produced and displayed only by the
scikit-learn pipeline in `price_estimator/`.

Unknown municipalities are accepted by the encoder but receive an explicit
weak-support warning. The UI also displays the random-holdout and
municipality-grouped uncertainty evidence.

Required secret: `OPENAI_API_KEY`. Optional model override: `OPENAI_MODEL`
(default `gpt-4o-mini`). Never place keys in source files or `.env` files
committed to Git.

Run from the repository root:

```bash
pip install -r conversational_agent/requirements.txt
python -m conversational_agent.app
```
