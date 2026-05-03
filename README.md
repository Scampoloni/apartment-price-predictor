# Apartment AI Applications — Canton of Zurich

Three end-to-end AI applications built around Swiss apartment data, combining machine learning, computer vision, and natural language processing. All apps are deployed as interactive web applications on Hugging Face Spaces.

---

## Applications

### 1. Apartment Price Predictor
> Regression · scikit-learn · Feature Engineering · Gradio

Estimates the monthly gross rental price (CHF) for apartments in the Canton of Zurich based on size, location, and listing features extracted from text.

**Live demo:** [huggingface.co/spaces/Scampolonii/apartment-price-predictor](https://huggingface.co/spaces/Scampolonii/apartment-price-predictor)

| | |
|---|---|
| Task | Regression |
| Dataset | Canton of Zurich rental listings — 819 listings, 112 municipalities |
| Best model | RandomForestRegressor |
| CV RMSE | 798 CHF (±164) |
| Holdout RMSE | 840 CHF |
| R² | 0.58 |

**What makes it interesting:**
Two training iterations with full cross-validation. Iteration 2 introduced engineered features extracted from raw listing text (furnished, balcony, luxury, temporary rental flags) alongside a room density metric — improving generalization over the baseline.

| Iteration | Model | CV RMSE | CV R² |
|---|---|---|---|
| 1 — Baseline | LinearRegression | 880 CHF | 0.48 |
| 1 — Baseline | RandomForest v1 | 800 CHF | 0.58 |
| 2 — Feature Engineering | MLPRegressor | 886 CHF | 0.47 |
| 2 — Feature Engineering | **RandomForest v2** | **798 CHF** | **0.58** |

**Features:** `rooms` · `area` · `rooms_per_m2` · `municipality` (one-hot, 112 classes) · `is_furnished` · `has_balcony` · `is_luxurious` · `is_temporary` · `is_zurich_city`

**Stack:** Python · scikit-learn · pandas · Gradio

---

### 2. Conversational Apartment Agent
> NLP · LLM · Prompt Engineering · Gradio

A conversational agent that turns free-text German apartment queries into structured prediction inputs, runs them through the trained regression model, and returns a natural language explanation of the result.

**Live demo:** [huggingface.co/spaces/Scampolonii/apartement-conversational-agent](https://huggingface.co/spaces/Scampolonii/apartement-conversational-agent)

**How it works:**
1. User describes their apartment wish in German free text
2. GPT-4o-mini extracts `rooms`, `area_m2`, and `town` as structured JSON
3. BFS municipality data is joined to build the full 7-feature model input
4. RandomForest predicts the monthly rent in CHF
5. A second LLM call generates a German explanation with an uncertainty note

**Example:**
> *"Ich suche eine 3.5-Zimmer-Wohnung mit 85 m² in Winterthur."*
> → `{"rooms": 3.5, "area_m2": 85, "town": "Winterthur"}` → CHF 2'082/month + explanation

**Prompt design highlights:**
- Strict JSON-only output enforced via system instruction
- Temperature=0 for deterministic extraction
- Explanation prompt explicitly forbids the LLM from calculating its own price
- Validation layer catches malformed JSON before it reaches the model

**Stack:** Python · OpenAI API · scikit-learn · pandas · Gradio

---

### 3. Apartment Room Classifier
> Computer Vision · Transfer Learning · ViT · CLIP · GPT-4o · Gradio

Classifies apartment room images into 8 categories and compares three fundamentally different model approaches: fine-tuned ViT, zero-shot CLIP, and GPT-4o vision.

**Live demo:** [huggingface.co/spaces/Scampolonii/apartment-room-classifier](https://huggingface.co/spaces/Scampolonii/apartment-room-classifier)  
**Trained model:** [huggingface.co/Scampolonii/vit-apartment-rooms](https://huggingface.co/Scampolonii/vit-apartment-rooms)

| | |
|---|---|
| Task | Multi-class image classification (8 classes) |
| Dataset | MIT Indoor Scenes — ~15,571 images, filtered to 8 apartment-relevant classes |
| Base model | `google/vit-base-patch16-224` |
| Training strategy | Transfer learning — only classifier head trained |
| Trainable parameters | 4,614 out of 85,803,270 |
| Test accuracy | **90.16%** |

**Classes:** `bathroom` · `bedroom` · `children's room` · `corridor` · `dining room` · `kitchen` · `living room` · `nursery`

**Model comparison (8 example images):**

| Model | Type | Correct | Notes |
|---|---|---|---|
| Fine-tuned ViT | Transfer learning | 4/8 (50%) | Strong within training distribution; weaker on atypical scenes |
| CLIP zero-shot | Open-source zero-shot | 6/8 (75%) | Solid generalization with no task-specific training |
| GPT-4o | Closed-source LLM vision | 8/8 (100%) | Best results; understands context and scene composition |

**Stack:** Python · PyTorch · HuggingFace Transformers · CLIP · OpenAI API · Gradio

---

## Repository Structure

```
apartment-price-predictor/
│
├── app.py                      # Price predictor — Gradio web interface
├── requirements.txt
│
├── src/                        # Price predictor source package
│   ├── config.py               # Paths, column names, hyperparameters
│   ├── data_loader.py          # Data loading & cleaning
│   ├── features.py             # Feature engineering
│   ├── preprocessing.py        # sklearn ColumnTransformer
│   ├── train.py                # Training entry point (iterations 1 & 2)
│   ├── evaluate.py             # Metrics & CV evaluation
│   └── predict.py              # Inference module
│
├── models/
│   ├── pipeline.joblib         # Trained sklearn pipeline
│   └── metadata.json           # Model metrics snapshot
│
├── conversational_agent/       # Conversational agent app
│   ├── app.py                  # Gradio app with LLM pipeline
│   ├── random_forest_regression.pkl
│   ├── bfs_municipality_and_tax_data.csv
│   └── documentation.md
│
└── cv_app/                     # Room classifier app
    ├── app.py                  # Gradio app with 3-model comparison
    ├── README.md               # Detailed classifier documentation
    └── examples/               # 8 example room images
```

---

## Local Setup

### Price Predictor

```bash
git clone https://github.com/Scampoloni/apartment-price-predictor
cd apartment-price-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place dataset at: data/raw/original_apartment_data_analytics_hs24_with_lat_lon.csv
python -m src.train --iteration 1
python -m src.train --iteration 2
python app.py  # → http://localhost:7860
```

### Conversational Agent

```bash
cd conversational_agent
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python app.py  # → http://localhost:7860
```

### Room Classifier

```bash
cd cv_app
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python app.py  # → http://localhost:7860
```
