# ML Apartment Applications — Canton of Zurich

Two end-to-end machine learning applications built around apartment data:
a **rental price predictor** (regression) and a **room type classifier** (computer vision).
Both are deployed as interactive web apps on Hugging Face Spaces.

---

## Applications

### 1. Apartment Price Predictor
> Regression · scikit-learn · Gradio · Hugging Face Spaces

Estimates the monthly gross rental price (CHF) for apartments in the Canton of Zurich
based on size, location, and listing features.

**Live demo:** [huggingface.co/spaces/Scampolonii/apartment-price-predictor](https://huggingface.co/spaces/Scampolonii/apartment-price-predictor)

| Property | Value |
|---|---|
| Task | Regression |
| Dataset | Canton of Zurich rental listings (819 rows, 112 municipalities) |
| Best model | RandomForestRegressor |
| CV RMSE | 798 CHF (±164) |
| Holdout RMSE | 840 CHF |
| R² | 0.58 |

**Features used:**
- Numeric: `rooms`, `area`, `rooms_per_m2` (engineered)
- Categorical: `municipality` (one-hot, 112 classes)
- Binary flags extracted from listing text: `is_furnished`, `is_temporary`, `has_balcony`, `is_luxurious`, `is_zurich_city`

**Modeling approach:**
Two training iterations with cross-validation. Iteration 1 established a baseline (LinearRegression, RandomForest). Iteration 2 added feature engineering and hyperparameter tuning. RandomForest was selected as the final model in both iterations.

| Iteration | Model | CV RMSE | CV R² |
|---|---|---|---|
| 1 — Baseline | RandomForest v1 | 800 CHF | 0.58 |
| 1 — Baseline | LinearRegression | 880 CHF | 0.48 |
| 2 — Improved | RandomForest v2 | **798 CHF** | **0.58** |
| 2 — Improved | MLPRegressor | 886 CHF | 0.47 |

**Stack:** Python · scikit-learn · pandas · Gradio

---

### 2. Apartment Room Classifier
> Computer Vision · Transfer Learning · ViT · CLIP · GPT-4o · Hugging Face Spaces

Classifies apartment room images into 8 categories and compares three model approaches:
fine-tuned ViT, zero-shot CLIP, and GPT-4o.

**Live demo:** [huggingface.co/spaces/Scampolonii/apartment-room-classifier](https://huggingface.co/spaces/Scampolonii/apartment-room-classifier)  
**Trained model:** [huggingface.co/Scampolonii/vit-apartment-rooms](https://huggingface.co/Scampolonii/vit-apartment-rooms)

| Property | Value |
|---|---|
| Task | Multi-class image classification (8 classes) |
| Dataset | MIT Indoor Scenes — 8 apartment-relevant classes, ~15,571 images |
| Base model | `google/vit-base-patch16-224` |
| Training strategy | Transfer learning — only classifier head trained |
| Trainable parameters | 4,614 out of 85,803,270 |
| Test accuracy | **90.16%** |

**Classes:** `bathroom` · `bedroom` · `children's room` · `corridor` · `dining room` · `kitchen` · `living room` · `nursery`

**Model comparison (8 example images):**

| Model | Type | Accuracy | Notes |
|---|---|---|---|
| Fine-tuned ViT | Transfer learning | 4/8 (50%) | Strong on training distribution; weaker on atypical images |
| CLIP zero-shot | Open-source zero-shot | 6/8 (75%) | Good generalization without any fine-tuning |
| GPT-4o | Closed-source LLM | 8/8 (100%) | Best results; contextual understanding |

**Stack:** Python · PyTorch · HuggingFace Transformers · CLIP · OpenAI API · Gradio

---

## Repository Structure

```
apartment-price-predictor/
│
├── app.py                  # Price predictor — Gradio web interface
├── requirements.txt        # Price predictor dependencies
│
├── src/                    # Price predictor source package
│   ├── config.py           # Paths, column names, hyperparameters
│   ├── data_loader.py      # Data loading & cleaning
│   ├── features.py         # Feature engineering
│   ├── preprocessing.py    # sklearn ColumnTransformer
│   ├── train.py            # Training entry point
│   ├── evaluate.py         # Metrics & CV evaluation
│   └── predict.py          # Inference module
│
├── models/
│   ├── pipeline.joblib     # Trained sklearn pipeline
│   └── metadata.json       # Model metrics snapshot
│
├── results/
│   └── tables/
│       ├── model_comparison.csv   # All models × iterations
│       └── iterations.csv         # Summary per iteration
│
└── cv_app/                 # Room classifier (separate app)
    ├── app.py              # Classifier — Gradio web interface
    ├── requirements.txt    # Classifier dependencies
    ├── README.md           # Detailed classifier documentation
    └── examples/           # 8 example room images
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

### Room Classifier

```bash
cd cv_app
pip install -r requirements.txt
# Set OPENAI_API_KEY environment variable
python app.py  # → http://localhost:7860
```

---

*Academic ML project — Machine Learning Applications & Computer Vision, University course HS24/25.*
