# Zurich Apartment AI Suite

Three connected applied-AI prototypes for structured, textual, and visual
apartment data:

| Application | Modality | Role |
|---|---|---|
| [Price estimator](price_estimator/) | Structured listing data | Predicts an indicative monthly gross rent |
| [Conversational agent](conversational_agent/) | German free text → validated JSON | Collects inputs for the same price model |
| [Room classifier](room_classifier/) | Room images | Compares fine-tuned and zero-shot vision approaches |

This is a research and portfolio suite, not a production valuation service.
The rent data is small and restricted to the canton of Zurich. Results are
reported with their split design, class coverage, and known uncertainty.

Repository URL: [Scampoloni/apartment-price-predictor](https://github.com/Scampoloni/apartment-price-predictor).
`zurich-apartment-ai-suite` is the clearer project name, but the remote
repository has not been renamed: changing it without coordinating GitHub,
Hugging Face, clones, and portfolio links would create avoidable breakage.

## Application 1: price estimation

The estimator uses a scikit-learn `Pipeline` with median imputation,
`OneHotEncoder(handle_unknown="ignore")`, and a `RandomForestRegressor`.
Preprocessing is fitted only on training folds. Unknown municipalities are
accepted safely but produce an explicit warning.

Inputs available when a listing is viewed:

- rooms and living area;
- municipality;
- rooms per square metre;
- listing-text flags for furnished, temporary, balcony/terrace, and luxury
  language;
- a Zurich-city flag.

The target price, address, coordinates, and post-outcome information are not
used as features. Text flags are deterministic keyword features derived only
from the listing description available at prediction time.

Dataset evidence:

- 819 raw rows; 817 remain after documented validity filters;
- 93 observed municipalities, not 112;
- canton of Zurich only;
- raw listing data is intentionally excluded from Git and must be supplied
  locally with appropriate permission.

[Live price Space](https://huggingface.co/spaces/Scampolonii/apartment-price-predictor)
(URL verified 2026-07-20; Space was sleeping).

## Application 2: conversational agent

The agent is an input layer over the same regression artifact; it is not a
second pricing model.

1. An LLM extracts `rooms`, `area_m2`, `municipality`, and optional explicitly
   stated listing attributes.
2. Application code validates types, ranges, required fields, and municipality
   support.
3. The shared regression pipeline alone produces the numeric rent estimate.
4. A separate LLM call may write qualitative German prose, but it never
   receives the predicted price. Output containing a number or currency is
   rejected.
5. Deterministic application text displays the random and geographic error
   evidence and warns about unseen municipalities.

Malformed JSON, missing fields, unknown municipalities, forbidden price fields,
and price-like explanation output are covered by unit tests. API keys are read
only from `OPENAI_API_KEY`; no key is stored in the repository.

[Conversational Space](https://huggingface.co/spaces/Scampolonii/apartement-conversational-agent)
(URL verified 2026-07-20, but the deployed revision was in `RUNTIME_ERROR`;
local code and tests are the maintained reference).

## Application 3: room classifier

The vision app exposes:

- fine-tuned ViT: [`Scampolonii/vit-apartment-rooms`](https://huggingface.co/Scampolonii/vit-apartment-rooms);
- zero-shot CLIP: `openai/clip-vit-large-patch14`;
- optional GPT-4o vision for a small, paid qualitative demonstration.

Training used the existing train and validation splits from
[`keremberke/indoor-scene-classification`](https://huggingface.co/datasets/keremberke/indoor-scene-classification),
filtered to eight apartment-relevant labels. The filtered split sizes recorded
by the executed training run were 1,931 train, 975 validation, and 254 test.
Only the ViT classifier head was trained: 6,152 trainable parameters out of
85,804,808.

The full filtered test split was evaluated, but it contains labelled examples
for only five of the eight configured classes. Bathroom, bedroom, and
children's room have zero test support. Therefore the result is a full pass
over the available test split, but not a complete eight-class assessment.

[Live room-classifier Space](https://huggingface.co/spaces/Scampolonii/apartment-room-classifier)
(URL verified 2026-07-20; Space was sleeping).

## Shared architecture

```text
German request ──LLM extraction──> validated structured fields
                                          │
Listing form ─────────────────────────────┤
                                          ▼
                              shared rent pipeline
                                          │
                             numeric estimate + evidence

Room image ──> ViT / CLIP / optional qualitative GPT-4o comparison
```

The conversational application reuses the price artifact and uncertainty
metadata. Vision evaluation remains separate because its labels, splits, and
metrics are fundamentally different from rent regression.

## Results

### Rent regression

| Evaluation | Split discipline | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| Random holdout | 653 train / 164 test; municipalities may overlap | CHF 840 | CHF 504 | 0.563 |
| Geographic generalisation | 5-fold `GroupKFold`; every municipality absent from its test fold's training data | CHF 1,104 | CHF 657 | 0.235 |

The random result is preserved for comparison. The municipality-grouped result
is the more realistic test for a new location and is substantially weaker.
Its largest fold holds out all 274 Zurich-city rows, illustrating both genuine
geographic difficulty and the instability caused by an imbalanced, small
dataset. No fold has municipality overlap.

Random-holdout error analysis:

| Slice | n | RMSE | MAE | Note |
|---|---:|---:|---:|---|
| Rent below CHF 2,000 | 44 | CHF 458 | CHF 412 | R² is negative within this narrow band |
| CHF 2,000–2,999 | 77 | CHF 358 | CHF 268 | R² is negative within this narrow band |
| CHF 3,000 or more | 43 | CHF 1,500 | CHF 1,021 | Upper-tail errors dominate |
| Zurich city | 54 | CHF 1,090 | CHF 703 | Harder than the rest of the canton |
| Rest of canton | 110 | CHF 685 | CHF 407 | — |
| Frequent municipalities in training | 95 | CHF 889 | CHF 543 | At least 10 training rows |
| Sparse or unseen municipalities | 69 | CHF 768 | CHF 451 | Fewer than 10 training rows |
| Furnished flag | 6 | CHF 501 | CHF 437 | **Too few observations for a stable conclusion** |
| No furnished flag | 158 | CHF 850 | CHF 507 | — |

The largest anonymized residual examples are in
[`results/price_estimator/largest_residuals.csv`](results/price_estimator/largest_residuals.csv).
They omit addresses and listing text. Full aggregate evidence is in
[`results/price_estimator/evaluation_summary.json`](results/price_estimator/evaluation_summary.json).

### Room classification

| Model | Quantitative population | Accuracy | Macro F1 | Status |
|---|---|---:|---:|---|
| Fine-tuned ViT | All 254 filtered test images; only 5/8 classes have support | 90.16% | 91.67% across supported classes; 57.29% if all 8 configured classes are included | Completed |
| CLIP | Same labelled test population | — | — | Not run: the large checkpoint was not practical in the available CPU-only audit environment |
| GPT-4o | No quantitative test-set run | — | — | Kept qualitative to avoid material API cost |

ViT [per-class metrics](results/room_classifier/vit_per_class.csv) and the
[confusion matrix](results/room_classifier/vit_confusion_matrix.csv) are
committed. Zero-support classes are marked not evaluable. The shared evaluator
can run CLIP later with:

```bash
python -m room_classifier.evaluate --include-clip
```

### Non-representative qualitative examples

Exactly eight convenience-selected external images are stored in
`room_classifier/examples/`: one named example for bathroom, bedroom,
children's room, corridor, dining room, kitchen, living room, and nursery.
They were not randomly sampled, are outside the labelled test set, and their
original source URLs were not retained; they must not be treated as a
statistical benchmark.

Recorded top-one outcomes on these eight examples were ViT 4/8, CLIP 6/8, and
GPT-4o 8/8. These figures are qualitative observations only. They do not show
that GPT-4o is objectively best, do not establish statistical significance,
and are not mixed into the quantitative table above. No larger external
dataset is claimed or fabricated.

## Limitations

- The rent dataset is small, geographically narrow, and may not reflect current
  market conditions.
- Municipality one-hot encoding cannot learn a new municipality's local price
  level; safe handling is not the same as reliable generalisation.
- Random splitting is optimistic when the same municipalities appear on both
  sides.
- Price errors are much larger in the upper rent band.
- Furnished listings are too rare in the random test split for a stable
  subgroup conclusion.
- Keyword flags can miss synonyms, negation, or unusual listing language.
- ViT test coverage is only five of eight configured room classes.
- The external image gallery is selected and non-representative, and its source
  provenance is incomplete.
- CLIP has not yet received the same full-test quantitative run.
- GPT-4o comparison is intentionally small and paid; no significance is
  implied.
- None of the three applications should be used for production valuation,
  housing decisions, or claims of market coverage beyond the evidence above.

## Local setup

Use Python 3.11.

```bash
git clone https://github.com/Scampoloni/apartment-price-predictor.git
cd apartment-price-predictor
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

Price training and app:

```bash
# Supply, but do not commit:
# data/raw/original_apartment_data_analytics_hs24_with_lat_lon.csv
python -m price_estimator.src.train --iteration 2
python -m price_estimator.app
```

Conversational app:

```bash
pip install -r conversational_agent/requirements.txt
# Set OPENAI_API_KEY in the environment or deployment secret store.
python -m conversational_agent.app
```

Room app and evaluation:

```bash
pip install -r room_classifier/requirements.txt
python -m room_classifier.app
python -m room_classifier.evaluate
```

The default room evaluation runs ViT only. `--include-clip` downloads and runs
the much larger CLIP model. GPT-4o is never called by the evaluation script.

Tests:

```bash
pytest -q
```

CI runs syntax checks, unit tests, structured-JSON validation, and a small
synthetic price-pipeline smoke test. It does not download large vision models.

## Repository structure

```text
.
├── price_estimator/       # regression app, training, evaluation, artifact
├── conversational_agent/ # JSON extraction, validation, shared-model frontend
├── room_classifier/      # ViT/CLIP/GPT app and labelled-test evaluator
├── results/              # committed aggregate evidence, no raw listings
├── tests/                # unit, import, and smoke tests
├── .github/workflows/    # lightweight CI
└── app.py                # compatibility launcher for the price Space
```

## My contribution and AI-assisted development

My project contribution covers dataset preparation and validation, rent feature
engineering, leakage-safe preprocessing, random and municipality-grouped model
evaluation, residual analysis, integration of the conversational input layer
with the regression artifact, prompt constraints and JSON validation, vision
model training/evaluation workflow, and Hugging Face deployment setup.

AI coding support was used for implementation assistance, refactoring, tests,
and documentation review. Model choices, data inclusion, evaluation design,
metric interpretation, prompts, validation rules, and final claims remain
human-reviewed project decisions. AI assistance is not presented as a source
of ground-truth prices, labels, municipality coverage, or fabricated
experiments.
