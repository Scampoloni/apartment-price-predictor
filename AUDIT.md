# Repository audit

Audit date: 2026-07-20

## Resolved findings

| Area | Finding | Resolution |
|---|---|---|
| Positioning | The project contained three apps but was titled as a single predictor. | Branded locally as **Zurich Apartment AI Suite**; remote URL retained to avoid breaking links. |
| Dataset claim | README claimed 819 listings and 112 municipalities. | Reports 819 raw rows, 817 usable rows, and 93 observed municipalities. |
| Price methodology | Random splitting allowed municipality overlap. | Preserved that benchmark and added five-fold municipality-disjoint `GroupKFold`. |
| Geographic result | No new-location evidence existed. | Added RMSE, MAE, R², fold membership, and zero-overlap assertions. |
| Error analysis | No subgroup or residual analysis. | Added price bands, Zurich/rest, frequent/sparse municipality, furnishing warning, and anonymized largest residuals. |
| Leakage | Feature engineering happened before splitting, without an explicit audit. | Verified it is deterministic and target-free; learned preprocessing remains inside pipelines fitted on training data only. Tests change the target and assert engineered features do not change. |
| Unknown categories | Coverage and behavior were unclear. | One-hot encoding ignores unseen values safely and both UIs warn that safe execution is not geographic support. |
| Conversational model | Used a separate seven-feature national BFS model, inconsistent with the Zurich suite. | Removed the duplicate pickle and BFS table; agent now calls the shared Zurich pipeline. |
| LLM price control | Prompt-only protection could still allow price output. | Price fields are rejected, the explanation call never receives the prediction, and numeric/currency explanation output is rejected. |
| Conversation tests | Malformed JSON, unknown locations, and missing fields were untested. | Added API-free unit tests for all requested cases. |
| Room evaluation | Only accuracy was reported. | Evaluated the complete 254-image filtered test split and saved per-class metrics and a confusion matrix. |
| Room coverage | README implied an eight-class test. | Disclosed that only five of eight classes have test support; unsupported classes are not evaluable. |
| ViT parameters | README claimed 4,614 trainable parameters. | Corrected to 6,152 for an eight-class `768 × 8 + 8` head. |
| Dataset split | README claimed a new 80/10/10 split. | Corrected: the provided train/validation/test splits were preserved. |
| Vision comparison | 4/8, 6/8, and 8/8 were presented as model performance and GPT-4o as best. | Relabelled as selected, non-representative qualitative examples with no significance claim. |
| Closed-source vision cost | Quantitative scope was unclear. | Zero paid vision audit calls; Claude remains optional and qualitative. |
| CLIP evaluation | Eight examples were the only evidence. | Added a same-test evaluator; full CLIP run remains explicitly incomplete due large CPU-only cost, so no quantitative CLIP metrics are claimed. |
| API keys | Environment handling existed but was not tested/documented consistently. | Repository scan found no committed key; apps read environment variables only; `.env` remains ignored. |
| Dependencies | Overlapping, unbounded app requirements and heavy imports complicated CI. | Added shared/lightweight requirements, separate vision requirements, version bounds, and lazy model loading. |
| Structure | Root `src`, `models`, `cv_app`, and two unrelated regressors obscured ownership. | Added explicit `price_estimator`, `conversational_agent`, and `room_classifier` directories plus a thin root compatibility launcher. |
| CI | No workflow or tests. | Added Python 3.11 syntax checks, unit tests, JSON validation, and a small synthetic price smoke test without vision downloads. |

## Artifact and data notes

- `price_estimator/models/pipeline.joblib` is a Git LFS artifact and matches the
  nine documented features.
- The stale conversational `random_forest_regression.pkl` was removed.
- Raw Zurich listings remain excluded from Git. Their acquisition permission
  and long-term redistribution rights cannot be verified from this repository,
  so no new listing data was scraped or committed.
- Aggregate result files contain no address or listing text.
- The eight external room images lack retained source URLs. Their provenance is
  incomplete; this is disclosed and no new images were downloaded.
- The old unexecuted room-training notebook and mutable image-download helpers
  were replaced by explicit `train.py` and `evaluate.py` entrypoints.
- Pre-existing local working copies `notebook_training(1).ipynb` and
  `conversational_agent/doc/` were preserved on disk and explicitly ignored;
  they were not folded into the repository changes.

## Link audit

Verified through Hugging Face API responses on 2026-07-20:

| Resource | Resolution | Runtime state |
|---|---|---|
| Price Space | URL exists | `SLEEPING` |
| Conversational Space | URL exists | `RUNTIME_ERROR` |
| Room-classifier Space | URL exists | `SLEEPING` |
| ViT model | URL exists | Model card and files available |

The conversational URL is not labelled healthy merely because it resolves.
Deployment repair requires publishing the restructured code and setting the
secret in the Space; this local repository change does not claim that external
deployment has already been updated.

## Remaining limitations

- The repository has no licensed, redistributable rent fixture representing the
  full real distribution; CI therefore uses synthetic smoke data only.
- The grouped price result is sensitive to Zurich city's 274-row concentration.
- CLIP still needs the full same-test run on suitable hardware.
- The room test split cannot measure three configured classes.
- External image provenance should be reconstructed or the gallery replaced
  with fully attributed assets before broader publication.
- The hosted ViT model card and deployed Spaces must be republished separately
  to inherit the corrected local documentation and code.
