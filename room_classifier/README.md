---
title: Zurich Apartment AI - Room Classifier
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Room-image classification comparison

The app compares a fine-tuned ViT, zero-shot CLIP, and an optional GPT-4o
single-image classification. Quantitative test evidence and external examples
are deliberately separated.

## Dataset and training

Source: [`keremberke/indoor-scene-classification`](https://huggingface.co/datasets/keremberke/indoor-scene-classification),
a repackaging of MIT Indoor Scenes.

Raw provided splits:

| Split | All 67 classes |
|---|---:|
| Train | 10,885 |
| Validation | 3,128 |
| Test | 1,558 |

After filtering to the configured room labels, the executed run recorded 1,931
train, 975 validation, and 254 test images. The provided splits were preserved;
the code did not create an 80/10/10 split.

Configured labels: bathroom, bedroom, children's room, corridor, dining room,
kitchen, living room, and nursery.

ViT base: `google/vit-base-patch16-224`. Only the eight-class classifier head
was trainable: 6,152 of 85,804,808 parameters. Training uses only the filtered
train split, model selection uses validation, and final metrics use test.

## Full filtered-test ViT evaluation

All 254 filtered test images were evaluated. The test split itself covers only
five configured classes:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Bathroom | 0 | — | — | — |
| Bedroom | 0 | — | — | — |
| Children's room | 0 | — | — | — |
| Corridor | 6 | 0.857 | 1.000 | 0.923 |
| Dining room | 97 | 0.975 | 0.794 | 0.875 |
| Kitchen | 83 | 0.952 | 0.964 | 0.958 |
| Living room | 57 | 0.789 | 0.982 | 0.875 |
| Nursery | 11 | 1.000 | 0.909 | 0.952 |

- Accuracy: **90.16%**
- Macro F1 across the five supported classes: **91.67%**
- Macro F1 when all eight configured classes are included and unsupported
  classes receive zero: **57.29%**

This reproduces the earlier accuracy but narrows its interpretation: it is not
a complete eight-class test because three classes have no labelled test
examples.

Evidence:

- [`results/room_classifier/dataset_evidence.json`](../results/room_classifier/dataset_evidence.json)
- [`results/room_classifier/vit_summary.json`](../results/room_classifier/vit_summary.json)
- [`results/room_classifier/vit_per_class.csv`](../results/room_classifier/vit_per_class.csv)
- [`results/room_classifier/vit_confusion_matrix.csv`](../results/room_classifier/vit_confusion_matrix.csv)

## CLIP and GPT-4o status

The evaluator supports CLIP on the identical labelled test set, but that
full-test run was not completed during the CPU-only audit because the
`openai/clip-vit-large-patch14` checkpoint is substantially larger and was not
cached. No quantitative CLIP metrics are claimed.

```bash
python -m room_classifier.evaluate --include-clip
```

GPT-4o was not called during quantitative evaluation. A full paid test-set pass
would incur material API cost, so it remains a small qualitative comparison.
It is not ranked as objectively best.

## Non-representative qualitative gallery

The gallery contains exactly eight convenience-selected external files, one
named example per configured class:

`bathroom.jpg`, `bedroom.jpg`, `childrens_room.jpg`, `corridor.jpg`,
`dining_room.jpg`, `kitchen.jpg`, `living_room.jpg`, and `nursery.jpg`.

Recorded top-one results:

| Model | Correct among eight selected examples | Interpretation |
|---|---:|---|
| Fine-tuned ViT | 4/8 | Qualitative observation only |
| CLIP zero-shot | 6/8 | Qualitative observation only |
| GPT-4o vision | 8/8 | Qualitative observation only; no significance claim |

The images are not random, not part of the labelled test set, and not large
enough for inference about model quality. Original source URLs were not
retained, so provenance is incomplete and must be resolved before reusing the
gallery beyond this demonstration.

## Run

```bash
pip install -r room_classifier/requirements.txt
python -m room_classifier.app
python -m room_classifier.evaluate
```

Models are loaded lazily. Import and unit tests do not download ViT or CLIP.
`OPENAI_API_KEY` is optional and read from the environment only.
