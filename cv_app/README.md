---
title: Apartment Room Classifier
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Apartment Room Classifier

This app compares 3 image classification approaches on apartment room images:

- Fine-tuned ViT model [`Scampolonii/vit-apartment-rooms`](https://huggingface.co/Scampolonii/vit-apartment-rooms)
- Zero-shot CLIP (`openai/clip-vit-large-patch14`)
- OpenAI vision model (GPT-4o image classification)

## Dataset Used For Training

- Hugging Face dataset: `keremberke/indoor-scene-classification` (MIT Indoor Scenes, repackaged by Roboflow)
- Original source: [MIT Indoor Scene Recognition](http://web.mit.edu/torralba/www/indoor.html)
- Total images: ~15,571 (train: 10,885 / val: 3,128 / test: 1,558)
- Number of classes used: **8** (filtered from 67 total MIT classes)

**Classes used:**
`bathroom`, `bedroom`, `children's room`, `corridor`, `dining room`, `kitchen`, `living room`, `nursery`

## Trained Model

- Hugging Face model link: [https://huggingface.co/Scampolonii/vit-apartment-rooms](https://huggingface.co/Scampolonii/vit-apartment-rooms)
- Base model: `google/vit-base-patch16-224`
- Training strategy: Transfer learning — all ViT layers frozen, only classifier head trained (4,614 trainable parameters out of 85,803,270)

## Preprocessing Steps

- Filtered MIT Indoor Scenes to 8 apartment-relevant room categories
- Remapped labels to new 0-based IDs
- Split dataset: 80% train / 10% validation / 10% test
- All images converted to RGB
- Resized to 224×224 pixels (via ViTImageProcessor)
- Normalized with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
- Rescale factor: 1/255

## Training Performance

| Epoch | Steps | Train Loss | Test Accuracy |
|---:|---:|---:|---:|
| 5 | 305 | 0.3851 | 0.9016 |

> Per-epoch metrics were not logged (`report_to="none"`). Final test accuracy: **90.16%** on the held-out test set.

## Evaluation

- Metric: **Accuracy**
- Evaluated using Hugging Face `evaluate` library
- Test set accuracy: **90.16%** (eval_loss: 0.3788)

## Example Image Results

The table below reports the true class and predictions for all three models on 8 example images.

| Image | True Class | ViT Top-3 (score) | CLIP Top-3 (score) | GPT-4o (label, confidence) |
|---|---|---|---|---|
| `bathroom.jpg` | `bathroom` | `corridor` (0.77)<br>`living room` (0.10)<br>`kitchen` (0.05) | `corridor` (0.93)<br>`bathroom` (0.04)<br>`bedroom` (0.01) | `bathroom` (0.85) |
| `bedroom.jpg` | `bedroom` | `living room` (0.51)<br>`nursery` (0.13)<br>`children's room` (0.09) | `bedroom` (0.90)<br>`children's room` (0.02)<br>`living room` (0.01) | `bedroom` (0.90) |
| `kitchen.jpg` | `kitchen` | `living room` (0.26)<br>`children's room` (0.23)<br>`corridor` (0.14) | `kitchen` (0.95)<br>`dining room` (0.04)<br>`living room` (0.01) | `kitchen` (0.90) |
| `living_room.jpg` | `living room` | `dining room` (0.51)<br>`living room` (0.24)<br>`children's room` (0.11) | `children's room` (0.61)<br>`nursery` (0.13)<br>`living room` (0.11) | `living room` (0.90) |
| `corridor.jpg` | `corridor` | `corridor` (0.92)<br>`living room` (0.05)<br>`dining room` (0.02) | `corridor` (0.96)<br>`nursery` (0.02)<br>`dining room` (0.01) | `corridor` (0.90) |
| `dining_room.jpg` | `dining room` | `dining room` (0.45)<br>`living room` (0.30)<br>`children's room` (0.12) | `dining room` (0.85)<br>`living room` (0.07)<br>`kitchen` (0.05) | `dining room` (0.88) |
| `childrens_room.jpg` | `children's room` | `living room` (0.38)<br>`children's room` (0.23)<br>`dining room` (0.21) | `children's room` (0.57)<br>`nursery` (0.40)<br>`dining room` (0.01) | `children's room` (0.90) |
| `nursery.jpg` | `nursery` | `nursery` (0.79)<br>`children's room` (0.16)<br>`living room` (0.03) | `nursery` (0.87)<br>`children's room` (0.00)<br>`bedroom` (0.00) | `nursery` (0.95) |

## Model Comparison Summary

| Model | Type | Top-1 Correct (8 samples) | Notes |
|---|---|---|---|
| Fine-tuned ViT | Transfer learning (fine-tuned) | 4/8 (50%) | Struggles with artistic/painting images |
| CLIP zero-shot | Open-source zero-shot | 6/8 (75%) | Strong generalization without fine-tuning |
| GPT-4o | Closed-source LLM vision | 8/8 (100%) | Best results, understands context |

## Links

- **Live App (HF Space):** [https://huggingface.co/spaces/Scampolonii/apartment-room-classifier](https://huggingface.co/spaces/Scampolonii/apartment-room-classifier)
- **Model:** [https://huggingface.co/Scampolonii/vit-apartment-rooms](https://huggingface.co/Scampolonii/vit-apartment-rooms)
