"""Reproducible ViT classifier-head training entrypoint.

The original dataset's train, validation, and test splits are preserved.
Only apartment-relevant labels are filtered and remapped; the test split is
never used for fitting or model selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from room_classifier.core import DATASET_LABELS, LABELS
from room_classifier.evaluate import DATASET_CONFIG, DATASET_ID, DATASET_REVISION

BASE_MODEL = "google/vit-base-patch16-224"


def load_filtered_splits() -> DatasetDict:
    raw = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        revision=DATASET_REVISION,
        trust_remote_code=True,
    )
    label_column = "labels" if "labels" in raw["train"].features else "label"
    names = raw["train"].features[label_column].names
    keep_ids = {names.index(label) for label in DATASET_LABELS}
    old_to_new = {
        names.index(label): index for index, label in enumerate(DATASET_LABELS)
    }
    filtered = DatasetDict()
    for split_name, split in raw.items():
        selected = split.filter(
            lambda batch: [value in keep_ids for value in batch[label_column]],
            batched=True,
        )
        filtered[split_name] = selected.map(
            lambda row: {"label": old_to_new[row[label_column]]}
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("vit-apartment-rooms"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--hub-model-id",
        default="Scampolonii/vit-apartment-rooms",
    )
    args = parser.parse_args()

    dataset = load_filtered_splits()
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL, use_fast=False)

    def transform(batch):
        inputs = processor(
            images=[image.convert("RGB") for image in batch["image"]],
            return_tensors="pt",
        )
        inputs["labels"] = batch["label"]
        return inputs

    processed = dataset.with_transform(transform)

    def collate(batch):
        import torch

        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": torch.tensor([item["labels"] for item in batch]),
        }

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    for parameter in model.vit.parameters():
        parameter.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    def compute_metrics(output):
        predictions = np.argmax(output.predictions, axis=1)
        return {
            "accuracy": accuracy_score(output.label_ids, predictions),
            "macro_f1": f1_score(
                output.label_ids,
                predictions,
                average="macro",
                zero_division=0,
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        learning_rate=5e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        seed=42,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        data_collator=collate,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    print(trainer.evaluate(processed["test"]))
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
