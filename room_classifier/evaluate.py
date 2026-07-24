"""Evaluate ViT and optionally CLIP on the same full labelled test split.

This script never calls Claude. The eight gallery images remain a separate,
qualitative demonstration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from room_classifier.core import (
    CLIP_PROMPTS,
    DATASET_LABELS,
    DATASET_TO_DISPLAY,
    LABELS,
)

DATASET_ID = "keremberke/indoor-scene-classification"
DATASET_CONFIG = "full"
DATASET_REVISION = "030dcb9ec61c436299b1df10d90ae1cbe1d1b401"
VIT_MODEL_ID = "Scampolonii/vit-apartment-rooms"
VIT_MODEL_REVISION = "657d11121d10182427776994df45817a4e2fa9cf"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def load_labelled_test_set():
    """Load and filter the dataset's existing held-out test split."""
    from datasets import load_dataset

    raw = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        revision=DATASET_REVISION,
        trust_remote_code=True,
    )
    test = raw["test"]
    label_column = "labels" if "labels" in test.features else "label"
    names = test.features[label_column].names
    keep_ids = {names.index(label) for label in DATASET_LABELS}
    filtered = test.filter(lambda batch: [v in keep_ids for v in batch[label_column]], batched=True)
    old_to_new = {
        names.index(dataset_label): index
        for index, dataset_label in enumerate(DATASET_LABELS)
    }
    references = np.asarray([old_to_new[value] for value in filtered[label_column]])
    return raw, filtered, references


def _batch_predict(
    images,
    predictor: Callable[[list], np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    predictions: list[int] = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        predictions.extend(predictor(images[start:stop]).tolist())
    return np.asarray(predictions)


def predict_vit(images, *, batch_size: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(
        VIT_MODEL_ID,
        revision=VIT_MODEL_REVISION,
        use_fast=False,
    )
    model = AutoModelForImageClassification.from_pretrained(
        VIT_MODEL_ID,
        revision=VIT_MODEL_REVISION,
    ).eval()
    model_label_to_index = {
        str(label): index for index, label in model.config.id2label.items()
    }
    output_to_suite = {
        output_index: LABELS.index(label)
        for label, output_index in model_label_to_index.items()
    }

    def run(batch):
        inputs = processor(
            images=[image.convert("RGB") for image in batch],
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs).logits.argmax(dim=1).cpu().numpy()
        return np.asarray([output_to_suite[int(value)] for value in outputs])

    return _batch_predict(images, run, batch_size=batch_size)


def predict_clip(images, *, batch_size: int = 8) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_ID,
        revision=CLIP_MODEL_REVISION,
    )
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        revision=CLIP_MODEL_REVISION,
    ).eval()

    def run(batch):
        inputs = processor(
            text=CLIP_PROMPTS,
            images=[image.convert("RGB") for image in batch],
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        return logits.argmax(dim=1).cpu().numpy()

    return _batch_predict(images, run, batch_size=batch_size)


def calculate_metrics(
    references: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    report = classification_report(
        references,
        predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    supported_indices = sorted(set(int(value) for value in references))
    missing_indices = [
        index for index in range(len(LABELS)) if index not in supported_indices
    ]
    summary = {
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1_supported_classes": float(
            f1_score(
                references,
                predictions,
                labels=supported_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_all_configured_classes": float(
            f1_score(
                references,
                predictions,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
        "n_test": int(len(references)),
        "supported_classes": [LABELS[index] for index in supported_indices],
        "missing_test_classes": [LABELS[index] for index in missing_indices],
        "class_coverage": f"{len(supported_indices)}/{len(LABELS)}",
        "coverage_warning": (
            "The official filtered test split does not contain every configured "
            "class; eight-class performance is therefore not fully identified."
        ),
    }
    per_class = pd.DataFrame(
        [
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": (
                    report[label]["recall"]
                    if report[label]["support"] > 0
                    else np.nan
                ),
                "f1": (
                    report[label]["f1-score"]
                    if report[label]["support"] > 0
                    else np.nan
                ),
                "support": int(report[label]["support"]),
                "evaluable": bool(report[label]["support"] > 0),
                "warning": (
                    ""
                    if report[label]["support"] > 0
                    else "No labelled examples for this class in the test split."
                ),
            }
            for label in LABELS
        ]
    )
    matrix = pd.DataFrame(
        confusion_matrix(
            references,
            predictions,
            labels=list(range(len(LABELS))),
        ),
        index=LABELS,
        columns=LABELS,
    )
    return summary, per_class, matrix


def save_model_results(
    output_dir: Path,
    model_name: str,
    summary: dict,
    per_class: pd.DataFrame,
    matrix: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(output_dir / f"{model_name}_per_class.csv", index=False)
    matrix.to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
    (output_dir / f"{model_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-clip",
        action="store_true",
        help="Also download/evaluate the much larger CLIP checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "room_classifier",
    )
    args = parser.parse_args()

    raw, test, references = load_labelled_test_set()
    images = test["image"]
    dataset_evidence = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "vit_model_revision": VIT_MODEL_REVISION,
        "source_split": "test",
        "raw_split_rows": {name: len(split) for name, split in raw.items()},
        "filtered_test_rows": int(len(test)),
        "class_counts": {
            LABELS[index]: int((references == index).sum())
            for index in range(len(LABELS))
        },
        "gallery_images_used_for_metrics": 0,
        "gpt4o_images_used_for_quantitative_metrics": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_evidence.json").write_text(
        json.dumps(dataset_evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    vit_predictions = predict_vit(images, batch_size=args.batch_size)
    vit_summary, vit_per_class, vit_matrix = calculate_metrics(
        references,
        vit_predictions,
    )
    save_model_results(
        args.output_dir,
        "vit",
        vit_summary,
        vit_per_class,
        vit_matrix,
    )
    print("ViT:", vit_summary)

    if args.include_clip:
        clip_predictions = predict_clip(
            images,
            batch_size=max(1, args.batch_size // 2),
        )
        clip_summary, clip_per_class, clip_matrix = calculate_metrics(
            references,
            clip_predictions,
        )
        save_model_results(
            args.output_dir,
            "clip",
            clip_summary,
            clip_per_class,
            clip_matrix,
        )
        print("CLIP:", clip_summary)
    else:
        print(
            "CLIP full-test evaluation skipped. Re-run with --include-clip "
            "to download and evaluate the large checkpoint."
        )


if __name__ == "__main__":
    main()
