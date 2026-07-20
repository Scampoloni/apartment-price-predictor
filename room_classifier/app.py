"""Interactive comparison of ViT, CLIP, and an optional GPT-4o example."""

from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path

import gradio as gr
from openai import OpenAI

from room_classifier.core import CLIP_PROMPTS, LABELS, parse_vision_response

HF_MODEL_ID = "Scampolonii/vit-apartment-rooms"
HF_MODEL_REVISION = "657d11121d10182427776994df45817a4e2fa9cf"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
EXAMPLE_DIR = Path(__file__).resolve().parent / "examples"


@lru_cache(maxsize=1)
def _load_vit_classifier():
    from transformers import pipeline

    return pipeline(
        "image-classification",
        model=HF_MODEL_ID,
        revision=HF_MODEL_REVISION,
    )


@lru_cache(maxsize=1)
def _load_clip():
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_ID,
        revision=CLIP_MODEL_REVISION,
    )
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        revision=CLIP_MODEL_REVISION,
    ).eval()
    return processor, model


def predict_vit(image) -> dict[str, float]:
    """Return top-three predictions from the fine-tuned ViT."""
    results = _load_vit_classifier()(image.convert("RGB"), top_k=3)
    return {item["label"]: round(float(item["score"]), 4) for item in results}


def predict_clip(image) -> dict[str, float]:
    """Return top-three zero-shot predictions from CLIP."""
    import torch

    processor, model = _load_clip()
    inputs = processor(
        text=CLIP_PROMPTS,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        probabilities = model(**inputs).logits_per_image[0].softmax(dim=0)
    indices = probabilities.topk(3).indices.tolist()
    return {
        LABELS[index]: round(float(probabilities[index]), 4)
        for index in indices
    }


def predict_openai(image) -> dict[str, float]:
    """Run one explicitly qualitative GPT-4o comparison when a key is set."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"API key not set — qualitative comparison skipped": 0.0}

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    labels = ", ".join(f'"{label}"' for label in LABELS)
    prompt = (
        f"Classify the image as exactly one of [{labels}]. "
        'Return JSON only: {"label": "...", "confidence": 0.0}.'
    )
    try:
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=80,
            temperature=0,
        )
        return parse_vision_response(response.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        return {f"Error: {exc}": 0.0}


def classify(image):
    if image is None:
        return {}, {}, {}
    return predict_vit(image), predict_clip(image), predict_openai(image)


def get_examples() -> list[list[str]]:
    if not EXAMPLE_DIR.is_dir():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        [str(path)]
        for path in sorted(EXAMPLE_DIR.iterdir())
        if path.suffix.lower() in extensions
    ]


with gr.Blocks(title="Zurich Apartment AI Suite — Room Classifier") as demo:
    gr.Markdown(
        """
        # Zurich Apartment AI Suite — Room Classifier

        Compare a fine-tuned ViT and zero-shot CLIP. GPT-4o is optional and
        shown only as a qualitative single-image comparison because API calls
        cost money. The eight gallery images are not an evaluation dataset.
        """
    )
    image_input = gr.Image(type="pil", label="Upload room image")
    classify_button = gr.Button("Classify", variant="primary")
    with gr.Row():
        vit_output = gr.Label(num_top_classes=3, label="Fine-tuned ViT")
        clip_output = gr.Label(num_top_classes=3, label="CLIP zero-shot")
        openai_output = gr.Label(
            num_top_classes=1,
            label="GPT-4o qualitative example",
        )
    examples = get_examples()
    if examples:
        gr.Examples(examples=examples, inputs=image_input, label="Eight examples")
    classify_button.click(
        fn=classify,
        inputs=image_input,
        outputs=[vit_output, clip_output, openai_output],
    )


if __name__ == "__main__":
    demo.launch()
