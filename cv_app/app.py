import os
import base64
import json
from io import BytesIO

import torch
import gradio as gr
from transformers import pipeline, CLIPProcessor, CLIPModel
import openai

# ── Configuration ────────────────────────────────────────────────────────────
HF_MODEL_ID = "Scampolonii/vit-apartment-rooms"

LABELS = [
    "bathroom",
    "bedroom",
    "children's room",
    "corridor",
    "dining room",
    "kitchen",
    "living room",
    "nursery",
]

CLIP_PROMPTS = [f"a photo of a {label} in an apartment" for label in LABELS]
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "examples")

# ── Load models (once at startup) ────────────────────────────────────────────
print("Loading ViT model...")
vit_classifier = pipeline("image-classification", model=HF_MODEL_ID)

print("Loading CLIP model...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval()

print("Models ready.")


# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_vit(image):
    """Return top-3 predictions from fine-tuned ViT."""
    results = vit_classifier(image.convert("RGB"))[:3]
    return {r["label"]: round(r["score"], 4) for r in results}


def predict_clip(image):
    """Return top-3 zero-shot predictions from CLIP."""
    inputs = clip_processor(
        text=CLIP_PROMPTS,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image[0].softmax(dim=0)
    top3 = probs.topk(3)
    return {LABELS[i]: round(float(probs[i]), 4) for i in top3.indices}


def predict_openai(image):
    """Return top-1 prediction from GPT-4o vision."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"API key not set": 0.0}

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    labels_str = ", ".join(f'"{l}"' for l in LABELS)
    prompt = (
        f"Classify this room image into exactly one of these categories: [{labels_str}]. "
        f'Respond with JSON only, no markdown: {{"label": "...", "confidence": 0.0}}'
    )

    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=80,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip().strip("`").lstrip("json").strip()
        result = json.loads(raw)
        return {result["label"]: round(float(result["confidence"]), 4)}
    except Exception as e:
        return {f"Error: {e}": 0.0}


# ── Main prediction function ──────────────────────────────────────────────────
def classify(image):
    if image is None:
        return {}, {}, {}
    vit_out = predict_vit(image)
    clip_out = predict_clip(image)
    openai_out = predict_openai(image)
    return vit_out, clip_out, openai_out


# ── Collect example images ────────────────────────────────────────────────────
def get_examples():
    if not os.path.isdir(EXAMPLE_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        [os.path.join(EXAMPLE_DIR, f)]
        for f in sorted(os.listdir(EXAMPLE_DIR))
        if os.path.splitext(f)[1].lower() in exts
    ]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Apartment Room Classifier") as demo:
    gr.Markdown(
        """
        # Apartment Room Classifier
        Upload a room image to compare predictions from three different models:
        - **Fine-tuned ViT** — transfer learning on MIT Indoor Scenes (apartment rooms)
        - **CLIP** — zero-shot open-source model (`openai/clip-vit-large-patch14`)
        - **GPT-4o** — closed-source OpenAI vision model
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Room Image")

    classify_btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        vit_output = gr.Label(num_top_classes=3, label="Fine-tuned ViT (Top-3)")
        clip_output = gr.Label(num_top_classes=3, label="CLIP Zero-Shot (Top-3)")
        openai_output = gr.Label(num_top_classes=1, label="GPT-4o Vision")

    examples = get_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_input,
            label="Example Images",
        )

    classify_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[vit_output, clip_output, openai_output],
    )

if __name__ == "__main__":
    demo.launch()
