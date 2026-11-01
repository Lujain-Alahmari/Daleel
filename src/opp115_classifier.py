import json
import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "opp115_distilbert_other_f2_experiment"
)
MAX_LENGTH = 256
BATCH_SIZE = 8


@lru_cache(maxsize=1)
def load_classifier():
    results = json.loads(
        (MODEL_DIR / "results.json").read_text(encoding="utf-8")
    )
    thresholds = results["thresholds"]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    labels = [
        model.config.id2label[i]
        for i in range(model.config.num_labels)
    ]

    missing = [label for label in labels if label not in thresholds]
    if missing:
        raise ValueError(f"Missing thresholds: {missing}")

    return tokenizer, model, labels, thresholds, device


def classify_policy(text: str) -> dict:
    """Classify privacy-policy paragraphs using the saved OPP-115 model."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Privacy-policy text cannot be empty.")

    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\n\s*\n+", text.strip())
    ]
    paragraphs = [p for p in paragraphs if p]

    tokenizer, model, labels, thresholds, device = load_classifier()
    all_results = []

    for start in range(0, len(paragraphs), BATCH_SIZE):
        batch = paragraphs[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            batch_scores = torch.sigmoid(model(**inputs).logits).cpu().tolist()

        for offset, scores in enumerate(batch_scores):
            predictions = [
                {
                    "label": label,
                    "score": float(score),
                    "threshold": float(thresholds[label]),
                    "predicted": float(score) >= float(thresholds[label]),
                }
                for label, score in zip(labels, scores)
            ]
            predictions.sort(key=lambda item: item["score"], reverse=True)

            all_results.append({
                "paragraph_number": start + offset + 1,
                "text": batch[offset],
                "predicted_categories": [
                    item["label"] for item in predictions if item["predicted"]
                ],
                "predictions": predictions,
            })

    return {
        "model": str(MODEL_DIR),
        "paragraph_count": len(all_results),
        "results": all_results,
        "note": "Scores are model outputs, not calibrated confidence.",
    }
