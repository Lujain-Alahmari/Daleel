import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("models/opp115_distilbert_other_f2_experiment")
MAX_LENGTH = 256
BATCH_SIZE = 8

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")

results_path = MODEL_DIR / "results.json"
if not results_path.exists():
    raise FileNotFoundError(f"Results file not found: {results_path}")

results = json.loads(results_path.read_text(encoding="utf-8"))
thresholds = results["thresholds"]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Loading saved model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

# Read the full policy, then classify each paragraph separately.
print("\nPaste the privacy-policy text below.")
print("When finished, type END on a new line and press Enter.\n")

lines = []
while True:
    try:
        line = input()
    except EOFError:
        break
    if line.strip() == "END":
        break
    lines.append(line)

text = "\n".join(lines).strip()

if not text:
    raise SystemExit("No text entered. Please run the script again.")

# Blank lines define paragraph boundaries.
paragraphs = [
    re.sub(r"\s+", " ", part).strip()
    for part in re.split(r"\n\s*\n+", text)
]
paragraphs = [p for p in paragraphs if p]

if not paragraphs:
    raise SystemExit("No paragraphs found in the supplied text.")

id2label = model.config.id2label
labels = [
    id2label.get(i, id2label.get(str(i)))
    for i in range(model.config.num_labels)
]

missing_thresholds = [label for label in labels if label not in thresholds]
if missing_thresholds:
    raise ValueError(
        "Missing thresholds for labels: " + ", ".join(missing_thresholds)
    )

all_results = []

for start in range(0, len(paragraphs), BATCH_SIZE):
    batch_paragraphs = paragraphs[start:start + BATCH_SIZE]

    inputs = tokenizer(
        batch_paragraphs,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        batch_scores = torch.sigmoid(model(**inputs).logits).cpu().tolist()

    for offset, scores in enumerate(batch_scores):
        paragraph_number = start + offset + 1
        predictions = []

        for label, score in zip(labels, scores):
            threshold = float(thresholds[label])
            predictions.append({
                "label": label,
                "score": float(score),
                "threshold": threshold,
                "predicted": float(score) >= threshold,
            })

        predictions.sort(key=lambda item: item["score"], reverse=True)
        selected = [item for item in predictions if item["predicted"]]

        all_results.append({
            "paragraph_number": paragraph_number,
            "text": batch_paragraphs[offset],
            "predicted_categories": [item["label"] for item in selected],
            "predictions": predictions,
        })

print("\n" + "=" * 75)
print("PRIVACY POLICY PARAGRAPH CLASSIFICATION")
print("=" * 75)
print(f"Paragraphs processed: {len(all_results)}")
print(f"Model: {MODEL_DIR}")
print(f"Device: {device}")

for result in all_results:
    print("\n" + "-" * 75)
    print(f"PARAGRAPH {result['paragraph_number']}")
    print("-" * 75)
    print(result["text"])

    selected = [
        item for item in result["predictions"] if item["predicted"]
    ]
    print("\nPredicted categories:")
    if selected:
        for item in selected:
            print(
                f"- {item['label']}: score={item['score']:.3f} "
                f"(threshold={item['threshold']:.2f})"
            )
    else:
        print("No category passed its threshold.")

# Save structured output for later integration into PDPL-Copilot.
output_path = Path("outputs")
output_path.mkdir(parents=True, exist_ok=True)
json_path = output_path / "policy_paragraph_predictions.json"

json_path.write_text(
    json.dumps({
        "model": str(MODEL_DIR),
        "paragraph_count": len(all_results),
        "results": all_results,
        "note": "Scores are model outputs, not calibrated confidence.",
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 75)
print(f"Structured results saved to: {json_path}")
print("Note: scores are model outputs, not calibrated confidence.")
print("Each paragraph is still limited to 256 tokens by the model input.")
