import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("models/opp115_distilbert_other_f2_experiment")
results = json.loads(
    (MODEL_DIR / "results.json").read_text(encoding="utf-8")
)
thresholds = results["thresholds"]

examples = [
    (
        "Data Retention",
        "We retain personal information only for as long as necessary "
        "to fulfill the purposes described in this policy."
    ),
    (
        "Data Security",
        "We use encryption, access controls, and security safeguards "
        "to protect personal information from unauthorized access."
    ),
    (
        "Do Not Track",
        "Our website does not currently respond to Do Not Track signals "
        "sent by web browsers."
    ),
    (
        "First Party Collection/Use",
        "We collect your name and email address when you create an account "
        "and use this information to provide our services."
    ),
    (
        "International and Specific Audiences",
        "Our services are not directed to children under the age of 13."
    ),
    (
        "Other",
        "This policy explains general information about our website "
        "and the services available to visitors."
    ),
    (
        "Policy Change",
        "We may update this privacy policy from time to time and will "
        "publish the updated version on our website."
    ),
    (
        "Third Party Sharing/Collection",
        "We share personal information with third-party service providers "
        "that help us operate our services."
    ),
    (
        "User Access, Edit and Deletion",
        "You may request access to, correction of, or deletion of your "
        "personal information by contacting us."
    ),
    (
        "User Choice/Control",
        "You can opt out of promotional emails at any time by clicking "
        "the unsubscribe link."
    ),
]

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

print(f"Device: {device}")
print("\nCATEGORY EXAMPLE TEST")
print("=" * 75)

correct_threshold_hits = 0

for expected, text in examples:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        scores = torch.sigmoid(model(**inputs).logits)[0].cpu().tolist()

    ranked = []
    for i, score in enumerate(scores):
        label = model.config.id2label.get(
            i, model.config.id2label.get(str(i))
        )
        ranked.append((label, float(score)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    expected_score = next(score for label, score in ranked if label == expected)
    expected_threshold = float(thresholds[expected])
    expected_passes = expected_score >= expected_threshold
    predicted = [
        label for label, score in ranked
        if score >= float(thresholds[label])
    ]

    if expected_passes:
        correct_threshold_hits += 1

    print(f"\nExpected: {expected}")
    print(
        f"Expected-category score: {expected_score:.3f} "
        f"(threshold {expected_threshold:.2f})"
    )
    print("Predicted:", " | ".join(predicted) if predicted else "None")
    print("Top 3:", " ; ".join(
        f"{label}={score:.3f}" for label, score in ranked[:3]
    ))

print("\n" + "=" * 75)
print(
    f"Expected category passed its threshold in "
    f"{correct_threshold_hits}/{len(examples)} examples."
)
print(
    "Exploratory examples only; this is not a formal accuracy metric "
    "or a replacement for held-out evaluation."
)
