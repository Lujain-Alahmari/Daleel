import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    hamming_loss,
    accuracy_score,
    classification_report,
)

# -------------------------
# 1. Configuration
# -------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test_no_train_duplicates.csv")
OUTPUT_DIR = Path("models/opp115_distilbert_other_f2_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "Data Retention",
    "Data Security",
    "Do Not Track",
    "First Party Collection/Use",
    "International and Specific Audiences",
    "Other",
    "Policy Change",
    "Third Party Sharing/Collection",
    "User Access, Edit and Deletion",
    "User Choice/Control",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("OPP-115 DISTILBERT MULTI-LABEL CLASSIFIER")
print("=" * 60)
print("Device:", DEVICE)
print("Model:", MODEL_NAME)

# -------------------------
# 2. Load and parse labels
# -------------------------
def parse_labels(value):
    if isinstance(value, list):
        items = value
    else:
        value = str(value).strip()

        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, (list, tuple, set)):
            items = list(parsed)
        elif value in LABELS:
            items = [value]
        elif "|" in value:
            items = [x.strip() for x in value.split("|") if x.strip()]
        elif ";" in value:
            items = [x.strip() for x in value.split(";") if x.strip()]
        else:
            raise ValueError(
                f"Could not parse labels: {value!r}. "
                "Check the labels column before training."
            )

    items = [str(x).strip() for x in items]
    unknown = set(items) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")

    return sorted(set(items))


train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["text", "labels"]).copy()
test_df = pd.read_csv(TEST_PATH).dropna(subset=["text", "labels"]).copy()

train_df["text"] = train_df["text"].astype(str)
test_df["text"] = test_df["text"].astype(str)
train_df["parsed_labels"] = train_df["labels"].apply(parse_labels)
test_df["parsed_labels"] = test_df["labels"].apply(parse_labels)

mlb = MultiLabelBinarizer(classes=LABELS)
mlb.fit([LABELS])

Y = mlb.transform(train_df["parsed_labels"]).astype(np.float32)
Y_test = mlb.transform(test_df["parsed_labels"]).astype(np.float32)

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Categories:", len(LABELS))

# Split by policy, not by individual segment
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
fit_idx, val_idx = next(
    splitter.split(train_df, groups=train_df["policy_file"])
)

fit_policies = set(train_df.iloc[fit_idx]["policy_file"])
val_policies = set(train_df.iloc[val_idx]["policy_file"])
assert fit_policies.isdisjoint(val_policies)

print("Training policies for validation experiment:", len(fit_policies))
print("Validation policies:", len(val_policies))

# -------------------------
# 3. Tokenizer and dataset
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = None if labels is None else torch.tensor(
            labels, dtype=torch.float32
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    return model.to(DEVICE)


def predict_probabilities(model, loader):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**inputs).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def train_model(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.BCEWithLogitsLoss()
    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 1
    best_val_probs = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            optimizer.zero_grad()

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_probs = predict_probabilities(model, val_loader)
        val_truth = np.concatenate(
            [batch["labels"].numpy() for batch in val_loader], axis=0
        )
        val_pred = (val_probs >= 0.5).astype(int)
        macro_f1 = f1_score(
            val_truth, val_pred, average="macro", zero_division=0
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Validation Macro F1 @0.5: {macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_val_probs = val_probs.copy()

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_probs


# -------------------------
# 4. Validation experiment
# -------------------------
fit_dataset = TextDataset(
    train_df.iloc[fit_idx]["text"].tolist(), Y[fit_idx]
)
val_dataset = TextDataset(
    train_df.iloc[val_idx]["text"].tolist(), Y[val_idx]
)

fit_loader = DataLoader(
    fit_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

model = make_model()
model, best_epoch, val_probs = train_model(
    model, fit_loader, val_loader, EPOCHS
)

val_truth = Y[val_idx]
thresholds = []

# Select each category's threshold using validation data only.
for i, label in enumerate(LABELS):
    if val_truth[:, i].sum() == 0:
        thresholds.append(0.5)
        continue

    candidates = np.arange(0.20, 0.81, 0.05)
    if label == "Other":
        scores = [
            fbeta_score(
                val_truth[:, i],
                (val_probs[:, i] >= threshold).astype(int),
                beta=2,
                zero_division=0,
            )
            for threshold in candidates
        ]
    else:
        scores = [
            f1_score(
                val_truth[:, i],
                (val_probs[:, i] >= threshold).astype(int),
                zero_division=0,
            )
            for threshold in candidates
        ]
    thresholds.append(float(candidates[int(np.argmax(scores))]))

print("\nSelected validation thresholds:")
for label, threshold in zip(LABELS, thresholds):
    print(f"  {label}: {threshold:.2f}")

# -------------------------
# 5. Retrain on all training data
# -------------------------
del model, fit_loader, val_loader, fit_dataset, val_dataset
if DEVICE.type == "mps":
    torch.mps.empty_cache()

full_train_dataset = TextDataset(train_df["text"].tolist(), Y)
test_dataset = TextDataset(test_df["text"].tolist(), Y_test)

full_train_loader = DataLoader(
    full_train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

final_model = make_model()
optimizer = torch.optim.AdamW(
    final_model.parameters(), lr=LEARNING_RATE
)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, best_epoch + 1):
    final_model.train()
    total_loss = 0.0

    for batch in full_train_loader:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        optimizer.zero_grad()
        logits = final_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Final training epoch {epoch}/{best_epoch} | "
        f"Loss: {total_loss / len(full_train_loader):.4f}"
    )

test_probs = predict_probabilities(final_model, test_loader)
test_pred = np.column_stack([
    test_probs[:, i] >= thresholds[i]
    for i in range(len(LABELS))
]).astype(int)

results = {
    "model": MODEL_NAME,
    "device": str(DEVICE),
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "best_epoch": best_epoch,
    "thresholds": dict(zip(LABELS, thresholds)),
    "micro_f1": float(f1_score(Y_test, test_pred, average="micro", zero_division=0)),
    "macro_f1": float(f1_score(Y_test, test_pred, average="macro", zero_division=0)),
    "hamming_loss": float(hamming_loss(Y_test, test_pred)),
    "exact_match_accuracy": float(accuracy_score(Y_test, test_pred)),
    "classification_report": classification_report(
        Y_test, test_pred, target_names=LABELS,
        output_dict=True, zero_division=0
    ),
}

print("\n" + "=" * 60)
print("DISTILBERT — HELD-OUT TEST RESULTS")
print("=" * 60)
print("Micro F1:", round(results["micro_f1"], 4))
print("Macro F1:", round(results["macro_f1"], 4))
print("Hamming Loss:", round(results["hamming_loss"], 4))
print("Exact Match Accuracy:", round(results["exact_match_accuracy"], 4))
print("\nPer-category report:")
print(classification_report(
    Y_test, test_pred, target_names=LABELS, zero_division=0
))

final_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

pd.DataFrame({
    "text": test_df["text"],
    "true_labels": test_df["parsed_labels"].apply(lambda x: " | ".join(x)),
    "predicted_labels": [
        " | ".join(LABELS[j] for j, value in enumerate(row) if value)
        for row in test_pred
    ],
}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\nSaved model and results to:", OUTPUT_DIR)
