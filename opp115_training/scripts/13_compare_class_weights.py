import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss

DATA_DIR = "data/processed"


def parse_labels(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]


df = pd.read_csv(f"{DATA_DIR}/train.csv")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
fit_idx, val_idx = next(
    splitter.split(df, groups=df["policy_file"])
)

fit_df = df.iloc[fit_idx].copy()
val_df = df.iloc[val_idx].copy()

mlb = MultiLabelBinarizer()
mlb.fit(df["labels"].apply(parse_labels))

y_fit = mlb.transform(fit_df["labels"].apply(parse_labels))
y_val = mlb.transform(val_df["labels"].apply(parse_labels))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)

X_fit = vectorizer.fit_transform(
    fit_df["text"].fillna("").astype(str)
)
X_val = vectorizer.transform(
    val_df["text"].fillna("").astype(str)
)

for weight in [None, "balanced"]:
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight=weight,
        )
    )
    model.fit(X_fit, y_fit)

    probs = model.predict_proba(X_val)
    pred_default = (probs >= 0.5).astype(int)

    # Also assess threshold tuning on validation data only
    pred_tuned = pred_default.copy()
    chosen_thresholds = []

    for i in range(y_val.shape[1]):
        best_f1 = -1
        best_threshold = 0.5

        if y_val[:, i].sum() > 0:
            for threshold in [
                0.10, 0.15, 0.20, 0.25, 0.30,
                0.35, 0.40, 0.45, 0.50, 0.60, 0.70
            ]:
                candidate = (probs[:, i] >= threshold).astype(int)
                score = f1_score(
                    y_val[:, i],
                    candidate,
                    zero_division=0,
                )
                if score > best_f1:
                    best_f1 = score
                    best_threshold = threshold

        chosen_thresholds.append(best_threshold)
        pred_tuned[:, i] = (
            probs[:, i] >= best_threshold
        ).astype(int)

    name = "balanced" if weight else "unweighted"

    print("\n" + "=" * 55)
    print("Experiment:", name)
    print("Validation samples:", len(val_df))
    print("Micro F1 @ 0.50:",
          round(f1_score(y_val, pred_default, average="micro", zero_division=0), 4))
    print("Macro F1 @ 0.50:",
          round(f1_score(y_val, pred_default, average="macro", zero_division=0), 4))
    print("Micro F1 tuned:",
          round(f1_score(y_val, pred_tuned, average="micro", zero_division=0), 4))
    print("Macro F1 tuned:",
          round(f1_score(y_val, pred_tuned, average="macro", zero_division=0), 4))
    print("Hamming Loss tuned:",
          round(hamming_loss(y_val, pred_tuned), 4))
    print("\nPer-category tuned F1:")
    for i, label in enumerate(mlb.classes_):
        score = f1_score(y_val[:, i], pred_tuned[:, i], zero_division=0)
        print(f"{label}: {score:.3f} (threshold={chosen_thresholds[i]:.2f})")

print("\nComparison complete. Test set was not used.")
