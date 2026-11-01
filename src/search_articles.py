import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "legal_articles.json"


def tokenize(text):
    text = text.lower()
    # إزالة التشكيل والتطويل وتوحيد أشكال الألف للبحث فقط
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670ـ]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


articles = json.loads(INPUT.read_text(encoding="utf-8"))

# فهرس مستقل لكل لغة حتى لا يؤثر اختلاف اللغتين في ترتيب النتائج
indexes = {}

for language in ("ar", "en"):
    records = [a for a in articles if a["language"] == language]
    indexes[language] = (
        records,
        BM25Okapi([tokenize(a["text"]) for a in records]),
    )

while True:
    question = input("\nاكتبي السؤال، أو exit للخروج: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    language = "ar" if re.search(r"[\u0600-\u06ff]", question) else "en"
    records, index = indexes[language]
    scores = index.get_scores(tokenize(question))

    ranked = sorted(
        range(len(records)),
        key=lambda i: scores[i],
        reverse=True,
    )
    matches = [i for i in ranked if scores[i] > 0][:5]

    if not matches:
        print("لم توجد كلمات مطابقة. جربي صياغة أخرى.")
        continue

    for position, i in enumerate(matches, start=1):
        article = records[i]
        print(f"\n{'=' * 50}")
        print(
            f"{position}. المادة {article['article_number']} "
            f"| {article['document_id']} | {article['language']}"
        )
        print(f"درجة البحث: {scores[i]:.2f}")
        print(article["text"])

    print("\nهذه نتائج بحث أولية؛ درجة البحث ليست نسبة ثقة.")