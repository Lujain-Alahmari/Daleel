import json
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "data" / "evaluation"
INPUT = FOLDER / "qa_gold.xlsx"
OUTPUT = FOLDER / "qa_gold.json"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False
    raise ValueError(f"Expected true or false, got: {value!r}")


workbook = load_workbook(INPUT, read_only=True, data_only=True)

try:
    rows = workbook["Questions"].iter_rows(values_only=True)
    headers = next(rows)
    questions = []
    seen = set()

    for row_number, row in enumerate(rows, start=2):
        record = dict(zip(headers, row))
        if not record.get("question_id"):
            continue

        question_id = str(record["question_id"]).strip()
        if question_id in seen:
            raise ValueError(f"Duplicate question: {question_id}")
        seen.add(question_id)

        answerable = parse_bool(record["answerable"])
        gold_articles = []

        for item in str(record.get("gold_articles") or "").split(";"):
            if not item.strip():
                continue
            document_id, number = item.strip().split(":")
            gold_articles.append({
                "document_id": document_id.strip(),
                "article_number": int(number),
            })

        answer = str(record.get("reference_answer") or "").strip()
        question = str(record.get("question") or "").strip()
        language = str(record.get("language") or "").strip().lower()

        if not question or not answer:
            raise ValueError(f"Row {row_number}: missing question or answer")
        if language not in ("ar", "en"):
            raise ValueError(f"Row {row_number}: invalid language")
        if answerable and not gold_articles:
            raise ValueError(f"{question_id}: missing gold articles")
        if not answerable and gold_articles:
            raise ValueError(f"{question_id}: unexpected gold articles")

        questions.append({
            "id": question_id,
            "language": language,
            "question": question,
            "answerable": answerable,
            "gold_articles": gold_articles,
            "reference_answer": answer,
            "scenario_based": parse_bool(record["scenario_based"]),
            "split": record.get("split"),
            "review_status": record.get("review_status"),
            "review_notes": record.get("review_notes"),
        })

finally:
    workbook.close()

if not questions:
    raise ValueError("No questions found; output not changed.")

# الاحتفاظ بنسخة من JSON السابق قبل تحديثه
if OUTPUT.exists():
    backup = OUTPUT.with_name("qa_gold.previous.json")
    backup.write_bytes(OUTPUT.read_bytes())

OUTPUT.write_text(
    json.dumps(questions, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Questions:", len(questions))
print("Answerable:", sum(q["answerable"] for q in questions))
print("Unanswerable:", sum(not q["answerable"] for q in questions))
print("Saved to:", OUTPUT)