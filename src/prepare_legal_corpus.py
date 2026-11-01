"""Read article spreadsheets without depending on column positions.

Run after copying the six workbooks into data/processed:
    python src/prepare_legal_corpus.py
For an explicitly incomplete draft:
    python src/prepare_legal_corpus.py --allow-incomplete
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from openpyxl import load_workbook

SOURCES = [
    ("pdpl", "en", 43, "PDPL_Articles_English*.xlsx", "PDPL Articles", "PersonalDataProtectionLawEN.pdf"),
    ("pdpl", "ar", 43, "نظام_حماية_البيانات_الشخصية.xlsx", "المواد", "PersonalDataAR.pdf"),
    ("implementing_regulation", "en", 38, "Implementing_Regulation_PDPL_Articles_English.xlsx", "Implementing Regulation", "ImplementingRegulationPersonalDataProtectionLaw.pdf"),
    ("implementing_regulation", "ar", 38, "Implementing_Regulation_PDPL_Articles_Arabic.xlsx", None, "ImplementingRegulationPersonalDataProtectionLawAr.pdf"),
    ("transfer_regulation", "en", 9, "Regulation_Personal_Data_Transfer_English.xlsx", "Transfer Regulation", "RegulationonPersonalDataEN.pdf"),
    ("transfer_regulation", "ar", 9, "Data_Transfer_Regulation_Articles AR (1).xlsx", "Articles", "RegulationonPersonalDataAR.pdf"),
]
ALIASES = {
    "number": ["Article No.", "article_number", "رقم المادة"],
    "text": ["Policy Content", "Article Content", "Article", "text", "النص"],
    "title": ["Article Title", "article_title", "المادة"],
    "document": ["Policy Name", "document_name"],
}


def header_key(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def read_articles(path, sheet_name, document_id, language, expected, pdf_name):
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = [workbook[sheet_name]] if sheet_name and sheet_name in workbook.sheetnames else list(workbook)
        selected = None
        for sheet in sheets:
            rows = list(sheet.iter_rows(values_only=True))
            for index, header in enumerate(rows[:10]):
                lookup = {header_key(v): c for c, v in enumerate(header) if v is not None}
                columns = {}
                for field, aliases in ALIASES.items():
                    columns[field] = next((lookup[header_key(a)] for a in aliases if header_key(a) in lookup), None)
                if columns["number"] is not None and columns["text"] is not None:
                    selected = (sheet, rows, index, columns)
                    break
            if selected:
                break
        if not selected:
            raise ValueError(f"{path.name}: no supported article-number/text headers")
        sheet, rows, header_index, columns = selected
        result = []
        for source_row, row in enumerate(rows[header_index + 1:], header_index + 2):
            if all(v is None for v in row):
                continue
            number, text = row[columns["number"]], row[columns["text"]]
            if isinstance(number, str) and number.strip().isdecimal():
                number = int(number.strip())
            if isinstance(number, bool) or not isinstance(number, (int, float)) or int(number) != number:
                raise ValueError(f"{path.name}, row {source_row}: invalid article number")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path.name}, row {source_row}: empty article text")
            def optional(field):
                column = columns[field]
                return str(row[column]).strip() if column is not None and row[column] is not None else None
            number = int(number)
            result.append({
                "id": f"{document_id}-{language}-{number}",
                "document_id": document_id,
                "document_name": optional("document"),
                "article_number": number,
                "article_title": optional("title"),
                "language": language,
                "text": text.strip(),
                "chapter": None,
                "source_file": path.name,
                "source_sheet": sheet.title,
                "source_row": source_row,
                "source_sha256": checksum,
                "source_pdf": pdf_name,
                "source_page_start": None,
                "source_page_end": None,
                "official_source_url": None,
                "source_version": None,
                "verification_status": "pending_pdf_comparison",
            })
        if sorted(a["article_number"] for a in result) != list(range(1, expected + 1)):
            raise ValueError(f"{path.name}: expected articles 1-{expected} exactly once")
        return result
    finally:
        workbook.close()


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    articles, missing = [], []
    for doc, language, expected, pattern, sheet, pdf in SOURCES:
        matches = list(args.input_dir.glob(pattern))
        if not matches:
            missing.append(f"{doc}/{language}: {pattern}")
            continue
        if len(matches) != 1:
            raise ValueError(f"Multiple files match {pattern}; keep one input version")
        records = read_articles(matches[0], sheet, doc, language, expected, pdf)
        articles.extend(records)
        print(f"{doc}/{language}: {len(records)} articles")
    if missing:
        print("Missing inputs:\n" + "\n".join(missing))
        if not args.allow_incomplete:
            raise SystemExit("No output written. Supply missing inputs or use --allow-incomplete for a draft.")
    if not articles:
        raise SystemExit("No articles found; no output written.")
    assert len({a["id"] for a in articles}) == len(articles)
    filename = "legal_articles_draft.json" if missing else "legal_articles.json"
    output = args.output or args.input_dir / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"Total: {len(articles)}; saved to {output}")
    print("Structural checks passed. Text/source verification and missing metadata still require review.")


if __name__ == "__main__":
    main()
