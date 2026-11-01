"""Preliminary OPP-115 to PDPL requirement suggestions.

Suggestions are not evidence of legal compliance.
"""

import json
import re
from pathlib import Path

CATEGORY_TO_REQUIREMENTS = {
    "First Party Collection/Use": ["N02", "N03", "N04", "N06", "N07", "N08", "N09", "N15"],
    "Data Retention": ["N12"],
    "User Access, Edit and Deletion": ["N05"],
    "Third Party Sharing/Collection": ["N10", "N11"],
    "User Choice/Control": ["N05", "N13"],
    "Data Security": ["N04", "N17"],
    "International and Specific Audiences": ["N11", "N18"],
    "Policy Change": ["N16"],
    "Do Not Track": [],
    "Other": [],
}


PATTERNS = {
    "N02": [
        r"\bcollect\b.*\b(to provide|to improve|for|purpose|services?)\b",
        r"نجمع.*(لغرض|بهدف|لتقديم|من أجل)",
    ],
    "N03": [
        r"\b(names?|emails?|phones?|addresses|date of birth)\b",
        r"(الاسم|البريد الإلكتروني|رقم الهاتف|العنوان)",
    ],
    "N04": [
        r"\b(via|through|using)\b.*\b(form|website|app|survey)\b",
        r"(نجمع|جمع).*?(عبر|من خلال|بواسطة)",
        r"\b(encrypt|delete|erase|store securely)\b",
    ],
    "N05": [
        r"\b(request access|request correction|request deletion|"
        r"right to access|right to correct|right to delete)\b",
        r"(حق الوصول|طلب الوصول|طلب التصحيح|طلب الحذف|ممارسة حقوق)",
    ],
    "N10": [
        r"\b(share|disclose|transfer)\b.*"
        r"\b(service providers?|third.party|partners?|recipients?|processors?)\b",
        r"(نشارك|نفصح).*?(مزود|طرف ثالث|جهة|معالج)",
    ],
    "N12": [
        r"\b(retain|retention|keep|stored for)\b",
        r"(نحتفظ|الاحتفاظ|مدة الحفظ|فترة الاحتفاظ)",
    ],
}


def suggest_requirements(classification, checklist):
    valid_ids = {item["id"] for item in checklist.get("items", [])}
    grouped = {}

    for paragraph in classification.get("classifications", []):
        text = str(paragraph.get("text", ""))

        for category in paragraph.get("predicted_categories", []):
            for rid in CATEGORY_TO_REQUIREMENTS.get(category, []):
                if rid not in valid_ids:
                    continue

                relevant = any(
                    re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    for pattern in PATTERNS.get(rid, [r"\S"])
                )

                if relevant:
                    match = {
                        "paragraph_number": paragraph.get("paragraph_number"),
                        "category": category,
                        "paragraph_text": text,
                    }
                    matches = grouped.setdefault(rid, [])
                    if match not in matches:
                        matches.append(match)

    return [
        {
            "requirement_id": rid,
            "matches": grouped[rid],
            "note": "Candidate evidence only; legal review required.",
        }
        for rid in sorted(grouped)
    ]


def analyze_integrated_policy(policy_text, checklist_path):
    from privacy_policy_pipeline import analyze_privacy_policy

    classification = analyze_privacy_policy(policy_text)
    checklist = json.loads(
        Path(checklist_path).read_text(encoding="utf-8-sig")
    )

    return {
        "classification": classification,
        "suggested_requirements": suggest_requirements(
            classification, checklist
        ),
        "gap_assessment": None,
        "legal_assessment_performed": False,
    }
