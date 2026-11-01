from opp115_classifier import classify_policy


def analyze_privacy_policy(policy_text: str) -> dict:
    """Classify privacy-policy paragraphs using OPP-115."""
    classification = classify_policy(policy_text)

    return {
        "status": "classification_completed",
        "paragraph_count": classification["paragraph_count"],
        "classifications": classification["results"],
        "gap_analysis": None,
        "note": (
            "OPP-115 identifies privacy-policy categories only. "
            "PDPL compliance gaps require a separate legal analysis."
        ),
    }
