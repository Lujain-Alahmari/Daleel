import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from privacy_policy_pipeline import analyze_privacy_policy


def test_opp115_integration():
    text = (
        "We collect names and email addresses to provide services.\n\n"
        "We share information with third-party service providers."
    )

    result = analyze_privacy_policy(text)

    assert result["status"] == "classification_completed"
    assert result["paragraph_count"] == 2
    assert len(result["classifications"]) == 2

    for item in result["classifications"]:
        assert "predicted_categories" in item
        assert "predictions" in item

    print("OPP-115 INTEGRATION: PASS")


if __name__ == "__main__":
    test_opp115_integration()
