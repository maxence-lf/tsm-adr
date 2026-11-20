import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_report_numbers import (DOCS, check_hygiene, check_links, check_numbers, normalise, numbers,
                                  slug, strip_markup)


def test_normalise_and_extract():
    assert normalise("-0.25") == "0.25"
    assert normalise("4,107") == "4107"
    assert normalise("19.2%") == "19.2"
    found = [n for n, _ in numbers("A Sharpe of 0.05 [-0.19, 0.29] over 2010-2026 (`python3.12`).")]
    assert found == ["0.05", "0.19", "0.29", "2010", "2026"]


def test_markup_is_ignored():
    text = "See [step 6](docs/x.md#6-out-of-sample) and https://example.com/a1 and `v1.2`."
    clean = strip_markup(text)
    assert "x.md" not in clean and "example.com" not in clean and "1.2" not in clean
    assert slug("6. Out-of-sample test (step 6)") == "6-out-of-sample-test-step-6"


@pytest.mark.skipif(not (DOCS / "technical_report.md").exists() or not (DOCS / "report.md").exists(),
                    reason="final reports not written yet")
def test_public_documents_are_consistent():
    assert check_numbers() == []
    assert check_links() == []
    assert check_hygiene() == []
