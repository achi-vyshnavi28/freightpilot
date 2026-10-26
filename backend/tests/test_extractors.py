import random

import pytest

from app.agents.extraction import grounded
from app.tools.extractors import classify_by_keywords, parse_amount, parse_date, parse_weight_kg, repair_gstin, rule_extract
from app.validators import is_valid_gstin
from evals.dataset import build_shipments, make_gstin, ocr_noise_gstin


@pytest.mark.parametrize(
    "raw,value",
    [
        ("Rs. 42,000.00", 42000.0),
        ("₹ 1,45,000/-", 145000.0),
        ("INR 7,680.00", 7680.0),
        ("64,0O0.00", 64000.0),
        ("I", None),
        ("", None),
    ],
)
def test_parse_amount(raw, value):
    assert parse_amount(raw) == value


@pytest.mark.parametrize("raw", ["14-10-2026", "14/10/2026", "14.10.2026", "14 Oct 2026", "2026-10-14"])
def test_parse_date_formats(raw):
    assert parse_date(raw) == "2026-10-14"


def test_parse_weight_units():
    assert parse_weight_kg("9.80 MT") == 9800
    assert parse_weight_kg("18,450 kg") == 18450
    assert parse_weight_kg("12 quintals") == 1200


def test_gstin_ocr_repair_recovers_checksum():
    rng = random.Random(5)
    repaired = 0
    for _ in range(100):
        good = make_gstin(rng, "36")
        noisy = ocr_noise_gstin(rng, good)
        if noisy != good:
            assert not is_valid_gstin(noisy)
            repaired += repair_gstin(noisy) == good
    assert repaired >= 60  # every swap in a digit position is reversible


def test_classifier_on_dataset():
    for ship in build_shipments(9):
        for doc in ship.docs:
            assert classify_by_keywords(doc.text)[0] == doc.doc_type


def test_unknown_document():
    assert classify_by_keywords("Dear team, lunch is at 1pm")[0] == "unknown"


def test_rule_extractor_matches_ground_truth():
    for ship in build_shipments(9):
        for doc in ship.docs:
            got = rule_extract(doc.doc_type, doc.text).fields
            for key, expected in doc.expected_fields.items():
                assert got.get(key) == pytest.approx(expected) if isinstance(expected, float) else got.get(key) == expected, (
                    doc.id,
                    key,
                )


def test_grounding_blocks_invented_values():
    text = "Invoice No: AB/12\nTotal: 42,000.00\nGSTIN: 36AABCS1429B1Z5"
    assert grounded(42000.0, text)
    assert grounded("36AABCS1429B1Z5", text)
    assert not grounded(47040.0, text)
    assert not grounded("36AABCS1429B1Z6", text)
