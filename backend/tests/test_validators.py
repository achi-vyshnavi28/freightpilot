import random

import pytest

from app.validators import (
    eway_bill_validity_days,
    gstin_check_char,
    is_valid_gstin,
    is_valid_vehicle,
    validate_eway_bill,
    validate_invoice,
    validate_lorry_receipt,
)
from evals.dataset import make_gstin


def test_known_real_format_gstin_checksum():
    # Reliance Industries' published GSTIN.
    assert is_valid_gstin("27AAACR5055K1Z7")
    assert not is_valid_gstin("27AAACR5055K1Z8")


def test_generated_gstins_round_trip():
    rng = random.Random(1)
    for _ in range(200):
        g = make_gstin(rng, "36")
        assert is_valid_gstin(g)
        assert gstin_check_char(g[:14]) == g[14]


@pytest.mark.parametrize("bad", ["", "36ABCDE1234F1Z", "99ABCDE1234F1Z5", "36abcde1234f1z5x"])
def test_malformed_gstin(bad):
    assert not is_valid_gstin(bad)


@pytest.mark.parametrize(
    "plate,ok",
    [
        ("TS09UB4521", True),
        ("ts 09 ub 4521", True),
        ("KA01A1234", True),
        ("22BH1234AA", True),
        ("TS09UB452", False),
        ("1234TS09", False),
    ],
)
def test_vehicle_numbers(plate, ok):
    assert is_valid_vehicle(plate) is ok


def test_eway_validity_200km_per_day():
    assert eway_bill_validity_days(150) == 1
    assert eway_bill_validity_days(200) == 1
    assert eway_bill_validity_days(201) == 2
    assert eway_bill_validity_days(630) == 4


def _invoice(**over):
    rng = random.Random(3)
    base = {
        "seller_gstin": make_gstin(rng, "36"),
        "buyer_gstin": make_gstin(rng, "33"),
        "taxable_value": 42000.0,
        "cgst": 0.0,
        "sgst": 0.0,
        "igst": 5040.0,
        "total_amount": 47040.0,
        "line_items": [{"description": "Freight", "amount": 42000.0}],
    }
    return {**base, **over}


def test_clean_invoice_has_no_issues():
    assert validate_invoice(_invoice()) == []


def test_invoice_arithmetic_error():
    codes = {i.code for i in validate_invoice(_invoice(total_amount=48040.0))}
    assert "arithmetic" in codes


def test_interstate_supply_must_use_igst():
    codes = {i.code for i in validate_invoice(_invoice(igst=0.0, cgst=2520.0, sgst=2520.0))}
    assert "tax_type" in codes


def test_eway_bill_rules():
    issues = validate_eway_bill(
        {
            "eway_bill_no": "12345",
            "vehicle_no": "XX",
            "generated_date": "2026-10-01",
            "valid_upto": "2026-10-09",
            "distance_km": 630,
        }
    )
    assert {i.code for i in issues} == {"format", "validity"}


def test_lorry_receipt_weight_range():
    issues = validate_lorry_receipt({"lr_number": "LR1", "vehicle_no": "TS09UB4521", "weight_kg": 80000, "freight_amount": 1})
    assert [i.code for i in issues] == ["range"]
