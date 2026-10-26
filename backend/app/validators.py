"""Deterministic business-rule checks for Indian logistics documents.

These are the agent's ground truth: an LLM can misread a digit, but a GSTIN
checksum cannot lie. Every extraction is run through here before it is trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
EWAY_BILL_RE = re.compile(r"^[0-9]{12}$")
# Standard registration (TS09AB1234, KA01A1234, DL1C1234) and Bharat series (22BH1234AA).
VEHICLE_RE = re.compile(r"^(?:[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}|[0-9]{2}BH[0-9]{4}[A-Z]{1,2})$")

# Rule 138(10), CGST Rules: one day of validity per 200 km for regular cargo.
EWAY_KM_PER_DAY = 200


@dataclass
class Issue:
    field: str
    code: str
    message: str
    severity: str = "error"  # "error" blocks auto-approval, "warning" does not

    def to_dict(self) -> dict:
        return {"field": self.field, "code": self.code, "message": self.message, "severity": self.severity}


def gstin_check_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        product = GSTIN_CHARS.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return GSTIN_CHARS[(36 - total % 36) % 36]


def is_valid_gstin(gstin: str | None) -> bool:
    if not gstin:
        return False
    gstin = gstin.strip().upper()
    if not GSTIN_RE.match(gstin):
        return False
    if not 1 <= int(gstin[:2]) <= 38:
        return False
    return gstin_check_char(gstin[:14]) == gstin[14]


def gstin_state_code(gstin: str) -> str:
    return gstin[:2]


def normalize_vehicle(raw: str | None) -> str | None:
    if not raw:
        return None
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def is_valid_vehicle(raw: str | None) -> bool:
    plate = normalize_vehicle(raw)
    return bool(plate and VEHICLE_RE.match(plate))


def is_valid_eway_bill(number: str | None) -> bool:
    return bool(number and EWAY_BILL_RE.match(number.replace(" ", "")))


def eway_bill_validity_days(distance_km: float) -> int:
    return max(1, -(-int(distance_km) // EWAY_KM_PER_DAY))


def _money_close(a: float, b: float, tol: float = 1.0) -> bool:
    return abs(a - b) <= tol


def validate_invoice(fields: dict) -> list[Issue]:
    issues: list[Issue] = []
    seller, buyer = fields.get("seller_gstin"), fields.get("buyer_gstin")
    for name, value in (("seller_gstin", seller), ("buyer_gstin", buyer)):
        if not value:
            issues.append(Issue(name, "missing", f"{name} is missing"))
        elif not is_valid_gstin(value):
            issues.append(Issue(name, "gstin_checksum", f"{value} fails GSTIN format/checksum"))

    taxable = fields.get("taxable_value")
    cgst, sgst, igst = (fields.get(k) or 0.0 for k in ("cgst", "sgst", "igst"))
    total = fields.get("total_amount")
    if taxable is None or total is None:
        issues.append(Issue("total_amount", "missing", "taxable value or total is missing"))
    elif not _money_close(taxable + cgst + sgst + igst, total):
        issues.append(
            Issue(
                "total_amount",
                "arithmetic",
                f"taxable {taxable:.2f} + taxes {cgst + sgst + igst:.2f} != total {total:.2f}",
            )
        )

    # Place of supply: same state -> CGST+SGST, different state -> IGST.
    if is_valid_gstin(seller) and is_valid_gstin(buyer):
        intra = gstin_state_code(seller) == gstin_state_code(buyer)
        if intra and igst:
            issues.append(Issue("igst", "tax_type", "intra-state supply charged IGST instead of CGST+SGST"))
        if not intra and (cgst or sgst):
            issues.append(Issue("cgst", "tax_type", "inter-state supply charged CGST/SGST instead of IGST"))
    if cgst and sgst and not _money_close(cgst, sgst):
        issues.append(Issue("sgst", "tax_split", "CGST and SGST should be equal", "warning"))

    line_items = fields.get("line_items") or []
    if line_items and taxable is not None:
        line_sum = sum(item.get("amount", 0.0) for item in line_items)
        if not _money_close(line_sum, taxable):
            issues.append(Issue("line_items", "arithmetic", f"line items sum {line_sum:.2f} != taxable {taxable:.2f}"))
    return issues


def validate_eway_bill(fields: dict) -> list[Issue]:
    issues: list[Issue] = []
    if not is_valid_eway_bill(fields.get("eway_bill_no")):
        issues.append(Issue("eway_bill_no", "format", "e-way bill number must be 12 digits"))
    for name in ("from_gstin", "to_gstin"):
        value = fields.get(name)
        # URP = unregistered person, allowed on e-way bills.
        if value and value != "URP" and not is_valid_gstin(value):
            issues.append(Issue(name, "gstin_checksum", f"{value} fails GSTIN format/checksum"))
    if fields.get("vehicle_no") and not is_valid_vehicle(fields["vehicle_no"]):
        issues.append(Issue("vehicle_no", "format", f"{fields['vehicle_no']} is not a valid Indian registration"))

    generated, valid_upto, distance = fields.get("generated_date"), fields.get("valid_upto"), fields.get("distance_km")
    if generated and valid_upto and distance:
        try:
            expected = date.fromisoformat(generated) + timedelta(days=eway_bill_validity_days(distance))
            if date.fromisoformat(valid_upto) > expected:
                issues.append(Issue("valid_upto", "validity", f"validity exceeds {EWAY_KM_PER_DAY} km/day rule", "warning"))
        except ValueError:
            issues.append(Issue("valid_upto", "date_format", "dates must be ISO YYYY-MM-DD"))
    return issues


def validate_lorry_receipt(fields: dict) -> list[Issue]:
    issues: list[Issue] = []
    if not fields.get("lr_number"):
        issues.append(Issue("lr_number", "missing", "LR number is missing"))
    if not is_valid_vehicle(fields.get("vehicle_no")):
        issues.append(Issue("vehicle_no", "format", f"{fields.get('vehicle_no')} is not a valid Indian registration"))
    weight = fields.get("weight_kg")
    if weight is not None and not 0 < weight <= 49_000:
        issues.append(Issue("weight_kg", "range", f"{weight} kg is outside legal truck payload range"))
    if fields.get("freight_amount") is not None and fields["freight_amount"] <= 0:
        issues.append(Issue("freight_amount", "range", "freight must be positive"))
    return issues


VALIDATORS = {
    "gst_invoice": validate_invoice,
    "eway_bill": validate_eway_bill,
    "lorry_receipt": validate_lorry_receipt,
}


def validate(doc_type: str, fields: dict) -> list[Issue]:
    fn = VALIDATORS.get(doc_type)
    return fn(fields) if fn else []
