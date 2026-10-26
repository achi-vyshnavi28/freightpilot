"""Three extraction tiers the extraction agent escalates through, cheapest first.

1. rules  - regex + OCR repair, ~0 cost, strong on typed documents
2. ml     - fine-tuned Donut model (training/finetune_donut.py), optional
3. llm    - Claude structured outputs, text or vision, highest cost
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from ..schemas import FIELD_MODELS, strict_json_schema
from ..validators import GSTIN_RE, is_valid_gstin, is_valid_vehicle, normalize_vehicle

# ---------------------------------------------------------------- classification

_DOC_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "eway_bill": [
        (r"e-?\s?way\s*bill\b(?!\s*no)", 3),
        (r"\bvalid\s*(?:upto|until|till)\b", 3),
        (r"\bpart[- ]?b\b", 2),
        (r"\bgenerated\s*(?:by|on|date)\b", 2),
        (r"\bapprox(?:imate)?\.?\s*distance\b", 2),
    ],
    "gst_invoice": [
        (r"\btax\s*invoice\b", 4),
        (r"\btaxable\s*(?:value|amount)\b", 3),
        (r"\b[cis]gst\b", 1),
        (r"\bhsn\b|\bsac\b", 2),
        (r"\binvoice\s*(?:no|number)\b", 1),
    ],
    "lorry_receipt": [
        (r"\blorry\s*receipt\b", 4),
        (r"\bconsignment\s*note\b", 4),
        (r"\bconsignor\b", 2),
        (r"\b(?:l\.?\s?r\.?|g\.?\s?c\.?)\s*no\b", 3),
        (r"\bbilty\b", 3),
        (r"\bconsignee\b", 1),
    ],
    "proof_of_delivery": [
        (r"\bproof\s*of\s*delivery\b", 5),
        (r"\bp\.?o\.?d\b", 2),
        (r"\breceived\s*in\s*good\s*condition\b", 3),
        (r"\breceiver'?s?\s*signature\b", 2),
    ],
}


def classify_by_keywords(text: str) -> tuple[str, float, dict[str, float]]:
    lower = text.lower()
    scores = {doc: sum(w * len(re.findall(p, lower)) for p, w in pats) for doc, pats in _DOC_KEYWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < 3:
        return "unknown", 0.3, scores
    margin = (top - second) / top
    return best, round(min(0.99, 0.5 + margin / 2), 2), scores


# ---------------------------------------------------------------- normalisation helpers

_TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"})
# GSTIN layout: 2 digits, 5 letters, 4 digits, 1 letter, 1 alnum, 'Z', 1 alnum
_GSTIN_SHAPE = "DDAAAAADDDDAXZX"


def repair_gstin(raw: str) -> str:
    """Fix common OCR confusions position by position (O<->0, S<->5, B<->8 ...)."""
    raw = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(raw) != 15:
        return raw
    out = []
    for ch, kind in zip(raw, _GSTIN_SHAPE, strict=True):
        if kind == "D":
            out.append(ch.translate(_TO_DIGIT))
        elif kind == "A":
            out.append(ch.translate(_TO_ALPHA))
        elif kind == "Z":
            out.append("Z" if ch in "Z2" else ch)
        else:
            out.append(ch)
    return "".join(out)


_CITY_CANON = {
    "hyd": "Hyderabad",
    "secunderabad": "Hyderabad",
    "bengaluru": "Bangalore",
    "blr": "Bangalore",
    "madras": "Chennai",
    "bombay": "Mumbai",
    "vizag": "Visakhapatnam",
    "new delhi": "Delhi",
    "gurgaon": "Gurugram",
}


def normalize_city(raw: str | None) -> str | None:
    if not raw:
        return None
    name = re.sub(r"\s+", " ", raw).strip(" .,").lower()
    return _CITY_CANON.get(name, name.title())


def parse_amount(raw: str | None) -> float | None:
    if not raw:
        return None
    cleaned = raw.upper().replace("RS.", " ").replace("INR", " ").replace("₹", " ").replace("/-", " ")
    # Look-alike repair (O->0) only inside a numeric token that already has real digits,
    # so the letters of "quintals" or "Rs" never turn into numbers.
    for token in re.findall(r"[0-9ODQILSBZG][0-9ODQILSBZG,.]*", cleaned):
        if re.search(r"[0-9]", token):
            m = re.search(r"\d+(?:\.\d+)?", token.translate(_TO_DIGIT).replace(",", ""))
            if m:
                return float(m.group(0))
    return None


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", raw)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.search(r"(\d{1,2})[-/. ](\d{1,2}|[A-Za-z]{3,9})[-/. ,]+(\d{2,4})", raw)
        if not m:
            return None
        d, mo_raw, y = int(m.group(1)), m.group(2), int(m.group(3))
        mo = int(mo_raw) if mo_raw.isdigit() else _MONTHS.get(mo_raw[:3].lower(), 0)
        if y < 100:
            y += 2000
    try:
        return datetime(y, mo, d).date().isoformat()
    except ValueError:
        return None


def parse_weight_kg(raw: str | None) -> float | None:
    if not raw:
        return None
    value = parse_amount(raw)
    if value is None:
        return None
    lower = raw.lower()
    if "quintal" in lower or re.search(r"\bqtl\b", lower):
        return value * 100
    if re.search(r"\b(?:mt|tonnes?|tons?|t)\b", lower):
        return value * 1000
    return value


# ---------------------------------------------------------------- rule extractor

_V = r"[ \t]*[:\-#.]?[ \t]*"  # separator between a label and its value

_LABELS: dict[str, str] = {
    "invoice_no": r"(?:invoice|inv)\.?\s*(?:no|number|#)\.?",
    "invoice_date": r"(?:invoice\s*date|inv\.?\s*date|dated|date)",
    "eway_bill_no": r"(?:e-?\s?way\s*bill|ewb)\s*(?:no|number)?\.?",
    "vehicle_no": r"(?:vehicle|truck|lorry|veh)\.?\s*(?:no|number|reg(?:istration)?)?\.?",
    "origin": r"(?:from|origin|place\s*of\s*dispatch|dispatch\s*from|source)",
    "destination": r"(?:destination|place\s*of\s*delivery|deliver\s*to)",
    "weight": r"(?:gross\s*)?(?:weight|wt|net\s*wt|quantity)\.?",
    "taxable_value": r"(?:taxable\s*(?:value|amount)|sub\s*-?\s*total)",
    "cgst": r"cgst(?:\s*@\s*[\d.]+\s*%)?",
    "sgst": r"(?:sgst|utgst)(?:\s*@\s*[\d.]+\s*%)?",
    "igst": r"igst(?:\s*@\s*[\d.]+\s*%)?",
    "total_amount": r"(?:grand\s*total|invoice\s*total|total\s*amount|amount\s*payable|(?<!sub )(?<!sub)(?<!sub-)total(?!\s*freight))",
    "lr_number": r"(?:l\.?\s?r\.?|g\.?\s?c\.?|consignment\s*note|bilty|lorry\s*receipt)\s*(?:no|number)\.?",
    "lr_date": r"(?:lr\s*date|date)",
    "consignor": r"consignor(?:'s)?(?:\s*name)?",
    "consignee": r"consignee(?:'s)?(?:\s*name)?",
    "freight_amount": r"(?:total\s*freight|freight\s*(?:amount|charges)?)",
    "generated_date": r"(?:generated\s*(?:date|on)|ewb\s*date|date\s*of\s*generation)",
    "valid_upto": r"valid\s*(?:upto|until|till)",
    "distance_km": r"(?:approx(?:imate)?\.?\s*)?distance",
    "goods_value": r"(?:value\s*of\s*goods|total\s*value|goods\s*value|assessable\s*value)",
}

_VALUE_STOP = r"(?=\s{2,}|\t|\s+(?:to|date|gstin|vehicle)\s*[:\-]|$)"


def _find(label_key: str, text: str, value: str = r"[^\n]+?") -> str | None:
    m = re.search(rf"(?im)(?<![a-z]){_LABELS[label_key]}{_V}({value}){_VALUE_STOP}", text)
    return m.group(1).strip() if m else None


_GSTIN_CANDIDATE = re.compile(r"\b[0-9OIL]{2}[A-Z0-9]{5}[0-9OISBZ]{4}[A-Z0-9]{2}[Z2][A-Z0-9]\b")
_BUYER_CONTEXT = re.compile(r"(?i)bill(?:ed)?\s*to|buyer|recipient|consignee|ship\s*to|customer|\bto\s*gstin|of\s*recipient")
_SELLER_CONTEXT = re.compile(r"(?i)supplier|seller|consignor|from\s*gstin|of\s*supplier")


@dataclass
class FieldHit:
    value: object
    confidence: float


@dataclass
class RuleExtraction:
    fields: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)

    def put(self, name: str, hit: FieldHit | None) -> None:
        if hit is None or hit.value in (None, ""):
            self.fields.setdefault(name, None)
            self.confidence.setdefault(name, 0.0)
        else:
            self.fields[name] = hit.value
            self.confidence[name] = hit.confidence


def _gstins(text: str) -> list[tuple[str, float, str]]:
    """All GSTINs found, as (value, confidence, role hint)."""
    found = []
    lines = text.upper().splitlines()
    for i, line in enumerate(lines):
        for m in _GSTIN_CANDIDATE.finditer(line):
            raw = m.group(0)
            context = line[: m.start()] + " " + (lines[i - 1] if i else "")
            if is_valid_gstin(raw):
                value, conf = raw, 0.97
            else:
                repaired = repair_gstin(raw)
                if not is_valid_gstin(repaired):
                    # Keep the raw value so the validator reports the bad checksum.
                    value, conf = repaired if GSTIN_RE.match(repaired) else raw, 0.4
                else:
                    value, conf = repaired, 0.85
            same_line_context = line[: m.start()]
            if _BUYER_CONTEXT.search(same_line_context):
                role = "buyer"
            elif _SELLER_CONTEXT.search(same_line_context):
                role = "seller"
            elif _BUYER_CONTEXT.search(context):
                role = "buyer"
            elif _SELLER_CONTEXT.search(context):
                role = "seller"
            else:
                role = ""
            found.append((value, conf, role))
    return found


def _assign_gstins(text: str) -> tuple[FieldHit | None, FieldHit | None]:
    found = _gstins(text)
    seller = next((FieldHit(v, c) for v, c, r in found if r == "seller"), None)
    buyer = next((FieldHit(v, c) for v, c, r in found if r == "buyer"), None)
    unlabelled = [(v, c) for v, c, r in found if not r]
    if seller is None and unlabelled:
        v, c = unlabelled.pop(0)
        seller = FieldHit(v, min(c, 0.8))
    if buyer is None and unlabelled:
        v, c = unlabelled.pop(0)
        buyer = FieldHit(v, min(c, 0.7))
    return seller, buyer


def _text_hit(label: str, text: str, conf: float = 0.9) -> FieldHit | None:
    value = _find(label, text)
    return FieldHit(value, conf) if value else None


def _amount_hit(label: str, text: str) -> FieldHit | None:
    value = parse_amount(_find(label, text, r"(?:rs\.?|inr|₹)?\s*[0-9OSIl,]+(?:\.\d{1,2})?(?:/-)?"))
    return FieldHit(value, 0.92) if value is not None else None


def _date_hit(label: str, text: str) -> FieldHit | None:
    value = parse_date(_find(label, text, r"[0-9A-Za-z\-/. ,]{6,20}?"))
    return FieldHit(value, 0.9) if value else None


def _vehicle_hit(text: str) -> FieldHit | None:
    raw = _find("vehicle_no", text, r"[A-Z0-9][A-Z0-9 \-]{5,14}[0-9A-Z]")
    if not raw:
        return None
    plate = normalize_vehicle(raw)
    if is_valid_vehicle(plate):
        return FieldHit(plate, 0.93)
    # OCR repair: last four characters of a plate are always digits.
    fixed = plate[:-4] + plate[-4:].translate(_TO_DIGIT) if plate and len(plate) > 4 else plate
    if fixed and is_valid_vehicle(fixed):
        return FieldHit(fixed, 0.8)
    return FieldHit(plate, 0.45)


def _eway_hit(text: str) -> FieldHit | None:
    raw = _find("eway_bill_no", text, r"[0-9OIl][0-9OIl \-]{10,16}[0-9OIl]")
    if not raw:
        return None
    digits = re.sub(r"[^0-9]", "", raw.upper().translate(_TO_DIGIT))
    return FieldHit(digits, 0.93 if len(digits) == 12 else 0.4)


def _route(text: str) -> tuple[FieldHit | None, FieldHit | None]:
    m = re.search(r"(?im)^\s*from\s*[:\-]?\s*([A-Za-z .]+?)\s+to\s*[:\-]?\s*([A-Za-z .]+?)\s*$", text)
    if m:
        return FieldHit(normalize_city(m.group(1)), 0.9), FieldHit(normalize_city(m.group(2)), 0.9)
    origin = _find("origin", text, r"[A-Za-z][A-Za-z .]+")
    dest = _find("destination", text, r"[A-Za-z][A-Za-z .]+")
    return (FieldHit(normalize_city(origin), 0.8) if origin else None, FieldHit(normalize_city(dest), 0.8) if dest else None)


_LINE_ITEM = re.compile(r"(?im)^\s*([A-Za-z][A-Za-z &/().,-]{2,40}?)\s{2,}(?:(\d{4,8})\s{2,})?((?:rs\.?\s*)?[\d,]+\.\d{2})\s*$")


def _line_items(text: str) -> list[dict]:
    items = []
    for m in _LINE_ITEM.finditer(text):
        desc = m.group(1).strip()
        if re.search(r"(?i)total|taxable|[cis]gst|freight\s*amount|value", desc):
            continue
        items.append({"description": desc, "hsn": m.group(2), "amount": parse_amount(m.group(3))})
    return items


def rule_extract(doc_type: str, text: str) -> RuleExtraction:
    out = RuleExtraction()
    origin, dest = _route(text)
    weight_raw = _find("weight", text, r"[0-9.,]+\s*(?:kg|kgs|mt|tonnes?|tons?|t|quintals?|qtl)?")
    weight = parse_weight_kg(weight_raw)

    if doc_type == "gst_invoice":
        seller, buyer = _assign_gstins(text)
        out.put("invoice_no", _text_hit("invoice_no", text, 0.92))
        out.put("invoice_date", _date_hit("invoice_date", text))
        out.put("seller_gstin", seller)
        out.put("buyer_gstin", buyer)
        out.put("eway_bill_no", _eway_hit(text))
        out.put("vehicle_no", _vehicle_hit(text))
        out.put("origin", origin)
        out.put("destination", dest)
        out.put("weight_kg", FieldHit(weight, 0.85) if weight else None)
        items = _line_items(text)
        out.fields["line_items"] = items
        out.confidence["line_items"] = 0.85 if items else 0.0
        for key in ("taxable_value", "cgst", "sgst", "igst"):
            out.put(key, _amount_hit(key, text))
        out.put("total_amount", _amount_hit("total_amount", text))
        # A tax head that is absent is zero, not unknown, once a total exists.
        if out.fields.get("total_amount") is not None:
            for key in ("cgst", "sgst", "igst"):
                if out.fields.get(key) is None:
                    out.fields[key], out.confidence[key] = 0.0, 0.8
    elif doc_type == "eway_bill":
        found = _gstins(text)
        from_g = next((FieldHit(v, c) for v, c, r in found if r == "seller"), None)
        to_g = next((FieldHit(v, c) for v, c, r in found if r == "buyer"), None)
        if re.search(r"(?i)gstin\s*of\s*recipient\s*[:\-]?\s*URP", text):
            to_g = FieldHit("URP", 0.95)
        out.put("eway_bill_no", _eway_hit(text))
        out.put("generated_date", _date_hit("generated_date", text))
        out.put("valid_upto", _date_hit("valid_upto", text))
        out.put("from_gstin", from_g)
        out.put("to_gstin", to_g)
        out.put("origin", origin)
        out.put("destination", dest)
        dist = parse_amount(_find("distance_km", text, r"[0-9OSIl,.]+\s*(?:km|kms)?"))
        out.put("distance_km", FieldHit(dist, 0.9) if dist else None)
        out.put("vehicle_no", _vehicle_hit(text))
        out.put("invoice_no", _text_hit("invoice_no", text, 0.85))
        out.put("goods_value", _amount_hit("goods_value", text))
    elif doc_type == "lorry_receipt":
        out.put("lr_number", _text_hit("lr_number", text, 0.92))
        out.put("lr_date", _date_hit("lr_date", text))
        out.put("consignor", _text_hit("consignor", text, 0.85))
        out.put("consignee", _text_hit("consignee", text, 0.85))
        out.put("origin", origin)
        out.put("destination", dest)
        out.put("vehicle_no", _vehicle_hit(text))
        out.put("weight_kg", FieldHit(weight, 0.88) if weight else None)
        out.put("freight_amount", _amount_hit("freight_amount", text))
        out.put("invoice_no", _text_hit("invoice_no", text, 0.85))
    return out


# ---------------------------------------------------------------- ML tier (optional)


def ml_available() -> bool:
    from ..config import settings

    if not settings.donut_model_path:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def ml_extract(doc_type: str, image_bytes: bytes) -> dict:
    """Run a Donut checkpoint produced by training/finetune_donut.py."""
    import io

    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    from ..config import settings

    processor = DonutProcessor.from_pretrained(settings.donut_model_path)
    model = VisionEncoderDecoderModel.from_pretrained(settings.donut_model_path).eval()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    prompt_ids = processor.tokenizer(f"<s_{doc_type}>", add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(pixel_values, decoder_input_ids=prompt_ids, max_length=512)
    sequence = processor.batch_decode(output)[0]
    parsed = processor.token2json(sequence)
    # Donut emits strings; the schema coerces "42000.00" -> 42000.0 and drops unknown keys.
    return (
        FIELD_MODELS[doc_type]
        .model_validate({k: v for k, v in parsed.items() if k in FIELD_MODELS[doc_type].model_fields})
        .model_dump()
    )


# ---------------------------------------------------------------- LLM tier

EXTRACTION_SYSTEM = (
    "You extract fields from Indian logistics documents (GST invoices, e-way bills, lorry receipts). "
    "Return dates as YYYY-MM-DD, amounts as plain numbers in rupees, weights in kg, GSTINs and vehicle "
    "numbers uppercase without spaces. Use null for anything not present; never invent values."
)


def llm_schema(doc_type: str) -> dict:
    return strict_json_schema(FIELD_MODELS[doc_type])
