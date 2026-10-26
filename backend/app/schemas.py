"""Typed contracts shared by agents, the API and the LLM's structured outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["gst_invoice", "eway_bill", "lorry_receipt", "proof_of_delivery", "unknown"]
DOC_TYPES: tuple[str, ...] = ("gst_invoice", "eway_bill", "lorry_receipt", "proof_of_delivery", "unknown")


class LineItem(BaseModel):
    description: str
    hsn: str | None = None
    amount: float


class InvoiceFields(BaseModel):
    invoice_no: str | None = None
    invoice_date: str | None = Field(None, description="ISO date YYYY-MM-DD")
    seller_gstin: str | None = None
    buyer_gstin: str | None = None
    eway_bill_no: str | None = None
    vehicle_no: str | None = None
    origin: str | None = None
    destination: str | None = None
    weight_kg: float | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    taxable_value: float | None = None
    cgst: float | None = None
    sgst: float | None = None
    igst: float | None = None
    total_amount: float | None = None


class EwayBillFields(BaseModel):
    eway_bill_no: str | None = None
    generated_date: str | None = Field(None, description="ISO date YYYY-MM-DD")
    valid_upto: str | None = Field(None, description="ISO date YYYY-MM-DD")
    from_gstin: str | None = None
    to_gstin: str | None = None
    origin: str | None = None
    destination: str | None = None
    distance_km: float | None = None
    vehicle_no: str | None = None
    invoice_no: str | None = None
    goods_value: float | None = None


class LorryReceiptFields(BaseModel):
    lr_number: str | None = None
    lr_date: str | None = Field(None, description="ISO date YYYY-MM-DD")
    consignor: str | None = None
    consignee: str | None = None
    origin: str | None = None
    destination: str | None = None
    vehicle_no: str | None = None
    weight_kg: float | None = None
    freight_amount: float | None = None
    invoice_no: str | None = None


FIELD_MODELS: dict[str, type[BaseModel]] = {
    "gst_invoice": InvoiceFields,
    "eway_bill": EwayBillFields,
    "lorry_receipt": LorryReceiptFields,
}


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema tightened for structured outputs: every field required, no extras."""
    schema = model.model_json_schema()

    def tighten(node: dict) -> None:
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}).keys())
            for prop in node.get("properties", {}).values():
                prop.pop("default", None)
                tighten(prop)
        for key in ("items", "anyOf"):
            child = node.get(key)
            if isinstance(child, dict):
                tighten(child)
            elif isinstance(child, list):
                for c in child:
                    tighten(c)
        for definition in node.get("$defs", {}).values():
            tighten(definition)

    tighten(schema)
    return schema


class ExtractionResult(BaseModel):
    doc_type: str
    fields: dict
    confidence: dict[str, float] = Field(default_factory=dict)
    method: str
    issues: list[dict] = Field(default_factory=list)
    attempts: list[dict] = Field(default_factory=list)
    needs_review: bool = False


class GPSPoint(BaseModel):
    ts: str = Field(description="ISO timestamp")
    lat: float
    lon: float
    speed_kmph: float = 0.0


class Trip(BaseModel):
    trip_id: str
    client_id: str
    vehicle_no: str
    origin: str
    destination: str
    planned_route: list[tuple[float, float]]
    deadline: str = Field(description="ISO timestamp the consignee expects delivery")
    points: list[GPSPoint]
