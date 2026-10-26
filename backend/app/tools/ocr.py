"""Turn an uploaded file into text. Images need OCR; text PDFs do not."""

from __future__ import annotations

import io
from dataclasses import dataclass

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


@dataclass
class TextResult:
    text: str
    method: str  # plain | pdf_text | tesseract | none
    is_image: bool = False
    media_type: str | None = None


def extract_text(filename: str, data: bytes) -> TextResult:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {".txt", ".md", ".csv", ""}:
        return TextResult(data.decode("utf-8", errors="replace"), "plain")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        # A scanned PDF has no text layer; the vision LLM tier handles it.
        return TextResult(text, "pdf_text" if text else "none", is_image=not text, media_type="application/pdf")
    if suffix in IMAGE_TYPES:
        return TextResult(_tesseract(data), "tesseract", is_image=True, media_type=IMAGE_TYPES[suffix])
    raise ValueError(f"unsupported file type: {suffix}")


def _tesseract(data: bytes) -> str:
    """Optional local OCR (pip install pytesseract pillow + the tesseract binary)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)), lang="eng")
    except Exception:
        return ""
