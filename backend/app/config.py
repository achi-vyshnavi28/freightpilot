"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'freightpilot.db'}"))
    # "auto" uses Claude when credentials exist, otherwise the deterministic offline engine.
    llm_mode: str = field(default_factory=lambda: os.getenv("FP_LLM_MODE", "auto"))
    model: str = field(default_factory=lambda: os.getenv("FP_MODEL", "claude-opus-5-5"))
    # Cheap model for high-volume routing (doc classification). Override to FP_MODEL to use one model.
    fast_model: str = field(default_factory=lambda: os.getenv("FP_FAST_MODEL", "claude-haiku-5-5"))
    effort: str = field(default_factory=lambda: os.getenv("FP_EFFORT", "medium"))
    max_agent_turns: int = field(default_factory=lambda: int(os.getenv("FP_MAX_AGENT_TURNS", "8")))
    # Fields below this confidence go to a human instead of being trusted.
    review_threshold: float = field(default_factory=lambda: float(os.getenv("FP_REVIEW_THRESHOLD", "0.75")))
    # Freight billed above contract rate by more than this fraction is flagged.
    rate_tolerance: float = field(default_factory=lambda: float(os.getenv("FP_RATE_TOLERANCE", "0.05")))
    redact_pii: bool = field(default_factory=lambda: _env_bool("FP_REDACT_PII", True))
    donut_model_path: str | None = field(default_factory=lambda: os.getenv("FP_DONUT_MODEL"))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.getenv("FP_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
        )
    )


settings = Settings()
