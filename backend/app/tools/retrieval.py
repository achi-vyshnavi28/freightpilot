"""Hybrid retrieval over client rate cards and contracts.

BM25 (exact terms like city names and tonnage) is fused with TF-IDF cosine
(fuzzier wording) using reciprocal rank fusion. Results carry chunk ids so the
reconciliation agent can cite the exact clause it relied on.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import DATA_DIR

_TOKEN = re.compile(r"[a-z0-9]+")

# Common spellings that must match each other ("Hyd" on a receipt, "Hyderabad" in a contract).
CITY_ALIASES = {
    "hyd": "hyderabad",
    "secunderabad": "hyderabad",
    "blr": "bangalore",
    "bengaluru": "bangalore",
    "bangaluru": "bangalore",
    "madras": "chennai",
    "maa": "chennai",
    "bombay": "mumbai",
    "bom": "mumbai",
    "vizag": "visakhapatnam",
    "vsp": "visakhapatnam",
    "vijayawada": "vijayawada",
    "bza": "vijayawada",
    "ndl": "delhi",
    "new": "",
    "gurgaon": "gurugram",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        tok = CITY_ALIASES.get(tok, tok)
        if tok:
            tokens.append(tok)
    return tokens


def canonical_city(name: str | None) -> str:
    toks = tokenize(name or "")
    return " ".join(toks)


@dataclass
class Chunk:
    id: str
    source: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {"id": self.chunk.id, "source": self.chunk.source, "text": self.chunk.text, "score": round(self.score, 4)}


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """One chunk per table row or paragraph, each prefixed with its section heading for context."""
    chunks: list[Chunk] = []
    heading = ""
    header_row = ""
    para: list[str] = []

    def flush() -> None:
        if para:
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, f"{heading}\n{' '.join(para)}".strip()))
            para.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("# ")
            header_row = ""
        elif stripped.startswith("|"):
            flush()
            if set(stripped) <= set("|-: "):
                continue
            if not header_row:
                header_row = stripped
                continue
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, f"{heading}\n{header_row}\n{stripped}"))
        elif not stripped:
            flush()
        else:
            para.append(stripped)
    flush()
    return chunks


class HybridIndex:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self._tfidf = None
        if chunks:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2))
            self._tfidf = self._vectorizer.fit_transform([c.text for c in chunks])

    def _bm25(self, query: list[str]) -> list[float]:
        scores = []
        for tf, doc in zip(self.tf, self.docs, strict=True):
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 5, source_prefix: str | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        q = tokenize(query)
        bm25 = self._bm25(q)
        dense = (self._tfidf @ self._vectorizer.transform([query]).T).toarray().ravel()
        allowed = [i for i, c in enumerate(self.chunks) if not source_prefix or c.source.startswith(source_prefix)]
        bm_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -bm25[i]))}
        dn_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -dense[i]))}
        # Reciprocal rank fusion, k=60 as in the original RRF paper.
        fused = {i: 1 / (60 + bm_rank[i]) + 1 / (60 + dn_rank[i]) for i in allowed if bm25[i] > 0 or dense[i] > 0}
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], fused[i]) for i in best]


def load_corpus(directory: Path | None = None) -> list[Chunk]:
    directory = directory or DATA_DIR / "rate_cards"
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    return HybridIndex(load_corpus())


_ROW = re.compile(r"\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|")


def lookup_lane_rate(client_id: str, origin: str, destination: str, weight_kg: float | None) -> dict | None:
    """Retrieve, then parse, the contracted rate for a lane. Returns the cited chunk."""
    want_o, want_d = canonical_city(origin), canonical_city(destination)
    tonnes = (weight_kg or 0) / 1000
    hits = get_index().search(f"{origin} {destination} rate per trip", k=10, source_prefix=client_id)
    candidates = []
    for hit in hits:
        for m in _ROW.finditer(hit.chunk.text):
            o, d, cap, rate = (g.strip() for g in m.groups())
            if canonical_city(o) != want_o or canonical_city(d) != want_d:
                continue
            cap_t = float(re.sub(r"[^0-9.]", "", cap) or 0)
            amount = float(re.sub(r"[^0-9.]", "", rate) or 0)
            candidates.append((cap_t, amount, hit))
    if not candidates:
        return None
    # Smallest vehicle class that can carry the load; the biggest one if weight is unknown.
    fitting = sorted((c for c in candidates if c[0] >= tonnes), key=lambda c: (c[0], c[1])) if tonnes else []
    cap_t, amount, hit = fitting[0] if fitting else max(candidates, key=lambda c: c[0])
    return {"capacity_t": cap_t, "rate": amount, "citation": hit.to_dict()}
