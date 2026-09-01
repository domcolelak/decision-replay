"""Embedding provider abstraction.

Two things this module has to get right:

* the model and its version are persisted with every vector, because comparing
  vectors from two different models produces confident nonsense;
* an offline provider exists, is deterministic, and needs no network -- so the
  test suite, CI and a local demo all work with no credentials.

The offline provider is a hashed bag-of-words projection. It is not a language
model and does not pretend to be: it captures lexical overlap, which is enough
to exercise the pipeline honestly and to show that the ranking still works when
the semantic component is weak. What it must never do is silently look like a
real embedding, so it identifies itself as ``offline-hashing`` in the stored
model name.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from app.core.config import settings

#: Dimensionality of the offline provider. Small enough to store as JSON.
OFFLINE_DIMENSIONS = 256

_TOKEN = re.compile(r"[a-z0-9]+")

#: Very common words carry no discriminative signal.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the to
    was were will with we our they their this these those but not
    """.split()
)


@dataclass
class EmbeddingResult:
    vectors: list[list[float]] = field(default_factory=list)
    model: str = ""
    version: str = ""
    dimensions: int = 0
    latency_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.vectors)


class EmbeddingProvider(ABC):
    name = "provider"
    version = "v1"
    dimensions = OFFLINE_DIMENSIONS

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a batch. Must never raise; failures come back on the result."""


class OfflineHashingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-words. No network, no credentials."""

    name = "offline-hashing"
    version = "v1"

    def __init__(self, dimensions: int = OFFLINE_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        started = time.perf_counter()
        vectors = [self._vector(text) for text in texts]
        return EmbeddingResult(
            vectors=vectors,
            model=self.name,
            version=self.version,
            dimensions=self.dimensions,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS]
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # A sign bit from a different part of the digest keeps unrelated
            # tokens from all pushing the vector the same way.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            # Sub-linear term weighting, as in TF-IDF: a word repeated ten
            # times is not ten times as informative.
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm > 0 else vector


class OpenAICompatibleProvider(EmbeddingProvider):
    """Any OpenAI-style ``/embeddings`` endpoint."""

    version = "v1"

    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int = 1536) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = model
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        started = time.perf_counter()
        try:
            import httpx

            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.name, "input": list(texts)},
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
            vectors = [item["embedding"] for item in payload["data"]]
            return EmbeddingResult(
                vectors=vectors,
                model=self.name,
                version=self.version,
                dimensions=len(vectors[0]) if vectors else self.dimensions,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # pragma: no cover - depends on a live endpoint
            return EmbeddingResult(
                model=self.name,
                version=self.version,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc)[:500],
            )


_provider: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    global _provider
    if _provider is not None:
        return _provider

    configured = (getattr(settings, "embedding_provider", "") or "offline").lower()
    if configured in ("openai", "openai_compatible") and getattr(settings, "ai_api_key", ""):
        _provider = OpenAICompatibleProvider(
            base_url=settings.ai_base_url,
            api_key=settings.ai_api_key,
            model=getattr(settings, "embedding_model", "text-embedding-3-small"),
        )
    else:
        _provider = OfflineHashingProvider()
    return _provider


def set_provider(provider: EmbeddingProvider | None) -> None:
    """Override the provider. Used by tests."""
    global _provider
    _provider = provider


def embedding_text(title: str, context_text: str, structured: dict) -> str:
    """The text an embedding is computed over.

    Structured values are included as ``field: value`` lines so that a context
    captured only through form fields still produces a usable vector -- a
    decision recorded without prose would otherwise have no semantic component
    at all.
    """
    parts = [title, context_text]
    for key in sorted(structured):
        value = structured[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        parts.append(f"{key}: {value}")
    return "\n".join(p for p in parts if p)
