"""Hybrid precedent ranking.

Four components decide how comparable two decisions are:

* **structured** -- field-by-field similarity, weighted by the template;
* **semantic**   -- cosine similarity between context embeddings;
* **type**       -- same decision type / template;
* **recency**    -- how recent the precedent is.

Two properties matter more than the exact weights.

**Every component score is returned.** The ranking is not hidden behind a model:
a user who disagrees with the order can see which part drove it, and change the
template weights rather than argue with a black box.

**A missing component redistributes its weight instead of scoring zero.** If
embeddings have not been generated -- or the provider is not configured at all
-- semantic similarity is unavailable, not zero. Treating it as zero would cap
every result at 55% and quietly reorder the list. Structured search has to keep
working on its own; that is a requirement, not a fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from app.templates.fields import (
    DecisionTemplate,
    StructuredSimilarity,
    coverage,
    structured_similarity,
)

#: Default component weights. Overridable per template.
DEFAULT_WEIGHTS = {
    "structured": 0.35,
    "semantic": 0.45,
    "type": 0.10,
    "recency": 0.10,
}

#: Age at which a precedent counts half as much on the recency component.
RECENCY_HALF_LIFE_DAYS = 365.0

CAUSAL_NOTE = (
    "Precedents are ranked by similarity, not by whether their outcome was good. "
    "A decision that worked before is historically associated with this situation, "
    "not evidence that it will work again."
)


@dataclass
class PrecedentContext:
    """The subset of a decision that ranking needs."""

    id: str
    title: str
    decision_type: str
    template_id: str | None
    context_structured: dict[str, Any]
    context_text: str = ""
    chosen_option: str | None = None
    rationale: str = ""
    decided_at: datetime | None = None
    embedding: Sequence[float] | None = None
    outcome_success: str | None = None
    outcome_metrics: dict[str, float] = field(default_factory=dict)
    confidentiality: str = "internal"


@dataclass
class ComponentScore:
    name: str
    weight: float
    score: float
    available: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": round(self.weight, 4),
            "score": round(self.score, 4),
            "available": self.available,
            "detail": self.detail,
        }


@dataclass
class Precedent:
    decision_id: str
    title: str
    score: float
    components: list[ComponentScore]
    structured: StructuredSimilarity
    chosen_option: str | None
    decided_at: datetime | None
    outcome_success: str | None
    outcome_metrics: dict[str, float] = field(default_factory=dict)
    context_coverage: float = 0.0

    def as_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "title": self.title,
            "score": round(self.score, 4),
            "components": [c.as_dict() for c in self.components],
            "structured": self.structured.as_dict(),
            "chosen_option": self.chosen_option,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "outcome_success": self.outcome_success,
            "outcome_metrics": self.outcome_metrics,
            "context_coverage": self.context_coverage,
        }


@dataclass
class SearchResult:
    precedents: list[Precedent] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)
    semantic_available: bool = False
    candidates_considered: int = 0
    note: str = CAUSAL_NOTE

    def as_dict(self) -> dict:
        return {
            "precedents": [p.as_dict() for p in self.precedents],
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "semantic_available": self.semantic_available,
            "candidates_considered": self.candidates_considered,
            "note": self.note,
        }


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Raw cosine similarity, clamped at 0.

    Not remapped from ``[-1, 1]`` onto ``[0, 1]``: that gives unrelated text a
    score of 0.5, which is a floor the ranking then has to climb out of. A
    negative cosine means "no relationship", and 0 says that honestly.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return float(max(0.0, dot / (norm_left * norm_right)))


def spread_semantic(raw: Sequence[float]) -> list[float]:
    """Rescale semantic scores against the candidate set's own distribution.

    An absolute cosine carries almost no information on its own: its scale
    depends entirely on the embedding model and on how alike the corpus is. On
    this demo corpus every context shares a sentence template, so raw cosine
    ranges 0.49-0.80 across *all* candidates -- a near-constant that adds the
    same amount to every score and flattens the ranking into noise.

    The least similar candidate maps to 0, the most similar to 1, and the rest
    spread linearly between. The raw value is kept in the component detail, so
    nothing is hidden -- but the number that enters the blend is the one that
    actually distinguishes candidates.

    Relative ordering is preserved exactly. An earlier version anchored at the
    median and clamped below it, which collapsed the whole bottom half to zero
    and threw away real ordering information among those candidates.

    This only affects the semantic component. Structured similarity is absolute
    and interpretable on its own, and is deliberately left alone.
    """
    values = [v for v in raw if v > 0]
    if len(values) < 3:
        return list(raw)

    low, high = min(values), max(values)
    if high <= low:
        return list(raw)
    return [float(min(1.0, max(0.0, (v - low) / (high - low)))) for v in raw]


def recency_score(decided_at: datetime | None, *, now: datetime | None = None) -> float:
    if decided_at is None:
        return 0.0
    reference = now or datetime.now(timezone.utc)
    stamp = decided_at if decided_at.tzinfo else decided_at.replace(tzinfo=timezone.utc)
    days = max((reference - stamp).total_seconds() / 86400.0, 0.0)
    return float(math.exp(-days / RECENCY_HALF_LIFE_DAYS * math.log(2)))


def rank_precedents(
    target: PrecedentContext,
    candidates: Sequence[PrecedentContext],
    template: DecisionTemplate,
    *,
    weights: dict[str, float] | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    now: datetime | None = None,
) -> SearchResult:
    """Rank historical decisions against the situation at hand."""
    configured = {**DEFAULT_WEIGHTS, **(template.ranking_weights or {}), **(weights or {})}

    pool = [c for c in candidates if c.id != target.id]

    # Semantic scores are rescaled against this candidate set, so the component
    # has to be computed for everyone before any single result can be scored.
    raw_semantic = [
        cosine(target.embedding or [], c.embedding or []) if target.embedding and c.embedding else 0.0
        for c in pool
    ]
    spread = spread_semantic(raw_semantic)

    ranked: list[Precedent] = []
    semantic_seen = False

    for candidate, raw_score, spread_score in zip(pool, raw_semantic, spread):

        structured = structured_similarity(
            template, target.context_structured, candidate.context_structured
        )

        components: list[ComponentScore] = []

        # Structured is unavailable when no field could be compared at all,
        # rather than scoring zero for two empty contexts.
        structured_available = bool(structured.contributions)
        components.append(
            ComponentScore(
                name="structured",
                weight=configured["structured"],
                score=structured.score,
                available=structured_available,
                detail=(
                    f"{len(structured.contributions)} field(s) compared"
                    if structured_available
                    else "no shared fields to compare"
                ),
            )
        )

        semantic_available = bool(target.embedding) and bool(candidate.embedding)
        semantic_seen = semantic_seen or semantic_available
        components.append(
            ComponentScore(
                name="semantic",
                weight=configured["semantic"],
                score=spread_score,
                available=semantic_available,
                detail=(
                    f"cosine {raw_score:.3f}, rescaled against this candidate set"
                    if semantic_available
                    else "no embedding available; weight redistributed"
                ),
            )
        )

        same_type = candidate.decision_type == target.decision_type
        components.append(
            ComponentScore(
                name="type",
                weight=configured["type"],
                score=1.0 if same_type else 0.0,
                available=True,
                detail=f"{candidate.decision_type} vs {target.decision_type}",
            )
        )

        components.append(
            ComponentScore(
                name="recency",
                weight=configured["recency"],
                score=recency_score(candidate.decided_at, now=now),
                available=candidate.decided_at is not None,
                detail=(
                    candidate.decided_at.date().isoformat()
                    if candidate.decided_at
                    else "no decision date"
                ),
            )
        )

        total = combine(components)
        if total < min_score:
            continue

        ranked.append(
            Precedent(
                decision_id=candidate.id,
                title=candidate.title,
                score=total,
                components=components,
                structured=structured,
                chosen_option=candidate.chosen_option,
                decided_at=candidate.decided_at,
                outcome_success=candidate.outcome_success,
                outcome_metrics=candidate.outcome_metrics,
                context_coverage=coverage(template, candidate.context_structured),
            )
        )

    ranked.sort(key=lambda p: (-p.score, p.decision_id))
    return SearchResult(
        precedents=ranked[:limit],
        weights_used=configured,
        semantic_available=semantic_seen,
        candidates_considered=len(pool),
    )


def combine(components: Sequence[ComponentScore]) -> float:
    """Weighted mean over the components that were actually available.

    Renormalising over what is available is the whole point: with no
    embeddings, structured, type and recency carry the full weight instead of
    the result being capped by a missing component.
    """
    usable = [c for c in components if c.available and c.weight > 0]
    total_weight = sum(c.weight for c in usable)
    if total_weight <= 0:
        return 0.0
    return round(sum(c.score * c.weight for c in usable) / total_weight, 6)
