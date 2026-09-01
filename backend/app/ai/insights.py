"""AI layer.

The model receives computed evidence — ranked precedents with their component
scores, aggregate outcome statistics, and the caveats the summariser attached —
and returns prose.

Two constraints are enforced here rather than hoped for:

* **Epistemic labelling.** Every claim the model makes must be tagged as an
  observed fact, a historical association, an inference, or unknown. A precedent
  system that blurs "this happened 9 times" into "this will work" is worse than
  no system, because it launders a small sample into confidence.
* **Confidentiality redaction.** Restricted decisions are removed before the
  prompt is built, and the number withheld is recorded on the call log. The
  caller is told the summary is partial rather than being handed a confident
  answer computed from less than it appears.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.ai.provider import AICallResult, get_provider
from app.models import AILogEntry

RESTRICTED = "restricted"

SYSTEM_PRECEDENT = (
    "You are helping someone make a business decision by summarising what the "
    "organisation did in comparable situations before. You are given ranked "
    "precedents and aggregate statistics computed from them. Use only those "
    "numbers. Never state or imply that a past outcome will repeat: describe "
    "findings as historically associated with the situation. Label every claim "
    "as an observed fact, a historical association, an inference, or unknown."
)

SYSTEM_EXTRACT = (
    "You extract structured fields from a free-text description of a business "
    "decision. Return only values the text actually supports. Leave a field out "
    "rather than guessing it; a wrong value is far worse than a missing one, "
    "because it silently changes which precedents are retrieved."
)

SYSTEM_BRIEF = (
    "You are drafting a decision brief for a manager. You are given the current "
    "situation, comparable past decisions and their outcomes. Present the options "
    "neutrally. Do not recommend one unless the evidence clearly supports it, and "
    "say plainly what is not known."
)

SYSTEM_RETRO = (
    "You are summarising retrospectives written after decisions were made. "
    "Identify recurring lessons. Use only what the notes say."
)


class Claim(BaseModel):
    statement: str
    kind: Literal["observed_fact", "historical_association", "inference", "unknown"] = (
        "historical_association"
    )
    basis: str = Field(default="", description="Which supplied figure supports this")


class PrecedentNarrative(BaseModel):
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ExtractedContext(BaseModel):
    fields: dict[str, Any] = Field(default_factory=dict)
    unmapped_notes: list[str] = Field(default_factory=list)
    confidence_notes: list[str] = Field(default_factory=list)


class DecisionBrief(BaseModel):
    situation: str
    options_considered: list[str] = Field(default_factory=list)
    what_history_shows: list[Claim] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_step: str = ""


class RetrospectiveSummary(BaseModel):
    recurring_lessons: list[str] = Field(default_factory=list)
    one_off_notes: list[str] = Field(default_factory=list)
    sample_size: int = 0


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{7,}\d)(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def redact_text(text: str) -> str:
    """Strip direct identifiers before anything leaves the process."""
    if not text:
        return text
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_DIGITS.sub("[number]", text)


def filter_confidential(items: Sequence[dict]) -> tuple[list[dict], int]:
    """Drop restricted decisions. Returns ``(kept, withheld_count)``."""
    kept = [i for i in items if i.get("confidentiality") != RESTRICTED]
    return kept, len(items) - len(kept)


def summarise_precedents(
    db: Session,
    tenant_id: uuid.UUID,
    *,
    situation: dict,
    precedents: Sequence[dict],
    statistics: dict,
) -> tuple[PrecedentNarrative | None, int]:
    """Explain a set of precedents. Returns ``(narrative, withheld_count)``."""
    kept, withheld = filter_confidential(precedents)
    evidence = {
        "situation": {
            "title": redact_text(str(situation.get("title", ""))),
            "context": redact_text(str(situation.get("context_text", ""))),
            "structured": situation.get("context_structured", {}),
        },
        "precedents": [
            {
                "title": redact_text(str(p.get("title", ""))),
                "similarity": p.get("score"),
                "chosen_option": p.get("chosen_option"),
                "decided_at": p.get("decided_at"),
                "outcome": p.get("outcome_success"),
                "outcome_metrics": p.get("outcome_metrics", {}),
            }
            for p in kept
        ],
        "aggregate_statistics": statistics,
        "withheld_for_confidentiality": withheld,
        "caveat": (
            "Outcome rates are computed only over decisions that have a recorded "
            "outcome. Decisions without one are counted separately and must not be "
            "treated as successes."
        ),
    }
    result = get_provider().structured(
        system=SYSTEM_PRECEDENT,
        evidence=evidence,
        output_model=PrecedentNarrative,
        prompt_version="precedent-summary-v1",
    )
    _log(db, tenant_id, "summarise_precedents", result, withheld)

    narrative = result.output if result.ok else None
    if narrative is not None and withheld:
        narrative.caveats.append(
            f"{withheld} restricted decision(s) were excluded from this summary."
        )
    return narrative, withheld


def extract_context(
    db: Session, tenant_id: uuid.UUID, *, text: str, template: dict
) -> ExtractedContext | None:
    """Pull template fields out of a free-text description."""
    evidence = {
        "text": redact_text(text),
        "template_fields": template.get("fields", []),
        "instruction": (
            "Only include a field when the text states or clearly implies its value."
        ),
    }
    result = get_provider().structured(
        system=SYSTEM_EXTRACT,
        evidence=evidence,
        output_model=ExtractedContext,
        prompt_version="context-extractor-v1",
    )
    _log(db, tenant_id, "extract_context", result, 0)
    return result.output if result.ok else None


def draft_brief(
    db: Session,
    tenant_id: uuid.UUID,
    *,
    situation: dict,
    precedents: Sequence[dict],
    statistics: dict,
) -> tuple[DecisionBrief | None, int]:
    kept, withheld = filter_confidential(precedents)
    evidence = {
        "situation": situation,
        "precedents": kept,
        "aggregate_statistics": statistics,
        "withheld_for_confidentiality": withheld,
    }
    result = get_provider().structured(
        system=SYSTEM_BRIEF,
        evidence=evidence,
        output_model=DecisionBrief,
        prompt_version="decision-brief-v1",
    )
    _log(db, tenant_id, "draft_brief", result, withheld)
    return (result.output if result.ok else None), withheld


def summarise_retrospectives(
    db: Session, tenant_id: uuid.UUID, notes: Sequence[str]
) -> RetrospectiveSummary | None:
    cleaned = [redact_text(n) for n in notes if n and n.strip()]
    if not cleaned:
        return RetrospectiveSummary(sample_size=0)

    result = get_provider().structured(
        system=SYSTEM_RETRO,
        evidence={"retrospectives": cleaned, "count": len(cleaned)},
        output_model=RetrospectiveSummary,
        prompt_version="retrospective-summary-v1",
    )
    _log(db, tenant_id, "summarise_retrospectives", result, 0)
    if not result.ok:
        return None
    summary = result.output
    summary.sample_size = len(cleaned)
    return summary


def _log(
    db: Session, tenant_id: uuid.UUID, purpose: str, result: AICallResult, redacted: int
) -> None:
    db.add(
        AILogEntry(
            tenant_id=tenant_id,
            purpose=purpose,
            model=result.model,
            prompt_version=result.prompt_version,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            redacted_count=redacted,
            error=result.error,
            created_at=datetime.now(timezone.utc),
        )
    )
    db.flush()
