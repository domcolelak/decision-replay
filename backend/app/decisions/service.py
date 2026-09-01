"""Service layer: database <-> ranking, comparison and embeddings.

Every query is tenant scoped. Confidentiality is applied here rather than in
the API layer, so no route can forget it.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.comparison.compare import DecisionView
from app.embeddings.provider import embedding_text, get_provider
from app.models import (
    Decision,
    DecisionEmbedding,
    DecisionOption,
    DecisionTemplateRow,
    Outcome,
)
from app.search.ranking import PrecedentContext
from app.templates.fields import DecisionTemplate

#: Decisions at this level are never included in precedent search results for
#: other decisions, and never leave the system in an AI prompt.
RESTRICTED = "restricted"


def get_decision(db: Session, tenant_id: uuid.UUID, decision_id: uuid.UUID) -> Decision | None:
    return db.scalar(
        select(Decision).where(Decision.tenant_id == tenant_id, Decision.id == decision_id)
    )


def get_template(
    db: Session, tenant_id: uuid.UUID, template_id: uuid.UUID | None
) -> DecisionTemplateRow | None:
    if template_id is None:
        return None
    return db.scalar(
        select(DecisionTemplateRow).where(
            DecisionTemplateRow.tenant_id == tenant_id,
            DecisionTemplateRow.id == template_id,
        )
    )


def template_spec(row: DecisionTemplateRow | None) -> DecisionTemplate:
    """A usable template even when a decision has none attached.

    Without this, an ad-hoc decision could not be compared at all. An empty
    template contributes no structured similarity, and the ranking falls back
    to the remaining components -- which is the honest behaviour.
    """
    if row is None:
        return DecisionTemplate(name="(none)", decision_type="", fields=[])
    return DecisionTemplate.from_dict(
        {
            "name": row.name,
            "decision_type": row.decision_type,
            "fields": row.fields or [],
            "ranking_weights": row.ranking_weights or {},
        }
    )


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def source_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:32]


def refresh_embedding(db: Session, tenant_id: uuid.UUID, decision: Decision) -> DecisionEmbedding | None:
    """Generate and store the context vector for one decision.

    Skips the work when the text has not changed and the stored vector came
    from the current model -- a model change invalidates every vector, and
    comparing across models silently produces nonsense.
    """
    provider = get_provider()
    text = embedding_text(decision.title, decision.context_text, decision.context_structured or {})
    digest = source_hash(text)

    existing = db.scalar(
        select(DecisionEmbedding).where(
            DecisionEmbedding.tenant_id == tenant_id,
            DecisionEmbedding.decision_id == decision.id,
        )
    )
    if (
        existing is not None
        and existing.source_hash == digest
        and existing.model == provider.name
        and existing.model_version == provider.version
    ):
        return existing

    result = provider.embed([text])
    if not result.ok:
        # Embeddings are an enhancement, never a prerequisite: the decision is
        # saved and structured search keeps working without a vector.
        return None

    vector = result.vectors[0]
    if existing is None:
        existing = DecisionEmbedding(tenant_id=tenant_id, decision_id=decision.id)
        db.add(existing)
    existing.model = result.model
    existing.model_version = result.version
    existing.dimensions = result.dimensions
    existing.vector = vector
    existing.source_hash = digest
    db.flush()
    return existing


def precedent_contexts(
    db: Session,
    tenant_id: uuid.UUID,
    *,
    decision_type: str | None = None,
    include_restricted: bool = False,
    exclude_id: uuid.UUID | None = None,
) -> list[PrecedentContext]:
    """Candidate precedents, with embeddings and outcomes attached."""
    stmt = select(Decision).where(Decision.tenant_id == tenant_id)
    if decision_type:
        stmt = stmt.where(Decision.decision_type == decision_type)
    if not include_restricted:
        stmt = stmt.where(Decision.confidentiality != RESTRICTED)
    if exclude_id is not None:
        stmt = stmt.where(Decision.id != exclude_id)

    contexts: list[PrecedentContext] = []
    for row in db.scalars(stmt):
        embedding = db.scalar(
            select(DecisionEmbedding).where(
                DecisionEmbedding.tenant_id == tenant_id,
                DecisionEmbedding.decision_id == row.id,
            )
        )
        outcome = db.scalar(
            select(Outcome).where(
                Outcome.tenant_id == tenant_id, Outcome.decision_id == row.id
            )
        )
        contexts.append(
            PrecedentContext(
                id=str(row.id),
                title=row.title,
                decision_type=row.decision_type,
                template_id=str(row.template_id) if row.template_id else None,
                context_structured=row.context_structured or {},
                context_text=row.context_text or "",
                chosen_option=row.chosen_option,
                rationale=row.rationale or "",
                decided_at=_aware(row.decided_at),
                embedding=embedding.vector if embedding else None,
                outcome_success=outcome.success_label if outcome else None,
                outcome_metrics=(outcome.metrics or {}) if outcome else {},
                confidentiality=row.confidentiality,
            )
        )
    return contexts


def as_precedent_context(
    db: Session, tenant_id: uuid.UUID, decision: Decision
) -> PrecedentContext:
    embedding = db.scalar(
        select(DecisionEmbedding).where(
            DecisionEmbedding.tenant_id == tenant_id,
            DecisionEmbedding.decision_id == decision.id,
        )
    )
    return PrecedentContext(
        id=str(decision.id),
        title=decision.title,
        decision_type=decision.decision_type,
        template_id=str(decision.template_id) if decision.template_id else None,
        context_structured=decision.context_structured or {},
        context_text=decision.context_text or "",
        chosen_option=decision.chosen_option,
        rationale=decision.rationale or "",
        decided_at=_aware(decision.decided_at),
        embedding=embedding.vector if embedding else None,
        confidentiality=decision.confidentiality,
    )


def decision_views(
    db: Session, tenant_id: uuid.UUID, decision_ids: Sequence[uuid.UUID]
) -> list[DecisionView]:
    """Load decisions in the shape comparison and summary need."""
    views: list[DecisionView] = []
    for decision_id in decision_ids:
        row = get_decision(db, tenant_id, decision_id)
        if row is None:
            continue
        outcome = db.scalar(
            select(Outcome).where(
                Outcome.tenant_id == tenant_id, Outcome.decision_id == row.id
            )
        )
        views.append(
            DecisionView(
                id=str(row.id),
                title=row.title,
                decision_type=row.decision_type,
                context_structured=row.context_structured or {},
                chosen_option=row.chosen_option,
                rationale=row.rationale or "",
                decided_at=_aware(row.decided_at),
                outcome_label=outcome.success_label if outcome else None,
                outcome_metrics=(outcome.metrics or {}) if outcome else {},
                outcome_notes=outcome.notes if outcome else "",
                retrospective=outcome.retrospective if outcome else "",
                owner=row.owner or "",
            )
        )
    return views


def replace_options(
    db: Session, tenant_id: uuid.UUID, decision: Decision, options: Sequence[dict]
) -> None:
    for existing in list(decision.options):
        db.delete(existing)
    db.flush()
    for position, option in enumerate(options):
        db.add(
            DecisionOption(
                tenant_id=tenant_id,
                decision_id=decision.id,
                key=str(option.get("key", f"option_{position}")),
                label=str(option.get("label", "")),
                notes=str(option.get("notes", "")),
                position=position,
            )
        )
    db.flush()


def overdue_outcomes(
    db: Session, tenant_id: uuid.UUID, *, now: datetime | None = None
) -> list[Decision]:
    """Decisions whose outcome was due and never recorded.

    The scheduler hook the brief asks for: this is the query a reminder job
    would run. Outcomes are what turn memory into evidence, and they are
    exactly what people forget.
    """
    reference = now or datetime.now(timezone.utc)
    rows = db.scalars(
        select(Decision).where(
            Decision.tenant_id == tenant_id,
            Decision.outcome_due_at.is_not(None),
        )
    ).all()

    overdue = []
    for row in rows:
        due = _aware(row.outcome_due_at)
        if due is None or due > reference:
            continue
        has_outcome = db.scalar(
            select(Outcome).where(
                Outcome.tenant_id == tenant_id, Outcome.decision_id == row.id
            )
        )
        if has_outcome is None:
            overdue.append(row)
    overdue.sort(key=lambda d: _aware(d.outcome_due_at) or reference)
    return overdue
