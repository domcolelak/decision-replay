"""HTTP API.

Every endpoint resolves a :class:`TenantContext` first and filters every query
by ``ctx.tenant_id``. Cross-tenant access returns 404, not 403.

Literal paths are declared before parameterised ones on the same prefix:
FastAPI matches in declaration order, so a literal declared afterwards is
unreachable.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.insights import draft_brief, extract_context, summarise_precedents
from app.audit.log import record_audit
from app.comparison.compare import build_comparison, summarise_precedents as aggregate
from app.core.db import get_db
from app.core.security import TenantContext, current_tenant
from app.decisions.service import (
    as_precedent_context,
    decision_views,
    get_decision,
    get_template,
    overdue_outcomes,
    precedent_contexts,
    refresh_embedding,
    replace_options,
    template_spec,
)
from app.embeddings.provider import embedding_text, get_provider
from app.models import (
    Decision,
    DecisionEmbedding,
    DecisionEvidence,
    DecisionPacket,
    DecisionTemplateRow,
    Outcome,
    SearchLog,
)
from app.packets.builder import build_payload, render_markdown
from app.schemas import (
    CompareRequest,
    CompareResponse,
    DecisionCreate,
    DecisionDetail,
    DecisionOut,
    DecisionUpdate,
    ExtractRequest,
    OutcomeIn,
    OutcomeOut,
    OverdueOutcome,
    OverviewResponse,
    PacketOut,
    PacketRequest,
    SearchRequest,
    SearchResponse,
    SummariseRequest,
    TemplateCreate,
    TemplateListItem,
    TemplateOut,
)
from app.search.ranking import PrecedentContext, rank_precedents
from app.templates.fields import TemplateField, coverage

router = APIRouter()


def _template_out(row: DecisionTemplateRow) -> TemplateOut:
    """Serialise a template with every field key present.

    Rows can be written from several places -- the API, the demo seed, a
    migration -- and a raw dict easily omits an optional key. A client then has
    to defend against a shape that should never have varied. Normalising
    through the canonical field type here guarantees one consistent response.
    """
    normalised = []
    for raw in row.fields or []:
        try:
            normalised.append(TemplateField.from_dict(raw).as_dict())
        except (KeyError, ValueError):
            # A field that cannot be parsed is surfaced as-is rather than
            # dropped: hiding it would make the template look complete.
            normalised.append(raw)
    return TemplateOut(
        id=row.id,
        name=row.name,
        decision_type=row.decision_type,
        description=row.description,
        fields=normalised,
        ranking_weights=row.ranking_weights or {},
        created_at=row.created_at,
    )


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


@router.get("/templates", response_model=list[TemplateListItem])
def list_templates(
    ctx: TenantContext = Depends(current_tenant), db: Session = Depends(get_db)
) -> list[TemplateListItem]:
    rows = db.scalars(
        select(DecisionTemplateRow)
        .where(DecisionTemplateRow.tenant_id == ctx.tenant_id)
        .order_by(DecisionTemplateRow.name)
    ).all()
    items = []
    for row in rows:
        used = db.scalar(
            select(func.count(Decision.id)).where(
                Decision.tenant_id == ctx.tenant_id, Decision.template_id == row.id
            )
        )
        items.append(
            TemplateListItem(**_template_out(row).model_dump(), decision_count=used or 0)
        )
    return items


@router.post("/templates", response_model=TemplateOut, status_code=status.HTTP_201_CREATED)
def create_template(
    body: TemplateCreate,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> TemplateOut:
    existing = db.scalar(
        select(DecisionTemplateRow).where(
            DecisionTemplateRow.tenant_id == ctx.tenant_id,
            DecisionTemplateRow.name == body.name,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="a template with that name already exists")

    names = [f.name for f in body.fields]
    if len(names) != len(set(names)):
        raise HTTPException(status_code=422, detail="field names must be unique")

    row = DecisionTemplateRow(
        tenant_id=ctx.tenant_id,
        name=body.name,
        decision_type=body.decision_type,
        description=body.description,
        fields=[f.model_dump() for f in body.fields],
        ranking_weights=body.ranking_weights,
    )
    db.add(row)
    db.flush()
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="template.created",
        object_type="template",
        object_id=row.id,
    )
    return _template_out(row)


@router.get("/templates/{template_id}", response_model=TemplateOut)
def get_template_detail(
    template_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> TemplateOut:
    return _template_out(_require_template(db, ctx, template_id))


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


@router.get("/decisions", response_model=list[DecisionOut])
def list_decisions(
    decision_type: str | None = None,
    undecided_only: bool = False,
    limit: int = Query(100, ge=1, le=500),
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> list[Decision]:
    stmt = select(Decision).where(Decision.tenant_id == ctx.tenant_id)
    if decision_type:
        stmt = stmt.where(Decision.decision_type == decision_type)
    if undecided_only:
        stmt = stmt.where(Decision.chosen_option.is_(None))
    return list(db.scalars(stmt.order_by(Decision.decided_at.desc().nullsfirst()).limit(limit)))


@router.get("/decisions/overdue-outcomes", response_model=list[OverdueOutcome])
def list_overdue(
    ctx: TenantContext = Depends(current_tenant), db: Session = Depends(get_db)
) -> list[OverdueOutcome]:
    """Decisions whose outcome was due and never recorded.

    Declared before `/decisions/{decision_id}` so the literal path is reachable.
    """
    now = datetime.now(timezone.utc)
    out = []
    for row in overdue_outcomes(db, ctx.tenant_id, now=now):
        due = row.outcome_due_at
        due = due if due.tzinfo else due.replace(tzinfo=timezone.utc)
        out.append(
            OverdueOutcome(
                decision_id=row.id,
                title=row.title,
                owner=row.owner,
                decided_at=row.decided_at,
                outcome_due_at=row.outcome_due_at,
                days_overdue=max((now - due).days, 0),
            )
        )
    return out


@router.post("/decisions", response_model=DecisionOut, status_code=status.HTTP_201_CREATED)
def create_decision(
    body: DecisionCreate,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> Decision:
    template = get_template(db, ctx.tenant_id, body.template_id) if body.template_id else None
    if body.template_id and template is None:
        raise HTTPException(status_code=404, detail="template not found")

    if template is not None:
        problems = template_spec(template).validate(body.context_structured)
        if problems:
            raise HTTPException(status_code=422, detail="; ".join(problems))

    if body.external_id:
        clash = db.scalar(
            select(Decision).where(
                Decision.tenant_id == ctx.tenant_id, Decision.external_id == body.external_id
            )
        )
        if clash is not None:
            raise HTTPException(status_code=409, detail="external_id already used")

    decision = Decision(
        tenant_id=ctx.tenant_id,
        template_id=body.template_id,
        external_id=body.external_id,
        title=body.title,
        decision_type=body.decision_type,
        context_text=body.context_text,
        context_structured=body.context_structured,
        chosen_option=body.chosen_option,
        rationale=body.rationale,
        owner=body.owner,
        stakeholders=body.stakeholders,
        decided_at=body.decided_at,
        expected_outcome=body.expected_outcome,
        outcome_due_at=body.outcome_due_at,
        confidentiality=body.confidentiality,
        tags=body.tags,
        source_links=body.source_links,
    )
    db.add(decision)
    db.flush()

    replace_options(db, ctx.tenant_id, decision, [o.model_dump() for o in body.options])
    for item in body.evidence:
        db.add(
            DecisionEvidence(
                tenant_id=ctx.tenant_id,
                decision_id=decision.id,
                kind=item.kind,
                summary=item.summary,
                url=item.url,
                payload=item.payload,
            )
        )
    db.flush()

    refresh_embedding(db, ctx.tenant_id, decision)
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="decision.created",
        object_type="decision",
        object_id=decision.id,
    )
    return decision


@router.get("/decisions/{decision_id}", response_model=DecisionDetail)
def get_decision_detail(
    decision_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> DecisionDetail:
    decision = _require_decision(db, ctx, decision_id)
    template = get_template(db, ctx.tenant_id, decision.template_id)
    spec = template_spec(template)

    outcome = db.scalar(
        select(Outcome).where(
            Outcome.tenant_id == ctx.tenant_id, Outcome.decision_id == decision_id
        )
    )
    embedding = db.scalar(
        select(DecisionEmbedding).where(
            DecisionEmbedding.tenant_id == ctx.tenant_id,
            DecisionEmbedding.decision_id == decision_id,
        )
    )

    return DecisionDetail(
        **DecisionOut.model_validate(decision).model_dump(),
        options=[
            {"key": o.key, "label": o.label, "notes": o.notes, "position": o.position}
            for o in sorted(decision.options, key=lambda o: o.position)
        ],
        evidence=[
            {"kind": e.kind, "summary": e.summary, "url": e.url, "payload": e.payload}
            for e in decision.evidence
        ],
        outcome=(
            {
                "success_label": outcome.success_label,
                "metrics": outcome.metrics,
                "notes": outcome.notes,
                "retrospective": outcome.retrospective,
                "recorded_at": outcome.recorded_at.isoformat(),
            }
            if outcome
            else None
        ),
        context_coverage=coverage(spec, decision.context_structured or {}),
        validation_problems=spec.validate(decision.context_structured or {}),
        embedding=(
            {
                "model": embedding.model,
                "version": embedding.model_version,
                "dimensions": embedding.dimensions,
            }
            if embedding
            else None
        ),
    )


@router.patch("/decisions/{decision_id}", response_model=DecisionOut)
def update_decision(
    decision_id: uuid.UUID,
    body: DecisionUpdate,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> Decision:
    decision = _require_decision(db, ctx, decision_id)
    for name, value in body.model_dump(exclude_unset=True).items():
        setattr(decision, name, value)
    db.flush()

    # The context drives the vector, so an edit invalidates it.
    refresh_embedding(db, ctx.tenant_id, decision)
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="decision.updated",
        object_type="decision",
        object_id=decision_id,
        payload={"fields": sorted(body.model_dump(exclude_unset=True))},
    )
    return decision


@router.put("/decisions/{decision_id}/outcome", response_model=OutcomeOut)
def record_outcome(
    decision_id: uuid.UUID,
    body: OutcomeIn,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> Outcome:
    """Record what actually happened. This is what turns memory into evidence."""
    _require_decision(db, ctx, decision_id)
    outcome = db.scalar(
        select(Outcome).where(
            Outcome.tenant_id == ctx.tenant_id, Outcome.decision_id == decision_id
        )
    )
    if outcome is None:
        outcome = Outcome(tenant_id=ctx.tenant_id, decision_id=decision_id)
        db.add(outcome)
    outcome.success_label = body.success_label
    outcome.metrics = body.metrics
    outcome.notes = body.notes
    outcome.retrospective = body.retrospective
    outcome.recorded_by = body.recorded_by
    outcome.recorded_at = datetime.now(timezone.utc)
    db.flush()

    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="outcome.recorded",
        object_type="decision",
        object_id=decision_id,
        payload={"label": body.success_label},
    )
    return outcome


# --------------------------------------------------------------------------
# Search, comparison, packets
# --------------------------------------------------------------------------


def _target_for(
    db: Session, ctx: TenantContext, body: SearchRequest | SummariseRequest
) -> tuple[PrecedentContext, DecisionTemplateRow | None, str]:
    """Build the search target from a saved decision or a live situation."""
    if body.decision_id is not None:
        decision = _require_decision(db, ctx, body.decision_id)
        template = get_template(db, ctx.tenant_id, decision.template_id)
        return (
            as_precedent_context(db, ctx.tenant_id, decision),
            template,
            decision.decision_type,
        )

    template = get_template(db, ctx.tenant_id, body.template_id) if body.template_id else None
    if body.template_id and template is None:
        raise HTTPException(status_code=404, detail="template not found")

    decision_type = body.decision_type or (template.decision_type if template else "")
    if not decision_type:
        raise HTTPException(
            status_code=422,
            detail="supply decision_id, or a decision_type / template_id for the situation",
        )

    # An unsaved situation is embedded on the fly so it can be compared
    # semantically without being stored first.
    text = embedding_text(body.title, body.context_text, body.context_structured)
    result = get_provider().embed([text])
    return (
        PrecedentContext(
            id="__target__",
            title=body.title or "Current situation",
            decision_type=decision_type,
            template_id=str(body.template_id) if body.template_id else None,
            context_structured=body.context_structured,
            context_text=body.context_text,
            embedding=result.vectors[0] if result.ok else None,
        ),
        template,
        decision_type,
    )


@router.post("/decisions/search", response_model=SearchResponse)
def search_precedents(
    body: SearchRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> SearchResponse:
    """Rank comparable historical decisions, with every component score shown."""
    target, template, decision_type = _target_for(db, ctx, body)
    spec = template_spec(template)

    candidates = precedent_contexts(
        db,
        ctx.tenant_id,
        decision_type=decision_type or None,
        exclude_id=body.decision_id,
    )
    result = rank_precedents(
        target,
        candidates,
        spec,
        weights=body.weights or None,
        limit=body.limit,
        min_score=body.min_score,
    )

    statistics = None
    if body.include_summary:
        ids = [uuid.UUID(p.decision_id) for p in result.precedents]
        statistics = aggregate(decision_views(db, ctx.tenant_id, ids)).as_dict()

    db.add(
        SearchLog(
            tenant_id=ctx.tenant_id,
            decision_id=body.decision_id,
            query={
                "decision_type": decision_type,
                "context_structured": body.context_structured,
            },
            weights_used=result.weights_used,
            semantic_available=result.semantic_available,
            result_ids=[p.decision_id for p in result.precedents],
        )
    )
    db.flush()

    return SearchResponse(
        precedents=[p.as_dict() for p in result.precedents],
        weights_used=result.weights_used,
        semantic_available=result.semantic_available,
        candidates_considered=result.candidates_considered,
        statistics=statistics,
        note=result.note,
    )


@router.post("/decisions/compare", response_model=CompareResponse)
def compare_decisions(
    body: CompareRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> CompareResponse:
    views = decision_views(db, ctx.tenant_id, body.decision_ids)
    if len(views) != len(body.decision_ids):
        raise HTTPException(status_code=404, detail="one or more decisions not found")

    template = get_template(db, ctx.tenant_id, body.template_id) if body.template_id else None
    table = build_comparison(views, template_spec(template) if template else None)
    return CompareResponse(table=table.as_dict(), statistics=aggregate(views).as_dict())


@router.post("/decisions/summarise")
def summarise(
    body: SummariseRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> dict:
    """Narrative summary of the precedents, with every claim labelled."""
    target, template, decision_type = _target_for(db, ctx, body)
    spec = template_spec(template)
    candidates = precedent_contexts(
        db, ctx.tenant_id, decision_type=decision_type or None, exclude_id=body.decision_id
    )
    result = rank_precedents(target, candidates, spec, limit=body.limit)
    ids = [uuid.UUID(p.decision_id) for p in result.precedents]
    statistics = aggregate(decision_views(db, ctx.tenant_id, ids))

    by_id = {c.id: c for c in candidates}
    precedent_payload = [
        p.as_dict() | {"confidentiality": by_id[p.decision_id].confidentiality}
        for p in result.precedents
    ]

    narrative, withheld = summarise_precedents(
        db,
        ctx.tenant_id,
        situation={
            "title": target.title,
            "context_text": target.context_text,
            "context_structured": target.context_structured,
        },
        precedents=precedent_payload,
        statistics=statistics.as_dict(),
    )
    return {
        "narrative": narrative.model_dump() if narrative else None,
        "statistics": statistics.as_dict(),
        "withheld_for_confidentiality": withheld,
        "note": result.note,
    }


@router.post("/decisions/extract")
def extract(
    body: ExtractRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> dict:
    """Propose template fields from free text. Advisory until a human accepts."""
    template = _require_template(db, ctx, body.template_id)
    extracted = extract_context(
        db, ctx.tenant_id, text=body.text, template={"fields": template.fields or []}
    )
    if extracted is None:
        return {"fields": {}, "unmapped_notes": [], "confidence_notes": [], "problems": []}

    spec = template_spec(template)
    return extracted.model_dump() | {
        "problems": spec.validate(extracted.fields),
        "note": "Extracted values are a suggestion; confirm them before saving.",
    }


@router.post("/decision-packets", response_model=PacketOut, status_code=status.HTTP_201_CREATED)
def create_packet(
    body: PacketRequest,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> DecisionPacket:
    decision = _require_decision(db, ctx, body.decision_id)
    template = get_template(db, ctx.tenant_id, decision.template_id)
    spec = template_spec(template)

    target = as_precedent_context(db, ctx.tenant_id, decision)
    candidates = precedent_contexts(
        db, ctx.tenant_id, decision_type=decision.decision_type, exclude_id=decision.id
    )
    result = rank_precedents(target, candidates, spec, limit=body.limit)
    ids = [uuid.UUID(p.decision_id) for p in result.precedents]
    statistics = aggregate(decision_views(db, ctx.tenant_id, ids))

    ai_summary = None
    withheld = 0
    if body.include_ai_summary:
        by_id = {c.id: c for c in candidates}
        narrative, withheld = summarise_precedents(
            db,
            ctx.tenant_id,
            situation={
                "title": decision.title,
                "context_text": decision.context_text,
                "context_structured": decision.context_structured,
            },
            precedents=[
                p.as_dict() | {"confidentiality": by_id[p.decision_id].confidentiality}
                for p in result.precedents
            ],
            statistics=statistics.as_dict(),
        )
        ai_summary = narrative.model_dump() if narrative else None

    payload = build_payload(
        decision={
            "title": decision.title,
            "context_text": decision.context_text,
            "context_structured": decision.context_structured,
            "chosen_option": decision.chosen_option,
            "rationale": decision.rationale,
            "options": [
                {"key": o.key, "label": o.label, "notes": o.notes}
                for o in sorted(decision.options, key=lambda o: o.position)
            ],
        },
        search=result,
        summary=statistics,
        ai_summary=ai_summary,
        withheld=withheld,
    )

    packet = DecisionPacket(
        tenant_id=ctx.tenant_id,
        decision_id=decision.id,
        title=f"Decision packet: {decision.title}",
        body=render_markdown(payload),
        payload=payload,
    )
    db.add(packet)
    db.flush()
    record_audit(
        db,
        tenant_id=ctx.tenant_id,
        action="packet.created",
        object_type="decision",
        object_id=decision.id,
    )
    return packet


@router.get("/decision-packets/{packet_id}", response_model=PacketOut)
def get_packet(
    packet_id: uuid.UUID,
    ctx: TenantContext = Depends(current_tenant),
    db: Session = Depends(get_db),
) -> DecisionPacket:
    packet = db.scalar(
        select(DecisionPacket).where(
            DecisionPacket.tenant_id == ctx.tenant_id, DecisionPacket.id == packet_id
        )
    )
    if packet is None:
        raise HTTPException(status_code=404, detail="packet not found")
    return packet


@router.get("/overview", response_model=OverviewResponse)
def overview(
    ctx: TenantContext = Depends(current_tenant), db: Session = Depends(get_db)
) -> OverviewResponse:
    def count(model, *conditions):
        return (
            db.scalar(
                select(func.count(model.id)).where(model.tenant_id == ctx.tenant_id, *conditions)
            )
            or 0
        )

    decisions = count(Decision)
    embeddings = count(DecisionEmbedding)
    outcomes = db.scalars(
        select(Outcome.success_label).where(Outcome.tenant_id == ctx.tenant_id)
    ).all()

    mix: dict[str, int] = {}
    for label in outcomes:
        mix[label] = mix.get(label, 0) + 1

    recent = list(
        db.scalars(
            select(Decision)
            .where(Decision.tenant_id == ctx.tenant_id)
            .order_by(Decision.decided_at.desc().nullsfirst())
            .limit(8)
        )
    )

    return OverviewResponse(
        template_count=count(DecisionTemplateRow),
        decision_count=decisions,
        decided_count=count(Decision, Decision.chosen_option.is_not(None)),
        with_outcome=len(outcomes),
        overdue_outcomes=len(overdue_outcomes(db, ctx.tenant_id)),
        embedding_coverage=round(embeddings / decisions, 4) if decisions else 0.0,
        embedding_model=get_provider().name,
        outcome_mix=mix,
        recent_decisions=[DecisionOut.model_validate(d) for d in recent],
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _require_decision(db: Session, ctx: TenantContext, decision_id: uuid.UUID) -> Decision:
    decision = get_decision(db, ctx.tenant_id, decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="decision not found")
    return decision


def _require_template(
    db: Session, ctx: TenantContext, template_id: uuid.UUID
) -> DecisionTemplateRow:
    template = get_template(db, ctx.tenant_id, template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="template not found")
    return template
