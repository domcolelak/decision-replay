"""Demo workspace seeding."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import DEMO_TENANT_SLUG, hash_api_key
from app.decisions.service import refresh_embedding, replace_options
from app.demo.dataset import DEMO_NEW_DECISION, DISCOUNT_TEMPLATE, generate_decisions
from app.models import Decision, DecisionTemplateRow, Outcome, Tenant, User

DEMO_API_KEY = "pk_demo_decision_replay"


def ensure_demo_tenant(db: Session) -> Tenant:
    tenant = db.scalar(select(Tenant).where(Tenant.slug == DEMO_TENANT_SLUG))
    if tenant is None:
        tenant = Tenant(
            slug=DEMO_TENANT_SLUG,
            name="Demo workspace",
            api_key_hash=hash_api_key(DEMO_API_KEY),
        )
        db.add(tenant)
        db.flush()
        db.add(
            User(
                tenant_id=tenant.id,
                email="revops@demo.local",
                display_name="Demo revenue operations lead",
                role="admin",
            )
        )
        db.flush()
    return tenant


def seed_demo(db: Session, *, count: int = 90, force: bool = False) -> dict:
    """Create the demo tenant, template, decision history and a live situation."""
    tenant = ensure_demo_tenant(db)

    template = db.scalar(
        select(DecisionTemplateRow).where(
            DecisionTemplateRow.tenant_id == tenant.id,
            DecisionTemplateRow.name == DISCOUNT_TEMPLATE["name"],
        )
    )
    if template is not None and not force:
        existing = db.scalar(
            select(func.count(Decision.id)).where(Decision.tenant_id == tenant.id)
        )
        if existing:
            return {
                "tenant_id": str(tenant.id),
                "template_id": str(template.id),
                "created": False,
                "decision_count": existing,
            }

    if template is None:
        template = DecisionTemplateRow(
            tenant_id=tenant.id,
            name=DISCOUNT_TEMPLATE["name"],
            decision_type=DISCOUNT_TEMPLATE["decision_type"],
            description=(
                "Discount approval requests. Field weights say what makes two "
                "requests comparable: the size of the ask and the segment matter "
                "far more than the region."
            ),
            fields=DISCOUNT_TEMPLATE["fields"],
            ranking_weights=DISCOUNT_TEMPLATE["ranking_weights"],
        )
        db.add(template)
        db.flush()

    for row in generate_decisions(count=count):
        decision = Decision(
            tenant_id=tenant.id,
            template_id=template.id,
            external_id=row["external_id"],
            title=row["title"],
            decision_type=row["decision_type"],
            context_text=row["context_text"],
            context_structured=row["context_structured"],
            chosen_option=row["chosen_option"],
            rationale=row["rationale"],
            owner=row["owner"],
            decided_at=row["decided_at"],
            outcome_due_at=row["decided_at"] + timedelta(days=90),
            tags=row["tags"],
        )
        db.add(decision)
        db.flush()
        replace_options(db, tenant.id, decision, row["options"])
        refresh_embedding(db, tenant.id, decision)

        outcome = row["outcome"]
        if outcome:
            db.add(
                Outcome(
                    tenant_id=tenant.id,
                    decision_id=decision.id,
                    success_label=outcome["success_label"],
                    metrics=outcome["metrics"],
                    notes=outcome["notes"],
                    retrospective=outcome["retrospective"],
                    recorded_by="demo",
                    recorded_at=outcome["recorded_at"],
                )
            )
    db.flush()

    # A live, undecided situation so the demo opens on the screen that matters.
    live = Decision(
        tenant_id=tenant.id,
        template_id=template.id,
        external_id="DEC-LIVE",
        title=DEMO_NEW_DECISION["title"],
        decision_type=DEMO_NEW_DECISION["decision_type"],
        context_text=DEMO_NEW_DECISION["context_text"],
        context_structured=DEMO_NEW_DECISION["context_structured"],
        owner="ae_3",
        tags=["enterprise", "de", "renewal"],
    )
    db.add(live)
    db.flush()
    refresh_embedding(db, tenant.id, live)

    return {
        "tenant_id": str(tenant.id),
        "template_id": str(template.id),
        "live_decision_id": str(live.id),
        "created": True,
        "decision_count": count + 1,
    }
