from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

_TMP_DIR = Path(tempfile.mkdtemp(prefix="decision-replay-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+pysqlite:///{_TMP_DIR / 'test.db'}")
os.environ.setdefault("SEED_DEMO_ON_STARTUP", "false")
os.environ.setdefault("AI_PROVIDER", "offline")
os.environ.setdefault("EMBEDDING_PROVIDER", "offline")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.demo.dataset import DISCOUNT_TEMPLATE, generate_decisions  # noqa: E402
from app.demo.seed import ensure_demo_tenant, seed_demo  # noqa: E402
from app.embeddings.provider import OfflineHashingProvider, embedding_text  # noqa: E402
from app.main import app  # noqa: E402
from app.search.ranking import PrecedentContext  # noqa: E402
from app.templates.fields import DecisionTemplate  # noqa: E402

#: A fixed "now" so recency scores are stable across runs.
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


@pytest.fixture()
def tenant(db):
    t = ensure_demo_tenant(db)
    db.commit()
    return t


@pytest.fixture()
def seeded(db, tenant):
    result = seed_demo(db, count=60)
    db.commit()
    return result


@pytest.fixture()
def client(seeded):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def template():
    return DecisionTemplate.from_dict(DISCOUNT_TEMPLATE)


@pytest.fixture(scope="session")
def demo_rows():
    return generate_decisions(count=90)


@pytest.fixture(scope="session")
def demo_contexts(demo_rows):
    """The demo history as ranking candidates, embedded once."""
    provider = OfflineHashingProvider()
    vectors = provider.embed(
        [
            embedding_text(r["title"], r["context_text"], r["context_structured"])
            for r in demo_rows
        ]
    ).vectors
    return [
        PrecedentContext(
            id=row["external_id"],
            title=row["title"],
            decision_type=row["decision_type"],
            template_id="t",
            context_structured=row["context_structured"],
            context_text=row["context_text"],
            chosen_option=row["chosen_option"],
            rationale=row["rationale"],
            decided_at=row["decided_at"],
            embedding=vector,
            outcome_success=(row["outcome"] or {}).get("success_label"),
            outcome_metrics=(row["outcome"] or {}).get("metrics", {}),
        )
        for row, vector in zip(demo_rows, vectors)
    ]


def context(**overrides: Any) -> dict:
    """A discount-request context with sensible defaults."""
    payload = {
        "customer_segment": "Enterprise",
        "requested_discount_pct": 18.0,
        "deal_value_eur": 240_000.0,
        "contract_months": 12,
        "region": "DE",
        "renewal_status": "renewal",
        "competitor_in_deal": True,
    }
    payload.update(overrides)
    return payload


def precedent(
    identifier: str,
    *,
    days_ago: int = 30,
    decision_type: str = "sales_discount",
    chosen: str | None = "approve_as_requested",
    outcome: str | None = None,
    metrics: dict | None = None,
    embedding: list[float] | None = None,
    **context_overrides: Any,
) -> PrecedentContext:
    return PrecedentContext(
        id=identifier,
        title=identifier,
        decision_type=decision_type,
        template_id="t",
        context_structured=context(**context_overrides),
        chosen_option=chosen,
        decided_at=NOW - timedelta(days=days_ago),
        embedding=embedding,
        outcome_success=outcome,
        outcome_metrics=metrics or {},
    )
