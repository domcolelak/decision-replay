"""SQLAlchemy models.

Every tenant-owned table carries ``tenant_id`` and indexes it first.

Two shapes are worth noting:

* **Embeddings are a separate table** carrying the model name and version. A
  vector is only comparable to vectors from the same model, so storing the
  model alongside is not bookkeeping -- it is what stops a model upgrade from
  silently producing confident nonsense.
* **Outcomes are separate from decisions**, and nullable. A decision without an
  outcome is memory; a decision with one is evidence. Conflating "no outcome
  recorded" with "it went fine" would quietly bias every precedent summary.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base, GUID


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    api_key_hash: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(320))
    display_name: Mapped[str] = mapped_column(String(200), default="")
    role: Mapped[str] = mapped_column(String(32), default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_user_email_per_tenant"),)


class DecisionTemplateRow(Base):
    __tablename__ = "decision_templates"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    decision_type: Mapped[str] = mapped_column(String(120), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    fields: Mapped[list] = mapped_column(JSON, default=list)
    ranking_weights: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_template_name_per_tenant"),)


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID, ForeignKey("decision_templates.id"), nullable=True, index=True
    )
    external_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    decision_type: Mapped[str] = mapped_column(String(120), index=True)
    context_text: Mapped[str] = mapped_column(Text, default="")
    context_structured: Mapped[dict] = mapped_column(JSON, default=dict)
    chosen_option: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rationale: Mapped[str] = mapped_column(Text, default="")
    owner: Mapped[str] = mapped_column(String(200), default="")
    stakeholders: Mapped[list] = mapped_column(JSON, default=list)
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    expected_outcome: Mapped[dict] = mapped_column(JSON, default=dict)
    #: internal | confidential | restricted. Restricted decisions never leave
    #: the tenant and are never sent to an AI provider.
    confidentiality: Mapped[str] = mapped_column(String(32), default="internal", index=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    source_links: Mapped[list] = mapped_column(JSON, default=list)
    #: Due date for recording what actually happened.
    outcome_due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    options: Mapped[list["DecisionOption"]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )
    evidence: Mapped[list["DecisionEvidence"]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )
    outcome: Mapped["Outcome | None"] = relationship(
        back_populates="decision", cascade="all, delete-orphan", uselist=False
    )
    embedding: Mapped["DecisionEmbedding | None"] = relationship(
        back_populates="decision", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        Index("ix_decision_tenant_type", "tenant_id", "decision_type"),
        UniqueConstraint("tenant_id", "external_id", name="uq_decision_external_id"),
    )


class DecisionOption(Base):
    __tablename__ = "decision_options"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("decisions.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(200))
    label: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text, default="")
    position: Mapped[int] = mapped_column(Integer, default=0)

    decision: Mapped[Decision] = relationship(back_populates="options")


class DecisionEvidence(Base):
    __tablename__ = "decision_evidence"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("decisions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(64), default="note")
    summary: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(1000), default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    decision: Mapped[Decision] = relationship(back_populates="evidence")


class DecisionEmbedding(Base):
    """A context vector, tied to the model that produced it."""

    __tablename__ = "decision_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("decisions.id", ondelete="CASCADE"), unique=True, index=True
    )
    model: Mapped[str] = mapped_column(String(120), index=True)
    model_version: Mapped[str] = mapped_column(String(32), default="v1")
    dimensions: Mapped[int] = mapped_column(Integer, default=0)
    #: Stored as JSON so the product runs on SQLite and on Postgres without
    #: pgvector. A production deployment swaps this for a vector column.
    vector: Mapped[list] = mapped_column(JSON, default=list)
    #: Hash of the text embedded, so a stale vector can be detected cheaply.
    source_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    decision: Mapped[Decision] = relationship(back_populates="embedding")


class Outcome(Base):
    """What actually happened. Nullable on purpose."""

    __tablename__ = "outcomes"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("decisions.id", ondelete="CASCADE"), unique=True, index=True
    )
    #: success | mixed | failure | unknown
    success_label: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    retrospective: Mapped[str] = mapped_column(Text, default="")
    recorded_by: Mapped[str] = mapped_column(String(200), default="")
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    decision: Mapped[Decision] = relationship(back_populates="outcome")


class ComparisonSession(Base):
    __tablename__ = "comparison_sessions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_ids: Mapped[list] = mapped_column(JSON, default=list)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SearchLog(Base):
    """Every precedent search, for auditability and for tuning weights later."""

    __tablename__ = "searches"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    query: Mapped[dict] = mapped_column(JSON, default=dict)
    weights_used: Mapped[dict] = mapped_column(JSON, default=dict)
    semantic_available: Mapped[bool] = mapped_column(Boolean, default=False)
    result_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class DecisionPacket(Base):
    __tablename__ = "decision_packets"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    decision_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("decisions.id"), index=True)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    actor: Mapped[str] = mapped_column(String(200), default="system")
    action: Mapped[str] = mapped_column(String(120), index=True)
    object_type: Mapped[str] = mapped_column(String(64), default="")
    object_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class AILogEntry(Base):
    __tablename__ = "ai_log"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID, ForeignKey("tenants.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(120), default="")
    prompt_version: Mapped[str] = mapped_column(String(64), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    #: How many decisions were withheld from the model for confidentiality.
    redacted_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
