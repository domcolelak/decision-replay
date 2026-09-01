"""Pydantic request/response models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONFIDENTIALITY = Literal["internal", "confidential", "restricted"]
OUTCOME_LABEL = Literal["success", "mixed", "failure", "unknown"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- templates -------------------------------------------------------------


class TemplateFieldIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    label: str = ""
    type: Literal["string", "number", "boolean", "enum", "date"] = "string"
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    required: bool = False
    options: list[str] = Field(default_factory=list)
    tolerance: float | None = None
    unit: str = ""


class TemplateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    decision_type: str = Field(min_length=1, max_length=120)
    description: str = ""
    fields: list[TemplateFieldIn] = Field(min_length=1)
    ranking_weights: dict[str, float] = Field(default_factory=dict)


class TemplateOut(ORMModel):
    id: uuid.UUID
    name: str
    decision_type: str
    description: str
    fields: list[Any]
    ranking_weights: dict[str, float]
    created_at: datetime


class TemplateListItem(TemplateOut):
    decision_count: int = 0


# --- decisions -------------------------------------------------------------


class OptionIn(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    label: str = ""
    notes: str = ""


class EvidenceIn(BaseModel):
    kind: str = "note"
    summary: str
    url: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class DecisionCreate(BaseModel):
    title: str = Field(min_length=1)
    decision_type: str = Field(min_length=1, max_length=120)
    template_id: uuid.UUID | None = None
    external_id: str | None = None
    context_text: str = ""
    context_structured: dict[str, Any] = Field(default_factory=dict)
    options: list[OptionIn] = Field(default_factory=list)
    evidence: list[EvidenceIn] = Field(default_factory=list)
    chosen_option: str | None = None
    rationale: str = ""
    owner: str = ""
    stakeholders: list[str] = Field(default_factory=list)
    decided_at: datetime | None = None
    expected_outcome: dict[str, Any] = Field(default_factory=dict)
    outcome_due_at: datetime | None = None
    confidentiality: CONFIDENTIALITY = "internal"
    tags: list[str] = Field(default_factory=list)
    source_links: list[str] = Field(default_factory=list)


class DecisionUpdate(BaseModel):
    title: str | None = None
    context_text: str | None = None
    context_structured: dict[str, Any] | None = None
    chosen_option: str | None = None
    rationale: str | None = None
    decided_at: datetime | None = None
    outcome_due_at: datetime | None = None
    confidentiality: CONFIDENTIALITY | None = None
    tags: list[str] | None = None


class DecisionOut(ORMModel):
    id: uuid.UUID
    template_id: uuid.UUID | None
    external_id: str | None
    title: str
    decision_type: str
    context_text: str
    context_structured: dict[str, Any]
    chosen_option: str | None
    rationale: str
    owner: str
    stakeholders: list[Any]
    decided_at: datetime | None
    expected_outcome: dict[str, Any]
    outcome_due_at: datetime | None
    confidentiality: str
    tags: list[Any]
    created_at: datetime
    updated_at: datetime


class DecisionDetail(DecisionOut):
    options: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    outcome: dict[str, Any] | None = None
    #: Share of the template's weight this context actually fills in.
    context_coverage: float = 0.0
    validation_problems: list[str] = Field(default_factory=list)
    embedding: dict[str, Any] | None = None


# --- outcomes --------------------------------------------------------------


class OutcomeIn(BaseModel):
    success_label: OUTCOME_LABEL = "unknown"
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
    retrospective: str = ""
    recorded_by: str = ""


class OutcomeOut(ORMModel):
    id: uuid.UUID
    decision_id: uuid.UUID
    success_label: str
    metrics: dict[str, float]
    notes: str
    retrospective: str
    recorded_by: str
    recorded_at: datetime


class OverdueOutcome(BaseModel):
    decision_id: uuid.UUID
    title: str
    owner: str
    decided_at: datetime | None
    outcome_due_at: datetime | None
    days_overdue: int


# --- search and comparison -------------------------------------------------


class SearchRequest(BaseModel):
    """Search either from a saved decision, or from a situation not yet saved."""

    decision_id: uuid.UUID | None = None
    template_id: uuid.UUID | None = None
    decision_type: str | None = None
    title: str = ""
    context_text: str = ""
    context_structured: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)
    include_summary: bool = True


class SearchResponse(BaseModel):
    precedents: list[dict[str, Any]] = Field(default_factory=list)
    weights_used: dict[str, float] = Field(default_factory=dict)
    semantic_available: bool = False
    candidates_considered: int = 0
    statistics: dict[str, Any] | None = None
    note: str


class CompareRequest(BaseModel):
    decision_ids: list[uuid.UUID] = Field(min_length=2, max_length=10)
    template_id: uuid.UUID | None = None


class CompareResponse(BaseModel):
    table: dict[str, Any]
    statistics: dict[str, Any]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    template_id: uuid.UUID


class SummariseRequest(BaseModel):
    decision_id: uuid.UUID | None = None
    title: str = ""
    context_text: str = ""
    context_structured: dict[str, Any] = Field(default_factory=dict)
    decision_type: str | None = None
    template_id: uuid.UUID | None = None
    limit: int = Field(default=10, ge=1, le=50)


class PacketRequest(BaseModel):
    decision_id: uuid.UUID
    limit: int = Field(default=10, ge=1, le=50)
    include_ai_summary: bool = True


class PacketOut(ORMModel):
    id: uuid.UUID
    decision_id: uuid.UUID
    title: str
    body: str
    payload: dict[str, Any]
    created_at: datetime


class OverviewResponse(BaseModel):
    template_count: int
    decision_count: int
    decided_count: int
    with_outcome: int
    overdue_outcomes: int
    embedding_coverage: float
    embedding_model: str
    outcome_mix: dict[str, int] = Field(default_factory=dict)
    recent_decisions: list[DecisionOut] = Field(default_factory=list)
