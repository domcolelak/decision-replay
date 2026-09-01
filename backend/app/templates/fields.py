"""Decision templates and structured field similarity.

A template gives comparable decisions a comparable shape. Without it, "similar"
degrades into "the text sounds alike", which is exactly the failure mode this
product exists to avoid: two discount requests can be worded identically and be
nothing like each other once you look at the numbers.

Each field declares a type and a similarity weight, so the business decides what
makes two cases comparable -- deal value may matter far more than country.

The single most consequential decision here is how a **missing** field is
treated: it removes its own weight from the calculation rather than scoring
zero. Scoring it zero would systematically rank sparsely-filled historical
records as dissimilar to everything, which says more about data entry than
about the decisions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Sequence


class FieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"
    DATE = "date"


@dataclass
class TemplateField:
    name: str
    label: str
    type: FieldType
    weight: float = 1.0
    required: bool = False
    options: list[str] = field(default_factory=list)
    #: For numbers: the difference at which similarity reaches ~0.5. Without a
    #: scale, "is 7,200 close to 2,500?" has no defensible answer.
    tolerance: float | None = None
    unit: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type.value,
            "weight": self.weight,
            "required": self.required,
            "options": self.options,
            "tolerance": self.tolerance,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TemplateField:
        return cls(
            name=payload["name"],
            label=payload.get("label", payload["name"]),
            type=FieldType(payload.get("type", "string")),
            weight=float(payload.get("weight", 1.0)),
            required=bool(payload.get("required", False)),
            options=list(payload.get("options", [])),
            tolerance=payload.get("tolerance"),
            unit=payload.get("unit", ""),
        )

    def similarity(self, left: Any, right: Any) -> float | None:
        """0..1, or ``None`` when either side is missing.

        ``None`` means "cannot judge", which is different from 0 ("judged, not
        similar"). The caller drops the weight rather than counting a zero.
        """
        if _is_missing(left) or _is_missing(right):
            return None

        if self.type is FieldType.NUMBER:
            return _number_similarity(left, right, self.tolerance)
        if self.type is FieldType.DATE:
            return _date_similarity(left, right)
        if self.type is FieldType.BOOLEAN:
            return 1.0 if _as_bool(left) == _as_bool(right) else 0.0
        # Enum and string are compared as exact categories. Fuzzy string
        # matching belongs to the semantic component, not here -- mixing the
        # two would double-count the same signal.
        return 1.0 if _normalise(left) == _normalise(right) else 0.0


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _normalise(value: Any) -> str:
    return str(value).strip().lower()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalise(value) in ("true", "yes", "y", "1")


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _number_similarity(left: Any, right: Any, tolerance: float | None) -> float:
    """Closeness on a scale the template chose.

    Falls back to a relative comparison when no tolerance is configured, so a
    pair of 100,000s is as close as a pair of 100s -- an absolute difference
    would make every large-value pair look dissimilar simply because the
    numbers are big.
    """
    a, b = _as_number(left), _as_number(right)
    if a is None or b is None:
        return 0.0
    difference = abs(a - b)
    if difference == 0:
        return 1.0

    if tolerance and tolerance > 0:
        return float(1.0 / (1.0 + difference / tolerance))

    scale = max(abs(a), abs(b))
    if scale == 0:
        return 1.0
    return float(max(0.0, 1.0 - difference / scale))


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


#: Difference at which two dates are half as similar.
DATE_HALF_LIFE_DAYS = 180.0


def _date_similarity(left: Any, right: Any) -> float:
    a, b = _to_datetime(left), _to_datetime(right)
    if a is None or b is None:
        return 0.0
    days = abs((a - b).total_seconds()) / 86400.0
    return float(math.exp(-days / DATE_HALF_LIFE_DAYS * math.log(2)))


@dataclass
class DecisionTemplate:
    """A named shape for one class of decision."""

    name: str
    decision_type: str
    fields: list[TemplateField] = field(default_factory=list)
    #: Component weights for the hybrid ranking, overriding the defaults.
    ranking_weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "decision_type": self.decision_type,
            "fields": [f.as_dict() for f in self.fields],
            "ranking_weights": self.ranking_weights,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DecisionTemplate:
        return cls(
            name=payload["name"],
            decision_type=payload.get("decision_type", payload["name"]),
            fields=[TemplateField.from_dict(f) for f in payload.get("fields", [])],
            ranking_weights=dict(payload.get("ranking_weights", {})),
        )

    def field_by_name(self, name: str) -> TemplateField | None:
        for spec in self.fields:
            if spec.name == name:
                return spec
        return None

    def validate(self, context: dict[str, Any]) -> list[str]:
        """Problems with a structured context, as human-readable strings."""
        problems: list[str] = []
        for spec in self.fields:
            value = context.get(spec.name)
            if _is_missing(value):
                if spec.required:
                    problems.append(f"'{spec.label}' is required")
                continue
            if spec.type is FieldType.NUMBER and _as_number(value) is None:
                problems.append(f"'{spec.label}' must be a number, got {value!r}")
            elif spec.type is FieldType.DATE and _to_datetime(value) is None:
                problems.append(f"'{spec.label}' must be a date, got {value!r}")
            elif spec.type is FieldType.ENUM and spec.options:
                if _normalise(value) not in {_normalise(o) for o in spec.options}:
                    problems.append(
                        f"'{spec.label}' must be one of {', '.join(spec.options)}, "
                        f"got {value!r}"
                    )
        return problems


@dataclass
class FieldContribution:
    """One field's part in a structured similarity score."""

    field: str
    label: str
    weight: float
    similarity: float
    left: Any
    right: Any

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "label": self.label,
            "weight": round(self.weight, 4),
            "similarity": round(self.similarity, 4),
            "left": self.left,
            "right": self.right,
        }


@dataclass
class StructuredSimilarity:
    score: float
    contributions: list[FieldContribution] = field(default_factory=list)
    #: Fields that could not be compared because one side was empty.
    skipped: list[str] = field(default_factory=list)
    comparable_weight: float = 0.0

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "contributions": [c.as_dict() for c in self.contributions],
            "skipped": self.skipped,
            "comparable_weight": round(self.comparable_weight, 4),
        }


def structured_similarity(
    template: DecisionTemplate, left: dict[str, Any], right: dict[str, Any]
) -> StructuredSimilarity:
    """Weighted field-by-field similarity between two structured contexts."""
    contributions: list[FieldContribution] = []
    skipped: list[str] = []
    total_weight = 0.0
    weighted = 0.0

    for spec in template.fields:
        if spec.weight <= 0:
            continue
        similarity = spec.similarity(left.get(spec.name), right.get(spec.name))
        if similarity is None:
            skipped.append(spec.name)
            continue
        contributions.append(
            FieldContribution(
                field=spec.name,
                label=spec.label,
                weight=spec.weight,
                similarity=similarity,
                left=left.get(spec.name),
                right=right.get(spec.name),
            )
        )
        total_weight += spec.weight
        weighted += spec.weight * similarity

    contributions.sort(key=lambda c: -(c.weight * c.similarity))
    return StructuredSimilarity(
        score=weighted / total_weight if total_weight > 0 else 0.0,
        contributions=contributions,
        skipped=skipped,
        comparable_weight=total_weight,
    )


def coverage(template: DecisionTemplate, context: dict[str, Any]) -> float:
    """Share of the template's weight this context actually fills in.

    Surfaced alongside a similarity score so a high score computed from two
    fields is not mistaken for a high score computed from ten.
    """
    total = sum(f.weight for f in template.fields if f.weight > 0)
    if total <= 0:
        return 0.0
    present = sum(
        f.weight
        for f in template.fields
        if f.weight > 0 and not _is_missing(context.get(f.name))
    )
    return round(present / total, 4)
