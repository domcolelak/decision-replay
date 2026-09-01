"""Side-by-side comparison of decisions, and aggregate precedent statistics.

Two jobs:

* build a comparison table where each row is a field and each column a
  decision, so differences are visible rather than described;
* summarise a set of precedents into the numbers a decision-maker actually
  asks for -- what was chosen, how often, and how it turned out.

The summary is where causal overclaiming would creep in, so it is built to
resist it. Outcome rates are computed over decisions that *have* an outcome,
and the count without one is reported alongside. "We approved 12 and 9 went
well" is a different claim from "we approved 12, 4 have no outcome recorded,
and 5 of the remaining 8 went well" -- the second is the true one.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from app.templates.fields import DecisionTemplate, FieldType

#: Outcome labels, worst to best, for ordering in summaries.
OUTCOME_ORDER = ("failure", "mixed", "success", "unknown")

OBSERVATIONAL_NOTE = (
    "These figures describe what happened after similar past decisions. They are "
    "historical association, not a prediction: the situations differed in ways "
    "the recorded fields do not capture."
)


@dataclass
class ComparisonCell:
    decision_id: str
    value: Any
    #: True when this value differs from the majority in the row.
    differs: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonRow:
    field: str
    label: str
    kind: str
    cells: list[ComparisonCell] = field(default_factory=list)
    #: True when the row's values are not all the same.
    varies: bool = False

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "label": self.label,
            "kind": self.kind,
            "varies": self.varies,
            "cells": [c.as_dict() for c in self.cells],
        }


@dataclass
class ComparisonTable:
    decision_ids: list[str] = field(default_factory=list)
    titles: dict[str, str] = field(default_factory=dict)
    rows: list[ComparisonRow] = field(default_factory=list)
    note: str = OBSERVATIONAL_NOTE

    def as_dict(self) -> dict:
        return {
            "decision_ids": self.decision_ids,
            "titles": self.titles,
            "rows": [r.as_dict() for r in self.rows],
            "note": self.note,
        }


@dataclass
class DecisionView:
    """The subset of a decision that comparison and summary need."""

    id: str
    title: str
    decision_type: str
    context_structured: dict[str, Any]
    chosen_option: str | None = None
    rationale: str = ""
    decided_at: Any = None
    outcome_label: str | None = None
    outcome_metrics: dict[str, float] = field(default_factory=dict)
    outcome_notes: str = ""
    retrospective: str = ""
    owner: str = ""


def build_comparison(
    decisions: Sequence[DecisionView], template: DecisionTemplate | None = None
) -> ComparisonTable:
    """A field-by-field table across 2-10 decisions."""
    table = ComparisonTable(
        decision_ids=[d.id for d in decisions],
        titles={d.id: d.title for d in decisions},
    )
    if not decisions:
        return table

    # Template fields first, in the order the template defines, then anything
    # else present in the data so nothing is silently hidden.
    ordered: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    if template:
        for spec in template.fields:
            ordered.append((spec.name, spec.label, spec.type.value))
            seen.add(spec.name)
    extra = sorted(
        {key for d in decisions for key in d.context_structured} - seen
    )
    ordered += [(key, key, "string") for key in extra]

    for name, label, kind in ordered:
        row = ComparisonRow(field=name, label=label, kind=kind)
        values = [d.context_structured.get(name) for d in decisions]
        row.cells = [
            ComparisonCell(decision_id=d.id, value=v) for d, v in zip(decisions, values)
        ]
        _mark_variation(row, values)
        table.rows.append(row)

    for name, label, getter in (
        ("chosen_option", "Decision taken", lambda d: d.chosen_option),
        ("rationale", "Rationale", lambda d: d.rationale),
        ("outcome_label", "Outcome", lambda d: d.outcome_label or "not recorded"),
        ("retrospective", "Retrospective", lambda d: d.retrospective or ""),
        ("owner", "Owner", lambda d: d.owner),
    ):
        values = [getter(d) for d in decisions]
        row = ComparisonRow(field=name, label=label, kind="string")
        row.cells = [
            ComparisonCell(decision_id=d.id, value=v) for d, v in zip(decisions, values)
        ]
        _mark_variation(row, values)
        table.rows.append(row)

    metric_names = sorted({k for d in decisions for k in d.outcome_metrics})
    for metric in metric_names:
        values = [d.outcome_metrics.get(metric) for d in decisions]
        row = ComparisonRow(field=f"metric:{metric}", label=metric, kind="number")
        row.cells = [
            ComparisonCell(decision_id=d.id, value=v) for d, v in zip(decisions, values)
        ]
        _mark_variation(row, values)
        table.rows.append(row)

    return table


def _mark_variation(row: ComparisonRow, values: Sequence[Any]) -> None:
    """Flag the cells that stand out, so a reader's eye goes to the difference."""
    comparable = [_key(v) for v in values]
    row.varies = len(set(comparable)) > 1
    if not row.varies:
        return
    majority, count = Counter(comparable).most_common(1)[0]
    # With no majority every value is equally unusual, and highlighting all of
    # them highlights nothing.
    if count <= 1:
        return
    for cell, key in zip(row.cells, comparable):
        cell.differs = key != majority


def _key(value: Any) -> str:
    if value is None:
        return "\x00none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).strip().lower()


@dataclass
class OptionStats:
    option: str
    count: int
    share: float
    outcomes: dict[str, int] = field(default_factory=dict)
    #: Decisions that have an outcome recorded at all.
    with_outcome: int = 0
    mean_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float | None:
        """Successes among decisions with a recorded outcome, or ``None``.

        Returns ``None`` rather than 0.0 when nothing is known, so the caller
        cannot accidentally render "0% success" for "we never followed up".
        """
        if self.with_outcome == 0:
            return None
        return round(self.outcomes.get("success", 0) / self.with_outcome, 4)

    def as_dict(self) -> dict:
        return {
            "option": self.option,
            "count": self.count,
            "share": round(self.share, 4),
            "outcomes": self.outcomes,
            "with_outcome": self.with_outcome,
            "without_outcome": self.count - self.with_outcome,
            "success_rate": self.success_rate,
            "mean_metrics": {k: round(v, 4) for k, v in self.mean_metrics.items()},
        }


@dataclass
class PrecedentSummary:
    total: int
    with_outcome: int
    options: list[OptionStats] = field(default_factory=list)
    note: str = OBSERVATIONAL_NOTE
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "with_outcome": self.with_outcome,
            "without_outcome": self.total - self.with_outcome,
            "options": [o.as_dict() for o in self.options],
            "note": self.note,
            "caveats": self.caveats,
        }


#: Below this many decisions, a rate is not worth quoting.
MIN_FOR_RATE = 5


def summarise_precedents(decisions: Sequence[DecisionView]) -> PrecedentSummary:
    """Aggregate what was chosen and how it turned out."""
    total = len(decisions)
    summary = PrecedentSummary(total=total, with_outcome=0)
    if total == 0:
        summary.caveats.append("No comparable decisions were found.")
        return summary

    grouped: defaultdict[str, list[DecisionView]] = defaultdict(list)
    for decision in decisions:
        grouped[decision.chosen_option or "not recorded"].append(decision)

    for option, group in grouped.items():
        outcomes = Counter(
            d.outcome_label for d in group if d.outcome_label and d.outcome_label != "unknown"
        )
        with_outcome = sum(outcomes.values())
        summary.with_outcome += with_outcome

        metrics: defaultdict[str, list[float]] = defaultdict(list)
        for decision in group:
            for name, value in decision.outcome_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[name].append(float(value))

        summary.options.append(
            OptionStats(
                option=option,
                count=len(group),
                share=len(group) / total,
                outcomes=dict(outcomes),
                with_outcome=with_outcome,
                mean_metrics={k: sum(v) / len(v) for k, v in metrics.items() if v},
            )
        )

    summary.options.sort(key=lambda o: -o.count)

    if total < MIN_FOR_RATE:
        summary.caveats.append(
            f"Only {total} comparable decision(s); too few to read rates from."
        )
    missing = total - summary.with_outcome
    if missing:
        summary.caveats.append(
            f"{missing} of {total} have no recorded outcome. Rates below are computed "
            f"only over the {summary.with_outcome} that do."
        )
    thin = [o.option for o in summary.options if 0 < o.with_outcome < MIN_FOR_RATE]
    if thin:
        summary.caveats.append(
            "Outcome rates for " + ", ".join(sorted(thin)) + " rest on very few cases."
        )
    return summary
