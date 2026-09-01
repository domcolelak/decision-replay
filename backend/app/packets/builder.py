"""Decision packet generation.

A packet is the auditable record of how a decision was reached: the situation,
the options, what history showed, what was decided and why. It is rendered from
computed values; the AI summary, when available, is a clearly separated section.

The point of the packet is that six months later somebody can see not just what
was chosen but what was known at the time — including what was not known.
"""
from __future__ import annotations

from typing import Any, Sequence

from app.comparison.compare import PrecedentSummary
from app.search.ranking import SearchResult


def build_payload(
    *,
    decision: dict,
    search: SearchResult,
    summary: PrecedentSummary,
    ai_summary: dict | None = None,
    withheld: int = 0,
) -> dict:
    return {
        "decision": decision,
        "precedents": [p.as_dict() for p in search.precedents],
        "ranking": {
            "weights_used": search.weights_used,
            "semantic_available": search.semantic_available,
            "candidates_considered": search.candidates_considered,
            "note": search.note,
        },
        "statistics": summary.as_dict(),
        "withheld_for_confidentiality": withheld,
        "ai_summary": ai_summary,
    }


def render_markdown(payload: dict) -> str:
    """Render the packet as Markdown."""
    decision = payload["decision"]
    statistics = payload["statistics"]
    ranking = payload["ranking"]

    lines: list[str] = [
        f"# {decision.get('title', 'Decision')}",
        "",
        "## Situation",
        "",
        decision.get("context_text") or "_No narrative context recorded._",
        "",
    ]

    structured = decision.get("context_structured") or {}
    if structured:
        lines += ["| Field | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(structured.items())]
        lines.append("")

    options = decision.get("options") or []
    if options:
        lines += ["## Options considered", ""]
        for option in options:
            marker = " **(chosen)**" if option.get("key") == decision.get("chosen_option") else ""
            lines.append(f"- **{option.get('label', option.get('key'))}**{marker}")
            if option.get("notes"):
                lines.append(f"  - {option['notes']}")
        lines.append("")

    lines += ["## What history shows", ""]
    if statistics["total"] == 0:
        lines += ["No comparable decisions were found.", ""]
    else:
        lines += [
            f"{statistics['total']} comparable decision(s). "
            f"{statistics['with_outcome']} have a recorded outcome; "
            f"{statistics['without_outcome']} do not.",
            "",
            "| Option | Times chosen | Share | With outcome | Success rate |",
            "|---|---|---|---|---|",
        ]
        for option in statistics["options"]:
            rate = option["success_rate"]
            # "not known" is not "0%". Rendering an unknown as a number is how
            # a packet ends up asserting something nobody ever measured.
            rate_text = "not known" if rate is None else f"{rate:.0%}"
            lines.append(
                f"| {option['option']} | {option['count']} | {option['share']:.0%} | "
                f"{option['with_outcome']} | {rate_text} |"
            )
        lines.append("")

        if statistics["caveats"]:
            lines += ["**Caveats:**", ""]
            lines += [f"- {c}" for c in statistics["caveats"]]
            lines.append("")

    precedents = payload.get("precedents") or []
    if precedents:
        lines += ["## Closest precedents", "", "| Decision | Similarity | Chosen | Outcome |", "|---|---|---|---|"]
        for precedent in precedents[:10]:
            outcome = precedent.get("outcome_success") or "not recorded"
            lines.append(
                f"| {precedent['title']} | {precedent['score']:.2f} | "
                f"{precedent.get('chosen_option') or '-'} | {outcome} |"
            )
        lines.append("")

    if payload.get("withheld_for_confidentiality"):
        lines += [
            f"> {payload['withheld_for_confidentiality']} restricted decision(s) were "
            f"excluded from this packet.",
            "",
        ]

    if decision.get("chosen_option"):
        lines += [
            "## Decision taken",
            "",
            f"**{decision['chosen_option']}**",
            "",
            decision.get("rationale") or "_No rationale recorded._",
            "",
        ]
    else:
        lines += ["## Decision taken", "", "_Not yet decided._", ""]

    ai_summary = payload.get("ai_summary")
    if ai_summary:
        lines += ["## Narrative summary", "", ai_summary.get("summary", ""), ""]
        claims = ai_summary.get("claims") or []
        if claims:
            lines += ["| Claim | Basis | Kind |", "|---|---|---|"]
            for claim in claims:
                lines.append(
                    f"| {claim.get('statement', '')} | {claim.get('basis', '')} | "
                    f"{claim.get('kind', '')} |"
                )
            lines.append("")
        for label, key in (("Open questions", "open_questions"), ("Caveats", "caveats")):
            values = ai_summary.get(key) or []
            if values:
                lines += [f"**{label}:**", ""] + [f"- {v}" for v in values] + [""]

    lines += [
        "---",
        "",
        ranking["note"],
        "",
        f"Ranking weights: "
        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(ranking["weights_used"].items()))
        + f". Semantic similarity {'was' if ranking['semantic_available'] else 'was not'} "
        f"available; {ranking['candidates_considered']} candidate decision(s) were considered.",
    ]
    return "\n".join(lines)
