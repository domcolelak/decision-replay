"""Deterministic demo dataset: B2B sales discount approvals.

Generates historical decisions with structured context, a chosen option, a
rationale and — for most of them — a recorded outcome.

The generator follows consistent (but unstated) practice, so precedent
retrieval has something real to find:

* Enterprise deals tolerate deeper discounts before escalation.
* Discounts above 20% were historically rejected outright for SMB.
* Long contracts buy discount room.
* One region (IT) approved far more aggressively than the rest.
* Margin outcomes fall as the discount rises, and win rates rise.

About a third of the decisions have no outcome yet — a realistic and important
case, since the product must not treat "we don't know yet" as "it went fine".
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

SEGMENTS = ("SMB", "MidMarket", "Enterprise")
REGIONS = ("SK", "CZ", "DE", "AT", "IT")
RENEWAL = ("new", "renewal", "expansion")

OPTIONS = ("approve_as_requested", "approve_reduced", "reject", "escalate")

START = datetime(2024, 1, 8, tzinfo=timezone.utc)

#: The template a customer would configure. Weights say what makes two discount
#: requests comparable: the ask and the segment matter far more than the region.
DISCOUNT_TEMPLATE: dict[str, Any] = {
    "name": "Sales discount request",
    "decision_type": "sales_discount",
    "fields": [
        {
            "name": "customer_segment",
            "label": "Customer segment",
            "type": "enum",
            "options": list(SEGMENTS),
            "weight": 2.5,
            "required": True,
        },
        {
            "name": "requested_discount_pct",
            "label": "Requested discount",
            "type": "number",
            "weight": 3.0,
            "tolerance": 4.0,
            "unit": "%",
            "required": True,
        },
        {
            "name": "deal_value_eur",
            "label": "Deal value",
            "type": "number",
            "weight": 2.0,
            "tolerance": 25_000.0,
            "unit": "EUR",
            "required": True,
        },
        {
            "name": "contract_months",
            "label": "Contract length",
            "type": "number",
            "weight": 1.5,
            "tolerance": 6.0,
            "unit": "months",
        },
        {
            "name": "region",
            "label": "Region",
            "type": "enum",
            "options": list(REGIONS),
            "weight": 1.0,
        },
        {
            "name": "renewal_status",
            "label": "Renewal status",
            "type": "enum",
            "options": list(RENEWAL),
            "weight": 1.0,
        },
        {
            "name": "competitor_in_deal",
            "label": "Competitor in the deal",
            "type": "boolean",
            "weight": 1.0,
        },
    ],
    "ranking_weights": {
        "structured": 0.45,
        "semantic": 0.35,
        "type": 0.10,
        "recency": 0.10,
    },
}


def _choose(features: dict[str, Any], rng: random.Random) -> str:
    """The unstated practice the product should surface as precedent."""
    segment = features["customer_segment"]
    discount = features["requested_discount_pct"]
    months = features["contract_months"]

    # Long contracts buy roughly five points of discount room.
    effective = discount - (5 if months >= 24 else 0)

    if features["region"] == "IT":
        return "approve_as_requested" if effective <= 28 else "approve_reduced"

    if segment == "Enterprise":
        if effective <= 20:
            return "approve_as_requested"
        return "approve_reduced" if effective <= 30 else "escalate"

    if segment == "MidMarket":
        if effective <= 15:
            return "approve_as_requested"
        return "approve_reduced" if effective <= 25 else "reject"

    # SMB
    if effective <= 10:
        return "approve_as_requested"
    if effective <= 20:
        return "approve_reduced"
    return "reject"


RATIONALES = {
    "approve_as_requested": [
        "Within the standard band for this segment; no escalation needed.",
        "Discount is normal for a deal of this size and length.",
        "Approved to close before quarter end; margin stays acceptable.",
    ],
    "approve_reduced": [
        "Approved at a lower level than requested to protect margin.",
        "Counter-offered a smaller discount tied to a longer commitment.",
        "Partial approval; the full ask was outside the usual range.",
    ],
    "reject": [
        "Ask is far outside anything approved for this segment.",
        "Margin impact could not be justified for a deal this size.",
        "Declined; the customer had no competing offer on the table.",
    ],
    "escalate": [
        "Referred upward: unusually large discount on a strategic account.",
        "Escalated for a second opinion given the contract value.",
    ],
}


def generate_decisions(*, count: int = 90, seed: int = 20260830) -> list[dict[str, Any]]:
    """Build the synthetic decision history. Pure function, no database."""
    rng = random.Random(seed)
    decisions: list[dict[str, Any]] = []

    for index in range(count):
        decided_at = START + timedelta(days=7 * index + rng.randint(0, 5))
        segment = rng.choices(SEGMENTS, weights=[5, 3, 2])[0]
        region = rng.choices(REGIONS, weights=[4, 3, 3, 2, 2])[0]
        months = rng.choice([12, 12, 24, 36, 6])
        discount = round(
            rng.choices(
                [rng.uniform(2, 10), rng.uniform(10, 20), rng.uniform(20, 35)],
                weights=[4, 4, 2],
            )[0],
            1,
        )
        value = round(
            {
                "SMB": rng.uniform(3_000, 30_000),
                "MidMarket": rng.uniform(25_000, 120_000),
                "Enterprise": rng.uniform(90_000, 600_000),
            }[segment],
            2,
        )

        features = {
            "customer_segment": segment,
            "requested_discount_pct": discount,
            "deal_value_eur": value,
            "contract_months": months,
            "region": region,
            "renewal_status": rng.choices(RENEWAL, weights=[4, 4, 2])[0],
            "competitor_in_deal": rng.random() < 0.45,
        }

        chosen = _choose(features, rng)
        # Real approvers are not perfectly consistent.
        if rng.random() < 0.07:
            chosen = rng.choice(OPTIONS)

        decisions.append(
            {
                "external_id": f"DEC-{1000 + index}",
                "title": (
                    f"{segment} {features['renewal_status']} deal, "
                    f"{discount:.0f}% discount on {value:,.0f} EUR"
                ),
                "decision_type": "sales_discount",
                "context_text": _context_text(features),
                "context_structured": features,
                "options": _options_for(features),
                "chosen_option": chosen,
                "rationale": rng.choice(RATIONALES[chosen]),
                "owner": f"ae_{rng.randint(1, 7)}",
                "decided_at": decided_at,
                "tags": [segment.lower(), region.lower(), features["renewal_status"]],
                "outcome": _outcome(rng, features, chosen, decided_at),
            }
        )

    return decisions


def _context_text(features: dict[str, Any]) -> str:
    competitor = (
        "A competitor is actively in the deal."
        if features["competitor_in_deal"]
        else "No competing vendor has been mentioned."
    )
    return (
        f"A {features['customer_segment']} customer in {features['region']} is asking for "
        f"a {features['requested_discount_pct']:.1f}% discount on a "
        f"{features['deal_value_eur']:,.0f} EUR {features['renewal_status']} deal with a "
        f"{features['contract_months']} month contract. {competitor}"
    )


def _options_for(features: dict[str, Any]) -> list[dict[str, Any]]:
    asked = features["requested_discount_pct"]
    return [
        {
            "key": "approve_as_requested",
            "label": f"Approve the full {asked:.1f}%",
            "notes": "Fastest path to signature; largest margin impact.",
        },
        {
            "key": "approve_reduced",
            "label": f"Counter at {max(asked - 7, 2):.1f}%",
            "notes": "Protects margin; risks a slower close.",
        },
        {"key": "reject", "label": "Decline the discount", "notes": "Highest margin, highest churn risk."},
        {"key": "escalate", "label": "Escalate for review", "notes": "Slower, but shares the risk."},
    ]


def _outcome(
    rng: random.Random, features: dict[str, Any], chosen: str, decided_at: datetime
) -> dict[str, Any] | None:
    """What happened afterwards. ``None`` for about a third of decisions."""
    # Recent decisions genuinely have no outcome yet, and some older ones were
    # simply never followed up -- both are realistic and both must be visible
    # as "unknown" rather than quietly counted as success.
    age_days = (datetime(2026, 6, 1, tzinfo=timezone.utc) - decided_at).days
    if age_days < 120 or rng.random() < 0.25:
        return None

    granted = {
        "approve_as_requested": features["requested_discount_pct"],
        "approve_reduced": max(features["requested_discount_pct"] - 7, 2),
        "reject": 0.0,
        "escalate": max(features["requested_discount_pct"] - 4, 2),
    }[chosen]

    win_probability = {
        "approve_as_requested": 0.78,
        "approve_reduced": 0.62,
        "reject": 0.34,
        "escalate": 0.55,
    }[chosen]
    if features["competitor_in_deal"]:
        win_probability -= 0.12

    won = rng.random() < win_probability
    margin = round(38.0 - granted * 0.85 + rng.uniform(-3, 3), 2)

    return {
        "recorded_at": decided_at + timedelta(days=rng.randint(60, 150)),
        "success_label": "success" if won and margin > 20 else ("mixed" if won else "failure"),
        "metrics": {
            "granted_discount_pct": round(granted, 1),
            "gross_margin_pct": margin,
            "won": 1.0 if won else 0.0,
            "days_to_close": round(rng.uniform(8, 60), 0),
        },
        "notes": (
            "Deal closed at the agreed discount."
            if won
            else "Customer did not sign; went with an alternative."
        ),
        "retrospective": (
            "The discount was probably deeper than it needed to be."
            if won and granted > 20
            else ""
        ),
    }


#: A live situation used by the demo to show precedent retrieval in action.
DEMO_NEW_DECISION: dict[str, Any] = {
    "title": "Enterprise renewal asking for 18% on a 12-month contract",
    "decision_type": "sales_discount",
    "context_text": (
        "An Enterprise customer in DE is asking for an 18% discount on a 240,000 EUR "
        "renewal with a 12 month contract. A competitor is actively in the deal."
    ),
    "context_structured": {
        "customer_segment": "Enterprise",
        "requested_discount_pct": 18.0,
        "deal_value_eur": 240_000.0,
        "contract_months": 12,
        "region": "DE",
        "renewal_status": "renewal",
        "competitor_in_deal": True,
    },
}
