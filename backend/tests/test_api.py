"""End-to-end API tests: decisions, outcomes, search, packets, isolation."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.comparison.compare import DecisionView, build_comparison, summarise_precedents
from app.core.security import hash_api_key
from app.demo.seed import DEMO_API_KEY
from app.models import Decision, DecisionTemplateRow, Tenant
from tests.conftest import context


def template_id(client) -> str:
    """The demo template, by name.

    Not `templates[0]`: the list is sorted by name and other tests create
    templates that sort ahead of it, which silently pointed validation at a
    different schema.
    """
    from app.demo.dataset import DISCOUNT_TEMPLATE

    for template in client.get("/v1/templates").json():
        if template["name"] == DISCOUNT_TEMPLATE["name"]:
            return template["id"]
    raise AssertionError("the demo template is missing")


def live_decision_id(client) -> str:
    for decision in client.get("/v1/decisions").json():
        if decision["external_id"] == "DEC-LIVE":
            return decision["id"]
    raise AssertionError("the live demo decision is missing")


class TestHealthAndOverview:
    def test_health(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_openapi(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_overview(self, client):
        body = client.get("/v1/overview").json()
        assert body["template_count"] >= 1
        assert body["decision_count"] > 0
        assert body["with_outcome"] > 0
        assert body["embedding_coverage"] > 0.9
        assert body["embedding_model"] == "offline-hashing"
        assert body["outcome_mix"]


class TestTemplates:
    def test_demo_template_is_listed(self, client):
        template = next(
            t for t in client.get("/v1/templates").json() if t["id"] == template_id(client)
        )
        assert template["decision_count"] > 0
        names = {f["name"] for f in template["fields"]}
        assert "requested_discount_pct" in names

    def test_create(self, client):
        response = client.post(
            "/v1/templates",
            json={
                "name": "Procurement exception",
                "decision_type": "procurement",
                "fields": [
                    {"name": "supplier", "type": "string", "weight": 2.0},
                    {"name": "amount", "type": "number", "weight": 3.0, "tolerance": 5000},
                ],
            },
        )
        assert response.status_code == 201
        assert client.get(f"/v1/templates/{response.json()['id']}").status_code == 200

    def test_duplicate_name_is_rejected(self, client):
        payload = {
            "name": "Dup template",
            "decision_type": "x",
            "fields": [{"name": "a", "type": "string"}],
        }
        assert client.post("/v1/templates", json=payload).status_code == 201
        assert client.post("/v1/templates", json=payload).status_code == 409

    def test_duplicate_field_names_are_rejected(self, client):
        response = client.post(
            "/v1/templates",
            json={
                "name": "Bad fields",
                "decision_type": "x",
                "fields": [
                    {"name": "a", "type": "string"},
                    {"name": "a", "type": "number"},
                ],
            },
        )
        assert response.status_code == 422

    def test_every_returned_field_has_a_complete_shape(self, client):
        """A client must not have to defend against a shape that should not vary.

        The seed writes raw dicts that omit optional keys, so a field with no
        enum options came back without an `options` key at all and the UI blew
        up on it.
        """
        expected = {
            "name",
            "label",
            "type",
            "weight",
            "required",
            "options",
            "tolerance",
            "unit",
        }
        for template in client.get("/v1/templates").json():
            for spec in template["fields"]:
                assert set(spec) == expected, f"incomplete field: {sorted(spec)}"

        single = client.get(f"/v1/templates/{template_id(client)}").json()
        for spec in single["fields"]:
            assert set(spec) == expected

    def test_a_template_needs_fields(self, client):
        assert (
            client.post(
                "/v1/templates", json={"name": "Empty", "decision_type": "x", "fields": []}
            ).status_code
            == 422
        )


class TestDecisions:
    def test_create_with_options_and_evidence(self, client):
        response = client.post(
            "/v1/decisions",
            json={
                "title": "SMB asking for 12%",
                "decision_type": "sales_discount",
                "template_id": template_id(client),
                "context_text": "An SMB customer wants a 12% discount.",
                "context_structured": context(
                    customer_segment="SMB", requested_discount_pct=12.0, deal_value_eur=9000.0
                ),
                "options": [
                    {"key": "approve", "label": "Approve"},
                    {"key": "reject", "label": "Reject"},
                ],
                "evidence": [{"kind": "note", "summary": "Competitor quoted 15%"}],
                "owner": "ae_1",
            },
        )
        assert response.status_code == 201
        detail = client.get(f"/v1/decisions/{response.json()['id']}").json()
        assert len(detail["options"]) == 2
        assert len(detail["evidence"]) == 1
        assert detail["context_coverage"] == 1.0
        assert detail["embedding"]["model"] == "offline-hashing"

    def test_context_is_validated_against_the_template(self, client):
        response = client.post(
            "/v1/decisions",
            json={
                "title": "Invalid",
                "decision_type": "sales_discount",
                "template_id": template_id(client),
                "context_structured": {
                    "customer_segment": "Government",
                    "requested_discount_pct": 5,
                    "deal_value_eur": 1000,
                },
            },
        )
        assert response.status_code == 422
        assert "one of" in response.json()["detail"]

    def test_missing_required_field_is_rejected(self, client):
        response = client.post(
            "/v1/decisions",
            json={
                "title": "Incomplete",
                "decision_type": "sales_discount",
                "template_id": template_id(client),
                "context_structured": {"customer_segment": "SMB"},
            },
        )
        assert response.status_code == 422

    def test_duplicate_external_id_is_rejected(self, client):
        payload = {
            "title": "Dup",
            "decision_type": "sales_discount",
            "external_id": "DEC-DUP",
            "context_structured": {},
        }
        assert client.post("/v1/decisions", json=payload).status_code == 201
        assert client.post("/v1/decisions", json=payload).status_code == 409

    def test_a_decision_without_a_template_is_allowed(self, client):
        """Ad-hoc decisions must still be recordable."""
        response = client.post(
            "/v1/decisions",
            json={
                "title": "Ad-hoc call",
                "decision_type": "other",
                "context_text": "Something that fits no template.",
            },
        )
        assert response.status_code == 201
        detail = client.get(f"/v1/decisions/{response.json()['id']}").json()
        assert detail["validation_problems"] == []

    def test_editing_the_context_refreshes_the_embedding(self, client, db):
        from app.models import DecisionEmbedding

        created = client.post(
            "/v1/decisions",
            json={
                "title": "Embedding refresh",
                "decision_type": "sales_discount",
                "context_text": "original text about widgets",
            },
        ).json()
        before = db.scalar(
            select(DecisionEmbedding).where(
                DecisionEmbedding.decision_id == uuid.UUID(created["id"])
            )
        ).vector

        client.patch(
            f"/v1/decisions/{created['id']}",
            json={"context_text": "completely different text about shipping logistics"},
        )
        db.expire_all()
        after = db.scalar(
            select(DecisionEmbedding).where(
                DecisionEmbedding.decision_id == uuid.UUID(created["id"])
            )
        ).vector
        assert before != after, "a stale vector would retrieve the wrong precedents"

    def test_unknown_decision_is_404(self, client):
        assert client.get(f"/v1/decisions/{uuid.uuid4()}").status_code == 404


class TestOutcomes:
    def test_record_and_read_back(self, client):
        decision_id = client.get("/v1/decisions").json()[0]["id"]
        response = client.put(
            f"/v1/decisions/{decision_id}/outcome",
            json={
                "success_label": "success",
                "metrics": {"gross_margin_pct": 31.5},
                "notes": "Closed on time",
                "retrospective": "The discount could have been smaller.",
                "recorded_by": "tester",
            },
        )
        assert response.status_code == 200
        detail = client.get(f"/v1/decisions/{decision_id}").json()
        assert detail["outcome"]["success_label"] == "success"
        assert detail["outcome"]["retrospective"]

    def test_recording_twice_updates_rather_than_duplicates(self, client):
        decision_id = client.get("/v1/decisions").json()[1]["id"]
        client.put(
            f"/v1/decisions/{decision_id}/outcome", json={"success_label": "mixed"}
        )
        client.put(
            f"/v1/decisions/{decision_id}/outcome", json={"success_label": "failure"}
        )
        assert (
            client.get(f"/v1/decisions/{decision_id}").json()["outcome"]["success_label"]
            == "failure"
        )

    def test_invalid_label_is_rejected(self, client):
        decision_id = client.get("/v1/decisions").json()[0]["id"]
        assert (
            client.put(
                f"/v1/decisions/{decision_id}/outcome", json={"success_label": "great"}
            ).status_code
            == 422
        )

    def test_overdue_outcomes_are_listed(self, client):
        overdue = client.get("/v1/decisions/overdue-outcomes").json()
        assert isinstance(overdue, list)
        for item in overdue:
            assert item["days_overdue"] >= 0

    def test_a_decision_with_an_outcome_is_not_overdue(self, client):
        created = client.post(
            "/v1/decisions",
            json={
                "title": "Due yesterday",
                "decision_type": "sales_discount",
                "outcome_due_at": (
                    datetime.now(timezone.utc) - timedelta(days=1)
                ).isoformat(),
            },
        ).json()
        overdue_ids = {o["decision_id"] for o in client.get("/v1/decisions/overdue-outcomes").json()}
        assert created["id"] in overdue_ids

        client.put(f"/v1/decisions/{created['id']}/outcome", json={"success_label": "success"})
        overdue_ids = {o["decision_id"] for o in client.get("/v1/decisions/overdue-outcomes").json()}
        assert created["id"] not in overdue_ids


class TestSearch:
    def test_search_from_a_saved_decision(self, client):
        response = client.post(
            "/v1/decisions/search", json={"decision_id": live_decision_id(client), "limit": 8}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["precedents"]
        assert body["semantic_available"] is True
        assert body["candidates_considered"] > 0
        assert "not evidence that it will work again" in body["note"]

    def test_search_from_an_unsaved_situation(self, client):
        response = client.post(
            "/v1/decisions/search",
            json={
                "decision_type": "sales_discount",
                "template_id": template_id(client),
                "title": "Enterprise wants 20%",
                "context_text": "An Enterprise renewal asking for 20% off.",
                "context_structured": context(requested_discount_pct=20.0),
                "limit": 5,
            },
        ).json()
        assert len(response["precedents"]) == 5

    def test_every_component_score_is_returned(self, client):
        body = client.post(
            "/v1/decisions/search", json={"decision_id": live_decision_id(client), "limit": 3}
        ).json()
        for precedent in body["precedents"]:
            names = {c["name"] for c in precedent["components"]}
            assert names == {"structured", "semantic", "type", "recency"}
            assert precedent["structured"]["contributions"], (
                "the field-by-field breakdown is what lets a user disagree with the ranking"
            )

    def test_results_are_ordered_by_score(self, client):
        body = client.post(
            "/v1/decisions/search", json={"decision_id": live_decision_id(client), "limit": 10}
        ).json()
        scores = [p["score"] for p in body["precedents"]]
        assert scores == sorted(scores, reverse=True)

    def test_the_summary_separates_unknown_outcomes(self, client):
        body = client.post(
            "/v1/decisions/search", json={"decision_id": live_decision_id(client), "limit": 12}
        ).json()
        statistics = body["statistics"]
        assert statistics["total"] == statistics["with_outcome"] + statistics["without_outcome"]
        for option in statistics["options"]:
            if option["with_outcome"] == 0:
                assert option["success_rate"] is None, (
                    "'no outcome recorded' must never render as a 0% success rate"
                )

    def test_weights_can_be_overridden(self, client):
        payload = {"decision_id": live_decision_id(client), "limit": 5}
        default = client.post("/v1/decisions/search", json=payload).json()
        recency = client.post(
            "/v1/decisions/search",
            json=payload | {"weights": {"structured": 0.0, "semantic": 0.0, "type": 0.0, "recency": 1.0}},
        ).json()
        assert recency["weights_used"]["recency"] == 1.0
        assert [p["decision_id"] for p in default["precedents"]] != [
            p["decision_id"] for p in recency["precedents"]
        ]

    def test_a_situation_without_a_type_is_rejected(self, client):
        assert (
            client.post("/v1/decisions/search", json={"title": "no type"}).status_code == 422
        )

    def test_search_is_logged(self, client, db):
        from app.models import SearchLog

        before = db.scalar(select(Decision.id))  # touch the session
        client.post("/v1/decisions/search", json={"decision_id": live_decision_id(client)})
        db.expire_all()
        assert db.scalars(select(SearchLog)).all(), "searches must be auditable"


class TestComparison:
    def test_compare_two_decisions(self, client):
        ids = [d["id"] for d in client.get("/v1/decisions").json()[:3]]
        body = client.post("/v1/decisions/compare", json={"decision_ids": ids}).json()
        assert body["table"]["decision_ids"] == ids
        assert body["table"]["rows"]
        assert body["statistics"]["total"] == 3

    def test_rows_flag_where_decisions_differ(self, client):
        ids = [d["id"] for d in client.get("/v1/decisions").json()[:4]]
        rows = client.post("/v1/decisions/compare", json={"decision_ids": ids}).json()["table"]["rows"]
        assert any(r["varies"] for r in rows)

    def test_fewer_than_two_is_rejected(self, client):
        ids = [d["id"] for d in client.get("/v1/decisions").json()[:1]]
        assert client.post("/v1/decisions/compare", json={"decision_ids": ids}).status_code == 422

    def test_more_than_ten_is_rejected(self, client):
        ids = [d["id"] for d in client.get("/v1/decisions").json()[:11]]
        assert client.post("/v1/decisions/compare", json={"decision_ids": ids}).status_code == 422

    def test_unknown_decision_is_404(self, client):
        ids = [client.get("/v1/decisions").json()[0]["id"], str(uuid.uuid4())]
        assert client.post("/v1/decisions/compare", json={"decision_ids": ids}).status_code == 404


class TestSummaryStatistics:
    """Pure-function checks on the honesty of the aggregate summary."""

    def _view(self, identifier, chosen, outcome=None):
        return DecisionView(
            id=identifier,
            title=identifier,
            decision_type="x",
            context_structured={},
            chosen_option=chosen,
            outcome_label=outcome,
        )

    def test_unknown_outcomes_are_excluded_from_rates(self):
        views = [self._view(f"a{i}", "approve", "success") for i in range(3)]
        views += [self._view(f"b{i}", "approve", None) for i in range(7)]
        summary = summarise_precedents(views)
        approve = summary.options[0]
        assert approve.count == 10
        assert approve.with_outcome == 3
        assert approve.success_rate == 1.0, "3 of 3 known outcomes succeeded"
        assert any("no recorded outcome" in c for c in summary.caveats)

    def test_no_outcomes_at_all_yields_none_not_zero(self):
        views = [self._view(f"a{i}", "approve", None) for i in range(6)]
        assert summarise_precedents(views).options[0].success_rate is None

    def test_a_tiny_sample_is_flagged(self):
        views = [self._view("a", "approve", "success")]
        assert any("too few" in c for c in summarise_precedents(views).caveats)

    def test_empty_input(self):
        summary = summarise_precedents([])
        assert summary.total == 0
        assert summary.caveats

    def test_comparison_marks_the_odd_one_out(self):
        views = [
            DecisionView(id=str(i), title=str(i), decision_type="x",
                         context_structured={"segment": "Enterprise" if i < 3 else "SMB"})
            for i in range(4)
        ]
        table = build_comparison(views)
        row = next(r for r in table.rows if r.field == "segment")
        assert row.varies is True
        assert [c.differs for c in row.cells] == [False, False, False, True]

    def test_no_majority_means_nothing_is_marked_as_odd(self):
        views = [
            DecisionView(id=str(i), title=str(i), decision_type="x",
                         context_structured={"segment": f"S{i}"})
            for i in range(3)
        ]
        row = next(r for r in build_comparison(views).rows if r.field == "segment")
        assert row.varies is True
        assert not any(c.differs for c in row.cells)


class TestAIAndPackets:
    def test_summarise_uses_the_offline_provider(self, client):
        body = client.post(
            "/v1/decisions/summarise", json={"decision_id": live_decision_id(client), "limit": 8}
        ).json()
        assert body["narrative"]["summary"]
        assert body["statistics"]["total"] > 0

    def test_extract_returns_a_suggestion_not_a_commitment(self, client):
        body = client.post(
            "/v1/decisions/extract",
            json={
                "template_id": template_id(client),
                "text": "Enterprise renewal, wants 18% off 240k over 12 months.",
            },
        ).json()
        assert "note" in body
        assert "suggestion" in body["note"]

    def test_packet_contains_the_computed_numbers(self, client):
        response = client.post(
            "/v1/decision-packets",
            json={"decision_id": live_decision_id(client), "limit": 8},
        )
        assert response.status_code == 201
        packet = response.json()
        assert "## What history shows" in packet["body"]
        assert "## Closest precedents" in packet["body"]
        assert packet["payload"]["statistics"]["total"] > 0
        assert client.get(f"/v1/decision-packets/{packet['id']}").status_code == 200

    def test_packet_never_renders_an_unknown_rate_as_zero(self, client):
        packet = client.post(
            "/v1/decision-packets",
            json={"decision_id": live_decision_id(client), "limit": 12},
        ).json()
        statistics = packet["payload"]["statistics"]
        for option in statistics["options"]:
            if option["success_rate"] is None:
                assert "not known" in packet["body"]

    def test_packet_states_the_ranking_it_used(self, client):
        packet = client.post(
            "/v1/decision-packets", json={"decision_id": live_decision_id(client)}
        ).json()
        assert "Ranking weights:" in packet["body"]
        assert "candidate decision(s) were considered" in packet["body"]


class TestConfidentiality:
    def test_restricted_decisions_are_excluded_from_precedent_search(self, client):
        created = client.post(
            "/v1/decisions",
            json={
                "title": "Restricted enterprise renewal at 18%",
                "decision_type": "sales_discount",
                "template_id": template_id(client),
                "context_text": "An Enterprise renewal asking for 18% off 240,000 EUR.",
                "context_structured": context(),
                "chosen_option": "approve_as_requested",
                "confidentiality": "restricted",
            },
        ).json()

        body = client.post(
            "/v1/decisions/search", json={"decision_id": live_decision_id(client), "limit": 20}
        ).json()
        assert created["id"] not in {p["decision_id"] for p in body["precedents"]}, (
            "a restricted decision must never appear as somebody else's precedent"
        )

    def test_internal_decisions_are_included(self, client):
        created = client.post(
            "/v1/decisions",
            json={
                "title": "Internal enterprise renewal at 18%",
                "decision_type": "sales_discount",
                "template_id": template_id(client),
                "context_text": "An Enterprise renewal asking for 18% off 240,000 EUR.",
                "context_structured": context(),
                "chosen_option": "approve_as_requested",
                "confidentiality": "internal",
            },
        ).json()
        body = client.post(
            "/v1/decisions/search", json={"decision_id": live_decision_id(client), "limit": 30}
        ).json()
        assert created["id"] in {p["decision_id"] for p in body["precedents"]}


class TestTenantIsolation:
    def test_other_tenant_sees_nothing(self, client, db):
        db.add(Tenant(slug="other", name="Other", api_key_hash=hash_api_key("pk_other_key")))
        db.commit()
        headers = {"X-API-Key": "pk_other_key"}
        assert client.get("/v1/decisions", headers=headers).json() == []
        assert client.get("/v1/templates", headers=headers).json() == []
        assert client.get("/v1/overview", headers=headers).json()["decision_count"] == 0

    def test_cross_tenant_access_is_404(self, client, db):
        db.add(Tenant(slug="snoop", name="Snoop", api_key_hash=hash_api_key("pk_snoop_key")))
        db.commit()
        headers = {"X-API-Key": "pk_snoop_key"}
        decision = db.scalar(select(Decision))
        template = db.scalar(select(DecisionTemplateRow))
        assert client.get(f"/v1/decisions/{decision.id}", headers=headers).status_code == 404
        assert client.get(f"/v1/templates/{template.id}", headers=headers).status_code == 404

    def test_cross_tenant_search_finds_nothing(self, client, db):
        db.add(Tenant(slug="peek", name="Peek", api_key_hash=hash_api_key("pk_peek_key")))
        db.commit()
        body = client.post(
            "/v1/decisions/search",
            json={"decision_type": "sales_discount", "title": "probe"},
            headers={"X-API-Key": "pk_peek_key"},
        ).json()
        assert body["precedents"] == []
        assert body["candidates_considered"] == 0

    def test_cross_tenant_outcome_write_is_refused(self, client, db):
        db.add(Tenant(slug="writer", name="Writer", api_key_hash=hash_api_key("pk_writer_key")))
        db.commit()
        decision = db.scalar(select(Decision))
        assert (
            client.put(
                f"/v1/decisions/{decision.id}/outcome",
                json={"success_label": "success"},
                headers={"X-API-Key": "pk_writer_key"},
            ).status_code
            == 404
        )

    def test_invalid_key_is_401(self, client):
        assert client.get("/v1/decisions", headers={"X-API-Key": "pk_nope"}).status_code == 401

    def test_demo_key_works(self, client):
        assert client.get("/v1/decisions", headers={"X-API-Key": DEMO_API_KEY}).status_code == 200
