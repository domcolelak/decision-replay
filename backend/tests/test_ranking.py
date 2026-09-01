"""Tests for structured similarity, embeddings and hybrid ranking."""
from __future__ import annotations

from collections import Counter

import pytest

from app.embeddings.provider import OfflineHashingProvider, embedding_text
from app.search.ranking import (
    DEFAULT_WEIGHTS,
    ComponentScore,
    combine,
    cosine,
    rank_precedents,
    recency_score,
    spread_semantic,
)
from app.templates.fields import (
    DecisionTemplate,
    FieldType,
    TemplateField,
    coverage,
    structured_similarity,
)
from tests.conftest import NOW, context, precedent


def field(name, kind, weight=1.0, **kwargs):
    return TemplateField(name=name, label=name, type=FieldType(kind), weight=weight, **kwargs)


class TestFieldSimilarity:
    def test_exact_enum_match(self):
        spec = field("segment", "enum")
        assert spec.similarity("Enterprise", "Enterprise") == 1.0
        assert spec.similarity("Enterprise", "SMB") == 0.0

    def test_enum_comparison_ignores_case_and_padding(self):
        assert field("segment", "enum").similarity(" enterprise ", "Enterprise") == 1.0

    def test_boolean(self):
        spec = field("competitor", "boolean")
        assert spec.similarity(True, "yes") == 1.0
        assert spec.similarity(True, False) == 0.0

    def test_number_uses_the_configured_tolerance(self):
        spec = field("discount", "number", tolerance=4.0)
        close = spec.similarity(18.0, 19.0)
        far = spec.similarity(18.0, 34.0)
        assert close > 0.7 and far < 0.25

    def test_number_without_tolerance_is_scale_free(self):
        """A pair of 100,000s is as close as a pair of 100s."""
        spec = field("value", "number")
        assert spec.similarity(100.0, 110.0) == pytest.approx(
            spec.similarity(100_000.0, 110_000.0), abs=1e-6
        )

    def test_identical_numbers_are_perfectly_similar(self):
        assert field("v", "number", tolerance=5.0).similarity(7.0, 7.0) == 1.0

    def test_dates_decay_with_distance(self):
        spec = field("when", "date")
        near = spec.similarity("2026-01-01", "2026-01-15")
        far = spec.similarity("2026-01-01", "2024-01-01")
        assert near > 0.9 and far < 0.1

    def test_a_missing_value_is_unjudgeable_not_dissimilar(self):
        """None means 'cannot compare', which is not the same as 0."""
        spec = field("segment", "enum")
        assert spec.similarity(None, "Enterprise") is None
        assert spec.similarity("", "Enterprise") is None
        assert spec.similarity("Enterprise", None) is None

    def test_unparseable_numbers_score_zero_rather_than_raising(self):
        assert field("v", "number").similarity("not a number", 5) == 0.0


class TestStructuredSimilarity:
    @pytest.fixture()
    def template(self):
        return DecisionTemplate(
            name="t",
            decision_type="sales_discount",
            fields=[
                field("customer_segment", "enum", weight=2.5),
                field("requested_discount_pct", "number", weight=3.0, tolerance=4.0),
                field("region", "enum", weight=1.0),
            ],
        )

    def test_identical_contexts_score_one(self, template):
        ctx = context()
        assert structured_similarity(template, ctx, ctx).score == pytest.approx(1.0)

    def test_weights_decide_what_matters(self, template):
        """Differing on the heaviest field must hurt more than on the lightest."""
        base = context()
        wrong_discount = structured_similarity(
            template, base, context(requested_discount_pct=34.0)
        ).score
        wrong_region = structured_similarity(template, base, context(region="SK")).score
        assert wrong_discount < wrong_region

    def test_missing_fields_drop_their_weight(self, template):
        """A sparse record must not be penalised for the fields it lacks."""
        base = context()
        sparse = {"customer_segment": "Enterprise"}
        result = structured_similarity(template, base, sparse)
        assert result.score == pytest.approx(1.0), (
            "the one comparable field matched, so similarity is 1 over what could be judged"
        )
        assert set(result.skipped) == {"requested_discount_pct", "region"}
        assert result.comparable_weight == pytest.approx(2.5)

    def test_contributions_are_returned_for_display(self, template):
        result = structured_similarity(template, context(), context(region="SK"))
        names = {c.field for c in result.contributions}
        assert names == {"customer_segment", "requested_discount_pct", "region"}
        assert all(0.0 <= c.similarity <= 1.0 for c in result.contributions)

    def test_two_empty_contexts_produce_no_contributions(self, template):
        result = structured_similarity(template, {}, {})
        assert result.contributions == []
        assert result.score == 0.0

    def test_coverage_reports_how_much_was_filled_in(self, template):
        assert coverage(template, context()) == pytest.approx(1.0)
        assert coverage(template, {"customer_segment": "SMB"}) < 0.5
        assert coverage(template, {}) == 0.0


class TestTemplateValidation:
    @pytest.fixture()
    def template(self):
        return DecisionTemplate(
            name="t",
            decision_type="x",
            fields=[
                field("segment", "enum", options=["SMB", "Enterprise"], required=True),
                field("amount", "number"),
                field("when", "date"),
            ],
        )

    def test_valid_context_has_no_problems(self, template):
        assert template.validate({"segment": "SMB", "amount": 10, "when": "2026-01-01"}) == []

    def test_missing_required_field_is_reported(self, template):
        assert any("required" in p for p in template.validate({"amount": 1}))

    def test_value_outside_the_enum_is_reported(self, template):
        assert any("one of" in p for p in template.validate({"segment": "Government"}))

    def test_non_numeric_number_is_reported(self, template):
        problems = template.validate({"segment": "SMB", "amount": "lots"})
        assert any("must be a number" in p for p in problems)

    def test_optional_missing_fields_are_fine(self, template):
        assert template.validate({"segment": "SMB"}) == []


@pytest.fixture(scope="session")
def provider():
    return OfflineHashingProvider()


class TestEmbeddings:

    def test_is_deterministic(self, provider):
        first = provider.embed(["enterprise discount request"]).vectors[0]
        second = provider.embed(["enterprise discount request"]).vectors[0]
        assert first == second

    def test_identifies_itself_as_offline(self, provider):
        result = provider.embed(["x"])
        assert result.model == "offline-hashing", (
            "a stand-in must never look like a real embedding model in stored data"
        )
        assert result.version and result.dimensions > 0

    def test_related_text_scores_above_unrelated(self, provider):
        a, b, c = provider.embed(
            [
                "enterprise customer requests an 18 percent discount on a renewal",
                "enterprise client asking for 18% off a renewal contract",
                "warehouse forklift maintenance schedule for the third quarter",
            ]
        ).vectors
        assert cosine(a, b) > cosine(a, c)

    def test_empty_text_gives_a_zero_vector(self, provider):
        assert set(provider.embed([""]).vectors[0]) == {0.0}

    def test_embedding_text_includes_structured_fields(self):
        text = embedding_text("Title", "", {"customer_segment": "Enterprise", "empty": None})
        assert "customer_segment: Enterprise" in text
        assert "empty" not in text, "empty fields add noise, not signal"


class TestCosine:
    def test_identical_vectors(self):
        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero_not_a_half(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_are_clamped_to_zero(self):
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0

    def test_mismatched_or_empty_inputs(self):
        assert cosine([], [1.0]) == 0.0
        assert cosine([1.0, 2.0], [1.0]) == 0.0


class TestSemanticSpread:
    def test_a_compressed_range_is_spread_out(self):
        """The bug this guards: near-constant cosines flatten the whole ranking."""
        raw = [0.80, 0.79, 0.78, 0.77, 0.76]
        spread = spread_semantic(raw)
        assert max(spread) == pytest.approx(1.0)
        assert min(spread) == 0.0
        assert spread[0] > spread[-1]

    def test_order_is_preserved(self):
        raw = [0.5, 0.9, 0.7, 0.6]
        spread = spread_semantic(raw)
        assert sorted(range(len(raw)), key=lambda i: raw[i]) == sorted(
            range(len(raw)), key=lambda i: spread[i]
        )

    def test_too_few_values_are_left_alone(self):
        assert spread_semantic([0.8, 0.7]) == [0.8, 0.7]

    def test_all_identical_values_are_left_alone(self):
        assert spread_semantic([0.5, 0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5, 0.5]


class TestRecency:
    def test_today_scores_one(self):
        assert recency_score(NOW, now=NOW) == pytest.approx(1.0)

    def test_a_year_ago_scores_a_half(self):
        from datetime import timedelta

        assert recency_score(NOW - timedelta(days=365), now=NOW) == pytest.approx(0.5, abs=0.01)

    def test_no_date_scores_zero(self):
        assert recency_score(None, now=NOW) == 0.0


class TestCombine:
    def test_weighted_mean_over_available_components(self):
        components = [
            ComponentScore("a", 1.0, 1.0, True),
            ComponentScore("b", 1.0, 0.0, True),
        ]
        assert combine(components) == pytest.approx(0.5)

    def test_an_unavailable_component_redistributes_its_weight(self):
        """Missing semantic must not cap the score; it must drop out."""
        with_semantic = [
            ComponentScore("structured", 0.5, 1.0, True),
            ComponentScore("semantic", 0.5, 0.0, False),
        ]
        assert combine(with_semantic) == pytest.approx(1.0)

    def test_no_available_components_scores_zero(self):
        assert combine([ComponentScore("a", 1.0, 1.0, False)]) == 0.0

    def test_zero_weights_are_ignored(self):
        components = [
            ComponentScore("a", 0.0, 1.0, True),
            ComponentScore("b", 1.0, 0.25, True),
        ]
        assert combine(components) == pytest.approx(0.25)


class TestRanking:
    @pytest.fixture()
    def template(self):
        from app.demo.dataset import DISCOUNT_TEMPLATE

        return DecisionTemplate.from_dict(DISCOUNT_TEMPLATE)

    def test_the_closest_context_ranks_first(self, template):
        target = precedent("TARGET")
        candidates = [
            precedent("EXACT"),
            precedent("DIFFERENT", customer_segment="SMB", requested_discount_pct=3.0),
            precedent("MIDDLING", requested_discount_pct=25.0),
        ]
        result = rank_precedents(target, candidates, template, now=NOW)
        assert result.precedents[0].decision_id == "EXACT"
        assert result.precedents[-1].decision_id == "DIFFERENT"

    def test_the_target_never_matches_itself(self, template):
        target = precedent("SAME")
        result = rank_precedents(target, [precedent("SAME")], template, now=NOW)
        assert result.precedents == []

    def test_every_component_score_is_exposed(self, template):
        result = rank_precedents(precedent("T"), [precedent("A")], template, now=NOW)
        names = {c.name for c in result.precedents[0].components}
        assert names == {"structured", "semantic", "type", "recency"}
        for component in result.precedents[0].components:
            assert 0.0 <= component.score <= 1.0
            assert component.detail

    def test_ranking_works_with_no_embeddings_at_all(self, template):
        """Structured search is a requirement, not a fallback."""
        target = precedent("T")
        candidates = [
            precedent("CLOSE"),
            precedent("FAR", customer_segment="SMB", requested_discount_pct=2.0),
        ]
        result = rank_precedents(target, candidates, template, now=NOW)
        assert result.semantic_available is False
        assert result.precedents[0].decision_id == "CLOSE"
        semantic = next(
            c for c in result.precedents[0].components if c.name == "semantic"
        )
        assert semantic.available is False
        assert "redistributed" in semantic.detail

    def test_recency_breaks_ties_between_identical_contexts(self, template):
        target = precedent("T")
        result = rank_precedents(
            target,
            [precedent("OLD", days_ago=900), precedent("NEW", days_ago=10)],
            template,
            now=NOW,
        )
        assert result.precedents[0].decision_id == "NEW"

    def test_a_different_decision_type_ranks_lower(self, template):
        target = precedent("T")
        result = rank_precedents(
            target,
            [precedent("SAME_TYPE"), precedent("OTHER_TYPE", decision_type="procurement")],
            template,
            now=NOW,
        )
        assert result.precedents[0].decision_id == "SAME_TYPE"

    def test_weights_can_be_overridden_per_search(self, template):
        target = precedent("T")
        candidates = [
            precedent("RECENT_BUT_DIFFERENT", days_ago=1, customer_segment="SMB",
                      requested_discount_pct=2.0),
            precedent("OLD_BUT_IDENTICAL", days_ago=1200),
        ]
        structured_first = rank_precedents(
            target, candidates, template,
            weights={"structured": 1.0, "semantic": 0.0, "type": 0.0, "recency": 0.0},
            now=NOW,
        )
        recency_first = rank_precedents(
            target, candidates, template,
            weights={"structured": 0.0, "semantic": 0.0, "type": 0.0, "recency": 1.0},
            now=NOW,
        )
        assert structured_first.precedents[0].decision_id == "OLD_BUT_IDENTICAL"
        assert recency_first.precedents[0].decision_id == "RECENT_BUT_DIFFERENT"

    def test_min_score_filters(self, template):
        result = rank_precedents(
            precedent("T"),
            [precedent("A", customer_segment="SMB", requested_discount_pct=1.0)],
            template,
            min_score=0.99,
            now=NOW,
        )
        assert result.precedents == []

    def test_limit_is_respected(self, template):
        candidates = [precedent(f"C{i}") for i in range(20)]
        assert len(rank_precedents(precedent("T"), candidates, template, limit=5, now=NOW).precedents) == 5

    def test_ranking_is_deterministic(self, template):
        candidates = [precedent(f"C{i}", requested_discount_pct=float(i)) for i in range(15)]
        first = [p.decision_id for p in rank_precedents(precedent("T"), candidates, template, now=NOW).precedents]
        second = [p.decision_id for p in rank_precedents(precedent("T"), candidates, template, now=NOW).precedents]
        assert first == second

    def test_the_result_carries_the_anti_causal_note(self, template):
        result = rank_precedents(precedent("T"), [precedent("A")], template, now=NOW)
        assert "not evidence that it will work again" in result.note


class TestRankingOnDemoData:
    """The product claim: comparable precedents actually surface."""

    @pytest.fixture()
    def ranked(self, demo_contexts, template):
        from app.demo.dataset import DEMO_NEW_DECISION

        provider = OfflineHashingProvider()
        situation = DEMO_NEW_DECISION
        vector = provider.embed(
            [
                embedding_text(
                    situation["title"],
                    situation["context_text"],
                    situation["context_structured"],
                )
            ]
        ).vectors[0]
        target = precedent("LIVE", embedding=vector, **situation["context_structured"])
        return rank_precedents(target, demo_contexts, template, limit=10, now=NOW)

    def test_the_top_precedent_shares_the_segment(self, ranked, demo_contexts):
        top = next(c for c in demo_contexts if c.id == ranked.precedents[0].decision_id)
        assert top.context_structured["customer_segment"] == "Enterprise"

    def test_the_top_precedent_asks_for_a_similar_discount(self, ranked, demo_contexts):
        top = next(c for c in demo_contexts if c.id == ranked.precedents[0].decision_id)
        assert abs(top.context_structured["requested_discount_pct"] - 18.0) < 5.0

    def test_the_result_set_is_dominated_by_the_right_segment(self, ranked, demo_contexts):
        by_id = {c.id: c for c in demo_contexts}
        segments = Counter(
            by_id[p.decision_id].context_structured["customer_segment"]
            for p in ranked.precedents
        )
        assert segments["Enterprise"] >= 5, f"expected Enterprise to dominate, got {segments}"

    def test_scores_actually_discriminate(self, ranked):
        """A flat ranking is a useless ranking."""
        scores = [p.score for p in ranked.precedents]
        assert scores[0] - scores[-1] > 0.1, f"ranking is too flat: {scores}"

    def test_scores_are_ordered(self, ranked):
        scores = [p.score for p in ranked.precedents]
        assert scores == sorted(scores, reverse=True)

    def test_every_precedent_reports_its_context_coverage(self, ranked):
        assert all(p.context_coverage > 0 for p in ranked.precedents)
