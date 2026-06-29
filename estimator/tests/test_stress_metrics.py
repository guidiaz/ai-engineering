"""Minimal unit tests for the stress-suite metrics.

Each metric gets a passing case, a failing case, and a boundary (limit) case.
The metrics are pure functions over plain dicts / strings, so no LLM, no app
wiring, and no fixtures are needed.
"""

from __future__ import annotations

from evals.metrics import MetricResult
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)


def _turn(latency_ms: float = 0, cost_usd: float = 0.0) -> dict:
    return {"latency_ms": latency_ms, "cost_usd": cost_usd}


# --------------------------------------------------------------------------- #
# LatencyBudgetMetric                                                          #
# --------------------------------------------------------------------------- #
def test_latency_budget_passes_when_all_within():
    result = LatencyBudgetMetric(budget_ms=2000).evaluate(
        [_turn(latency_ms=900), _turn(latency_ms=1500)]
    )
    assert isinstance(result, MetricResult)
    assert result.passed is True
    assert result.score == 1.0


def test_latency_budget_fails_when_a_turn_exceeds():
    result = LatencyBudgetMetric(budget_ms=1000).evaluate(
        [_turn(latency_ms=500), _turn(latency_ms=2500)]
    )
    assert result.passed is False
    assert result.score == 0.5  # 1 of 2 turns within budget
    assert "2500" in result.details  # offending turn surfaced


def test_latency_budget_boundary_equal_is_within():
    # Limit case: latency exactly at the budget counts as within (<=).
    result = LatencyBudgetMetric(budget_ms=1000).evaluate([_turn(latency_ms=1000)])
    assert result.passed is True
    assert result.score == 1.0


# --------------------------------------------------------------------------- #
# CostBudgetMetric                                                             #
# --------------------------------------------------------------------------- #
def test_cost_budget_passes_when_under():
    result = CostBudgetMetric(budget_usd=2.0).evaluate([_turn(cost_usd=0.5), _turn(cost_usd=0.5)])
    assert result.passed is True
    assert result.score == 1.0


def test_cost_budget_fails_when_over():
    result = CostBudgetMetric(budget_usd=1.0).evaluate(
        [_turn(cost_usd=0.8), _turn(cost_usd=0.7)]  # total 1.5 > 1.0
    )
    assert result.passed is False
    assert 0.0 < result.score < 1.0


def test_cost_budget_boundary_equal_passes():
    # Limit case: total exactly equal to the budget passes (<=), score 1.0.
    result = CostBudgetMetric(budget_usd=2.0).evaluate(
        [_turn(cost_usd=1.0), _turn(cost_usd=1.0)]  # total == budget
    )
    assert result.passed is True
    assert result.score == 1.0


# --------------------------------------------------------------------------- #
# MemoryDriftMetric                                                            #
# --------------------------------------------------------------------------- #
def test_memory_drift_passes_when_all_remembered():
    snapshot = {
        "metadata": "project name Nimbus, team of 5",
        "summary": "added authentication and CSV export",
    }
    facts = ["project name: Nimbus", "feature: authentication", "feature: CSV export"]
    result = MemoryDriftMetric().evaluate(facts, snapshot)
    assert result.passed is True
    assert result.score == 1.0
    assert "metadata" in result.details  # where-found is reported


def test_memory_drift_fails_when_fact_lost():
    snapshot = {"summary": "discussed onboarding and offline mode"}
    result = MemoryDriftMetric().evaluate(["stack includes Flutter"], snapshot)
    assert result.passed is False
    assert result.score == 0.0
    assert "Flutter" in result.details


def test_memory_drift_boundary_superseded_budget_leaks():
    # Limit case: the new budget must survive AND the old one must be gone, but
    # the snapshot still mentions both. The number normaliser maps "80.000" to
    # the canonical "80000" fact value.
    snapshot = {"summary": "budget raised to 80.000 EUR; earlier it was 30.000 EUR"}
    result = MemoryDriftMetric().evaluate(
        expected_facts=["budget locked: 80000 EUR"],
        snapshot=snapshot,
        forbidden_facts=["budget locked: 30000 EUR"],
    )
    assert result.passed is False  # 30000 leaked through
    assert result.score == 0.5  # 80000 found (1) but 30000 not absent (0), of 2
    assert "leaked" in result.details
