"""Stress-suite metrics over the conversational pipeline's per-turn output.

These live under ``evals/stress/`` rather than in ``evals/metrics.py`` on
purpose. The metrics in ``evals/metrics.py`` score a ``(GoldenCase,
EstimationResult)`` pair; the three here score a different shape entirely:

- ``LatencyBudgetMetric`` / ``CostBudgetMetric`` score a sequence of per-turn
  ``turn_observed`` observations (the event emitted by
  ``EstimationService.estimate_conversational``: ``latency_ms``, ``cost_usd`` …).
- ``MemoryDriftMetric`` scores a session-state *memory snapshot* against the
  facts a conversation should still remember.

They reuse ``MetricResult`` (imported back from ``evals.metrics``) so the report
format is identical across both suites.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from evals.metrics import MetricResult

# One ``turn_observed`` event, as a plain mapping (the runner collects these
# from structlog). Only ``latency_ms`` / ``cost_usd`` are read here.
TurnObservation = Mapping[str, Any]


class LatencyBudgetMetric:
    """Per-turn latency budget: no single turn may exceed ``budget_ms``.

    ``score`` is the fraction of turns within budget; ``passed`` is true only
    when every turn is within budget. The boundary (``latency == budget``)
    counts as within budget.
    """

    name = "latency_budget"

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms

    def evaluate(self, turns: Sequence[TurnObservation]) -> MetricResult:
        if not turns:
            return MetricResult(self.name, 1.0, True, "no turns observed")

        latencies = [float(t.get("latency_ms", 0) or 0) for t in turns]
        over = [(i + 1, ms) for i, ms in enumerate(latencies) if ms > self.budget_ms]
        score = (len(turns) - len(over)) / len(turns)
        worst = max(latencies)

        if over:
            offenders = ", ".join(f"turn {idx}@{ms:.0f}ms" for idx, ms in over)
            details = (
                f"{len(over)}/{len(turns)} turns over {self.budget_ms:.0f} ms "
                f"(worst {worst:.0f} ms; {offenders})"
            )
        else:
            details = (
                f"all {len(turns)} turns within {self.budget_ms:.0f} ms (worst {worst:.0f} ms)"
            )
        return MetricResult(self.name, score, not over, details)


class CostBudgetMetric:
    """Total-conversation cost budget: the sum of per-turn ``cost_usd`` may not
    exceed ``budget_usd``.

    ``score`` is ``clamp(budget / total, 0, 1)`` (1.0 when nothing was spent);
    ``passed`` is ``total <= budget``. The boundary (``total == budget``) passes.
    """

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, turns: Sequence[TurnObservation]) -> MetricResult:
        total = sum(float(t.get("cost_usd", 0.0) or 0.0) for t in turns)
        passed = total <= self.budget_usd

        if self.budget_usd <= 0:
            score = 1.0 if total == 0 else 0.0
        elif total <= 0:
            score = 1.0
        else:
            score = max(0.0, min(1.0, self.budget_usd / total))

        details = f"total ${total:.4f} vs budget ${self.budget_usd:.4f} over {len(turns)} turn(s)"
        return MetricResult(self.name, score, passed, details)


_STOPWORDS = frozenset(
    {
        "stack",
        "includes",
        "include",
        "feature",
        "features",
        "project",
        "name",
        "budget",
        "locked",
        "the",
        "a",
        "an",
        "of",
        "and",
        "to",
        "with",
        "eur",
        "usd",
    }
)


def _normalize(text: str) -> str:
    text = text.lower()
    # Collapse thousands separators between digits so "30.000" / "30,000" match
    # the canonical fact value "30000".
    return re.sub(r"(?<=\d)[.,](?=\d)", "", text)


def _tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", _normalize(text)) if tok]


def _fact_terms(fact: str) -> list[str]:
    """The salient terms a fact must contribute to count as remembered.

    A leading ``label:`` is dropped (the value carries the signal), then generic
    label words are filtered out. Falls back to the raw value tokens if filtering
    leaves nothing, so a degenerate fact never matches everything vacuously.
    """
    value = fact.split(":", 1)[1] if ":" in fact else fact
    raw = _tokens(value)
    return [t for t in raw if t not in _STOPWORDS] or raw


class MemoryDriftMetric:
    """Fact survival across a long conversation.

    Given the facts that should still be remembered (``expected_facts``) and,
    optionally, facts that should have been superseded (``forbidden_facts`` —
    e.g. a contradicted budget), search a memory ``snapshot`` and report what
    drifted away or leaked through.

    ``snapshot`` is either a single string or a mapping of named buckets
    (e.g. ``{"metadata": ..., "anchors": ..., "summary": ..., "window": ...}``)
    to text. With buckets, survivors are reported with *where* they were found,
    which is what distinguishes anchor-survival from summary-survival. The exact
    snapshot shape is produced by the runner; this metric only needs str-ish
    bucket values.

    A fact counts as remembered when all its salient terms appear in a bucket.
    ``score = (expected found + forbidden absent) / (expected + forbidden)``;
    ``passed`` is true when no expected fact drifted and no forbidden fact leaked.
    """

    name = "memory_drift"

    def evaluate(
        self,
        expected_facts: Sequence[str],
        snapshot: str | Mapping[str, str],
        forbidden_facts: Sequence[str] = (),
    ) -> MetricResult:
        buckets = self._buckets(snapshot)

        remembered: dict[str, list[str]] = {}
        missing: list[str] = []
        for fact in expected_facts:
            where = self._locate(fact, buckets)
            if where:
                remembered[fact] = where
            else:
                missing.append(fact)

        leaked: dict[str, list[str]] = {}
        for fact in forbidden_facts:
            where = self._locate(fact, buckets)
            if where:
                leaked[fact] = where

        total = len(expected_facts) + len(forbidden_facts)
        good = (len(expected_facts) - len(missing)) + (len(forbidden_facts) - len(leaked))
        score = 1.0 if total == 0 else good / total
        passed = not missing and not leaked

        parts: list[str] = []
        if missing:
            parts.append(f"drifted (lost): {missing}")
        if leaked:
            parts.append(
                "leaked (should be gone): "
                + ", ".join(f"{f} @ {'+'.join(w)}" for f, w in leaked.items())
            )
        if remembered and not missing:
            parts.append(
                "remembered: " + ", ".join(f"{f} @ {'+'.join(w)}" for f, w in remembered.items())
            )
        details = "; ".join(parts) or "no facts to check"
        return MetricResult(self.name, score, passed, details)

    @staticmethod
    def _buckets(snapshot: str | Mapping[str, str]) -> dict[str, str]:
        if isinstance(snapshot, Mapping):
            return {str(k): str(v) for k, v in snapshot.items()}
        return {"snapshot": str(snapshot)}

    @staticmethod
    def _locate(fact: str, buckets: dict[str, str]) -> list[str]:
        terms = _fact_terms(fact)
        found: list[str] = []
        for bucket_name, text in buckets.items():
            bucket_tokens = set(_tokens(text))
            if all(term in bucket_tokens for term in terms):
                found.append(bucket_name)
        return found
