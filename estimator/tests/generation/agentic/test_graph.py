"""Session 13 - tests for the estimation graph.

Network-free: the two LLM nodes get a fake ``LLMWrapper`` and ``search_budgets``
gets a fake retrieval backend, so the whole graph runs offline.

The test that matters most is ``test_accumulator_channels_do_not_double``: the
classic LangGraph mistake is a node returning the *whole* state instead of a
partial update, which re-applies ``operator.add`` and silently duplicates every
accumulated item. It is invisible in the final numbers unless something asserts
on the length.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.generation.agentic.agent_schemas import SearchBudgetsArgs
from app.generation.agentic.graph import compile_estimation_graph, run_estimation_graph
from app.generation.agentic.graph.builder import NODE_SEQUENCE
from app.generation.agentic.graph.nodes import (
    generate_estimate,
    search_budgets,
    validate_and_consolidate,
)
from app.generation.agentic.graph.schemas import (
    BudgetHit,
    Component,
    ComponentList,
    GraphEstimate,
    Requirement,
    RequirementList,
)

TRANSCRIPT = "We need a payments backend and an Android app for couriers."


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _FakeWrapper:
    """Stands in for ``LLMWrapper``, answering by requested response model."""

    def __init__(self, requirements: list[Requirement], components: list[Component]):
        self._requirements = requirements
        self._components = components
        self.calls: list[str] = []

    def complete_structured(self, *, response_model, **_kwargs):  # noqa: ANN001
        self.calls.append(response_model.__name__)
        if response_model is RequirementList:
            return RequirementList(requirements=self._requirements), {}
        if response_model is ComponentList:
            return ComponentList(components=self._components), {}
        raise AssertionError(f"unexpected response model {response_model!r}")


def _components() -> list[Component]:
    return [
        Component(
            name="Payments backend",
            category="backend",
            requirement_ids=["R1"],
            search_query="payments backend with card processing",
        ),
        Component(
            name="Courier mobile app",
            category="mobile",
            requirement_ids=["R2"],
            search_query="android delivery courier mobile application",
        ),
    ]


def _requirements() -> list[Requirement]:
    return [
        Requirement(id="R1", text="Process card payments."),
        Requirement(id="R2", text="Android app for couriers."),
    ]


def _hit(chunk_id: int, hours: float) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "content_preview": f"historical item {chunk_id}",
        "sector": "logistics",
        "budget_id": f"BUD-{chunk_id}",
        "estimated_hours": hours,
        "distance": 0.3,
    }


def _backend_with(mapping: dict[str, list[dict[str, Any]]]):
    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        for needle, items in mapping.items():
            if needle in args.query:
                return items
        return []

    return backend


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch the composition root so both LLM nodes use the fake wrapper."""
    wrapper = _FakeWrapper(_requirements(), _components())
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)
    return wrapper


# --------------------------------------------------------------------------- #
# Topology                                                                     #
# --------------------------------------------------------------------------- #
def test_graph_is_the_declared_sequence():
    graph = compile_estimation_graph().get_graph()
    assert list(NODE_SEQUENCE) == [n for n in graph.nodes if not n.startswith("__")]

    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "extract_requirements") in edges
    assert ("validate_and_consolidate", "__end__") in edges
    for source, target in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        assert (source, target) in edges


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #
async def test_full_run_returns_ok_estimate(fake_llm):
    backend = _backend_with(
        {
            "payments": [_hit(1, 100.0), _hit(2, 140.0)],
            "courier": [_hit(3, 200.0), _hit(4, 240.0)],
        }
    )
    estimate = await run_estimation_graph(TRANSCRIPT, backend=backend)

    assert isinstance(estimate, GraphEstimate)
    assert estimate.status == "ok"
    assert [c.name for c in estimate.components] == ["Payments backend", "Courier mobile app"]
    # median(100,140)=120 and median(200,240)=220, each +15% contingency.
    assert estimate.total_hours == pytest.approx(138.0 + 253.0, abs=0.05)
    assert estimate.components[0].category == "backend"
    assert estimate.components[1].cited_chunk_ids == [3, 4]
    assert estimate.errors == []


async def test_accumulator_channels_do_not_double(fake_llm):
    """``budgets`` must hold exactly the retrieved hits - not two copies."""
    backend = _backend_with(
        {"payments": [_hit(1, 100.0), _hit(2, 140.0)], "courier": [_hit(3, 200.0)]}
    )
    compiled = compile_estimation_graph(backend=backend)
    final = await compiled.ainvoke({"transcript": TRANSCRIPT})

    assert len(final["budgets"]) == 3
    assert sorted(hit.chunk_id for hit in final["budgets"]) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Degraded paths                                                               #
# --------------------------------------------------------------------------- #
async def test_retrieval_failure_is_accumulated_not_raised(fake_llm):
    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        if "payments" in args.query:
            raise RuntimeError("embedder unavailable")
        return [_hit(3, 200.0), _hit(4, 240.0)]

    estimate = await run_estimation_graph(TRANSCRIPT, backend=backend)

    # The healthy component is still costed; the failure only downgrades status.
    assert estimate.status == "needs_review"
    assert any("embedder unavailable" in e for e in estimate.errors)
    costed = {c.name: c for c in estimate.components}
    assert costed["Courier mobile app"].estimated_hours > 0
    assert costed["Payments backend"].unbudgeted is True


async def test_nothing_retrieved_is_insufficient_context(fake_llm):
    estimate = await run_estimation_graph(TRANSCRIPT, backend=_backend_with({}))

    assert estimate.status == "insufficient_context"
    assert estimate.total_hours == 0.0
    # An ungrounded run must not look like a cheap project.
    assert all(c.unbudgeted for c in estimate.components)


async def test_empty_transcript_short_circuits_to_insufficient(fake_llm):
    estimate = await run_estimation_graph("   ", backend=_backend_with({}))

    assert estimate.status == "insufficient_context"
    assert any("empty transcript" in e for e in estimate.errors)


# --------------------------------------------------------------------------- #
# Node purity                                                                  #
# --------------------------------------------------------------------------- #
async def test_nodes_return_partial_updates_and_do_not_mutate_state():
    state = {
        "transcript": TRANSCRIPT,
        "components": _components(),
        "budgets": [],
        "errors": [],
    }
    snapshot = {
        "budgets": list(state["budgets"]),
        "errors": list(state["errors"]),
        "components": list(state["components"]),
    }

    update = await search_budgets(state, backend=_backend_with({"payments": [_hit(1, 100.0)]}))

    # Only produced keys come back, and the input state is untouched.
    assert set(update) <= {"budgets", "errors"}
    assert state["budgets"] == snapshot["budgets"]
    assert state["errors"] == snapshot["errors"]
    assert state["components"] == snapshot["components"]


async def test_validate_sets_status_needs_review_on_guardrail_issue():
    # One component costed far outside the range its references imply.
    state = {
        "components": _components()[:1],
        "budgets": [
            BudgetHit(
                component="Payments backend",
                chunk_id=1,
                content_preview="x",
                estimated_hours=100.0,
                distance=0.1,
            )
        ],
        "errors": [],
    }
    generated = await generate_estimate(state)
    inflated = generated["estimate"].model_copy(
        update={
            "components": [
                generated["estimate"].components[0].model_copy(update={"estimated_hours": 9_000.0})
            ],
            "total_hours": 9_000.0,
        }
    )
    result = await validate_and_consolidate({**state, "estimate": inflated})

    assert result["estimate"].status == "needs_review"
    assert result["estimate"].issues


async def test_every_retrieved_budget_is_joined_back_to_its_component(fake_llm):
    """The three nodes join on the component NAME - pin that the join holds.

    ``search_budgets`` stamps ``component.name`` onto each hit and
    ``generate_estimate`` groups the flat ``budgets`` list back by that string. If
    the two ever drift apart, every line silently becomes ``unbudgeted`` with
    ``refs: 0`` and the run reports ``insufficient_context`` - which reads as "no
    historical data" rather than "the join broke". This invariant tells them apart.
    """
    backend = _backend_with(
        {
            "payments": [_hit(1, 100.0), _hit(2, 140.0)],
            "courier": [_hit(3, 200.0), _hit(4, 240.0), _hit(5, 260.0)],
        }
    )
    compiled = compile_estimation_graph(backend=backend)
    final = await compiled.ainvoke({"transcript": TRANSCRIPT})
    estimate = final["estimate"]

    # Nothing retrieved may be dropped on the way into the estimate.
    assert sum(c.reference_count for c in estimate.components) == len(final["budgets"])
    assert sum(len(c.cited_chunk_ids) for c in estimate.components) == len(final["budgets"])
    assert not any(c.unbudgeted for c in estimate.components)
