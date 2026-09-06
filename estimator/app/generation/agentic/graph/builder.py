"""Wiring and compilation of the Session 13 estimation graph.

The shape, for now, is a straight line::

    START
      -> extract_requirements
      -> classify_components
      -> search_budgets
      -> generate_estimate
      -> validate_and_consolidate
      -> END

Deliberately sequential. Making it a graph before making it concurrent is the
point of this first step: the topology becomes an explicit, inspectable object
(``compiled.get_graph().draw_ascii()``) instead of control flow buried in a
``while`` loop, and only then is it worth deciding which edges can run in
parallel.

**No checkpointer yet.** ``langgraph-checkpoint-postgres`` is installed and the
project's Postgres (the pgvector one) can host the checkpoint tables, but
persistence changes the invocation contract (every call needs a ``thread_id``)
and adds a table migration. ``compile()`` takes a ``checkpointer=`` argument, so
that is the seam; it stays empty in this step.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.generation.agentic.agent_tools import RetrievalBackend
from app.generation.agentic.graph import nodes
from app.generation.agentic.graph.schemas import GraphEstimate
from app.generation.agentic.graph.state import EstimationGraphState

# The node order is also the edge order: consecutive pairs become the direct
# edges below. Keeping it as data means adding a node is a one-line change and
# the sequence is readable at a glance.
NODE_SEQUENCE: tuple[str, ...] = (
    "extract_requirements",
    "classify_components",
    "search_budgets",
    "generate_estimate",
    "validate_and_consolidate",
)


def build_estimation_graph(*, backend: RetrievalBackend | None = None) -> StateGraph:
    """Build the (uncompiled) graph.

    Parameters
    ----------
    backend:
        Optional retrieval backend injected into ``search_budgets``. ``None`` uses
        the real S9/S10 pipeline; tests and the offline stub pass their own.
    """

    async def search_budgets_node(state: EstimationGraphState) -> dict[str, Any]:
        # A closure rather than functools.partial: the node keeps its own name in
        # traces and stays unambiguously a coroutine function for LangGraph.
        return await nodes.search_budgets(state, backend=backend)

    builder = StateGraph(EstimationGraphState)
    builder.add_node("extract_requirements", nodes.extract_requirements)
    builder.add_node("classify_components", nodes.classify_components)
    builder.add_node("search_budgets", search_budgets_node)
    builder.add_node("generate_estimate", nodes.generate_estimate)
    builder.add_node("validate_and_consolidate", nodes.validate_and_consolidate)

    builder.add_edge(START, NODE_SEQUENCE[0])
    for source, target in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        builder.add_edge(source, target)
    builder.add_edge(NODE_SEQUENCE[-1], END)

    return builder


def compile_estimation_graph(*, backend: RetrievalBackend | None = None):
    """Compile the graph into a runnable.

    The compiled object is stateless and reusable across requests; build it once
    and invoke it many times.
    """
    return build_estimation_graph(backend=backend).compile()


async def run_estimation_graph(
    transcript: str,
    *,
    backend: RetrievalBackend | None = None,
) -> GraphEstimate:
    """Run the graph over one transcript and return the final estimate.

    This is the function a router (or the demo script) calls. Whatever the graph
    does internally, the value returned here is always a ``GraphEstimate`` with a
    populated ``status`` - the contract the business backend sees does not change.
    """
    compiled = compile_estimation_graph(backend=backend)
    final_state = await compiled.ainvoke({"transcript": transcript})
    estimate = final_state.get("estimate")
    if estimate is None:
        # Defensive: validate_and_consolidate always sets one, but a caller that
        # interrupts the graph early should still get the documented shape.
        return GraphEstimate(
            status="insufficient_context",
            errors=list(final_state.get("errors") or []),
            issues=["The graph finished without producing an estimate."],
        )
    return estimate
