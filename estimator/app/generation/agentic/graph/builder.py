"""Wiring, instrumentation and compilation of the Session 13 estimation graph.

The shape, for now, is a straight line::

    START
      -> extract_requirements
      -> classify_components
      -> search_budgets
      -> generate_estimate
      -> validate_and_consolidate
      -> END

Deliberately sequential. Making it a graph before making it concurrent is the
point of the first step: the topology becomes an explicit, inspectable object
instead of control flow buried in a ``while`` loop, and only then is it worth
deciding which edges can run in parallel.

**Instrumentation lives here, not in the nodes.** Every node is wrapped by
:func:`_instrumented`, which opens one Logfire span per node execution. The nodes
stay pure functions of the state — they know nothing about tracing — and
``NODE_SEQUENCE`` remains the single source of node names.

**Checkpointing is optional.** ``compile()`` takes a ``checkpointer``; pass one
(see ``checkpointing.postgres_checkpointer``) and every invocation must then
carry a ``thread_id``, which is what makes a run resumable. Without one the graph
behaves exactly as in step 1.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import logfire
import structlog
from langgraph.graph import END, START, StateGraph

from app.generation.agentic.agent_tools import RetrievalBackend
from app.generation.agentic.graph import nodes
from app.generation.agentic.graph.schemas import GraphEstimate
from app.generation.agentic.graph.state import EstimationGraphState

log = structlog.get_logger()

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


def _span_attributes(update: dict[str, Any] | None) -> dict[str, Any]:
    """Summarise a node's partial update into flat span attributes.

    Deliberately a summary, not the payload: a span carrying whole transcripts
    and every retrieved chunk is unreadable and leaks content into the trace
    backend. Lists become counts, the estimate becomes its status and total.
    """
    attributes: dict[str, Any] = {}
    for key, value in (update or {}).items():
        if isinstance(value, list):
            attributes[f"{key}_count"] = len(value)
        elif isinstance(value, GraphEstimate):
            attributes["estimate_status"] = value.status
            attributes["estimate_total_hours"] = value.total_hours
        else:
            attributes[key] = value
    return attributes


def _instrumented(name: str, fn):
    """Wrap a node so each execution opens one Logfire span.

    The span is named ``node.<name>`` and carries the ``estimation_id`` (so the
    trace joins to the checkpoint rows on the same key) plus a summary of what
    the node returned. If Logfire was never configured, the span is a no-op and
    the graph runs unchanged.
    """

    async def instrumented_node(state: EstimationGraphState) -> dict[str, Any]:
        with logfire.span(
            "node.{node}",
            node=name,
            estimation_id=state.get("estimation_id"),
        ) as span:
            update = await fn(state)
            for key, value in _span_attributes(update).items():
                span.set_attribute(key, value)
            return update

    instrumented_node.__name__ = name
    return instrumented_node


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

    implementations = {
        "extract_requirements": nodes.extract_requirements,
        "classify_components": nodes.classify_components,
        "search_budgets": search_budgets_node,
        "generate_estimate": nodes.generate_estimate,
        "validate_and_consolidate": nodes.validate_and_consolidate,
    }

    builder = StateGraph(EstimationGraphState)
    for name in NODE_SEQUENCE:
        builder.add_node(name, _instrumented(name, implementations[name]))

    builder.add_edge(START, NODE_SEQUENCE[0])
    for source, target in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        builder.add_edge(source, target)
    builder.add_edge(NODE_SEQUENCE[-1], END)

    return builder


def compile_estimation_graph(
    *,
    backend: RetrievalBackend | None = None,
    checkpointer: Any | None = None,
):
    """Compile the graph into a runnable.

    The compiled object is stateless and reusable across requests; build it once
    and invoke it many times. Passing a ``checkpointer`` makes every invocation
    require a ``thread_id`` in its config — see :func:`run_estimation_graph`.
    """
    return build_estimation_graph(backend=backend).compile(checkpointer=checkpointer)


async def run_estimation_graph(
    transcript: str,
    *,
    backend: RetrievalBackend | None = None,
    checkpointer: Any | None = None,
    estimation_id: str | None = None,
) -> GraphEstimate:
    """Run the graph over one transcript and return the final estimate.

    Parameters
    ----------
    estimation_id:
        The identifier of this estimation. It is used as the checkpointer's
        ``thread_id`` and stamped on every node span, so a trace and its
        checkpoint rows join on the same key. Minted when not supplied — the same
        pattern as ``rag.estimator._current_request_id``.

    Whatever the graph does internally, the value returned here is always a
    ``GraphEstimate`` with a populated ``status``: the contract the business
    backend sees does not change.
    """
    estimation_id = estimation_id or str(uuid4())
    compiled = compile_estimation_graph(backend=backend, checkpointer=checkpointer)

    # thread_id is what the checkpointer keys on. It is harmless without one, so
    # it is always passed and the two code paths stay identical.
    config = {"configurable": {"thread_id": estimation_id}}

    with logfire.span("estimation_graph", estimation_id=estimation_id) as span:
        final_state = await compiled.ainvoke(
            {"transcript": transcript, "estimation_id": estimation_id}, config=config
        )
        estimate = final_state.get("estimate")
        if estimate is None:
            # Defensive: validate_and_consolidate always sets one, but a caller
            # that interrupts the graph early still gets the documented shape.
            estimate = GraphEstimate(
                status="insufficient_context",
                errors=list(final_state.get("errors") or []),
                issues=["The graph finished without producing an estimate."],
            )
        span.set_attribute("status", estimate.status)
        span.set_attribute("total_hours", estimate.total_hours)

    log.info(
        "graph_run_completed",
        estimation_id=estimation_id,
        status=estimate.status,
        checkpointed=checkpointer is not None,
    )
    return estimate
