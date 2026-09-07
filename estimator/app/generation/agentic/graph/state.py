"""The graph state: one ``TypedDict``, two accumulator channels.

In LangGraph the state is not an object you mutate — it is a set of **channels**.
Every node returns a *partial* update (a dict with only the keys it touched) and
LangGraph merges that update into the state using, per key, either:

* the **default reducer** — last write wins, the new value replaces the old one; or
* an **explicit reducer** declared with ``Annotated[T, fn]`` — the new value is
  combined with the accumulated one via ``fn``.

``operator.add`` on a list is therefore "append what this node produced to what
previous nodes produced", which is exactly what a field that grows along the flow
needs. Two channels here are accumulators:

``budgets``
    Filled by ``search_budgets``, one batch per component. Today the node loops
    over the components sequentially and could just as well return the whole list
    at once — but declaring it as an accumulator now is what will let the search
    fan out over components **in parallel** without touching the state at all:
    with ``operator.add`` two branches writing ``budgets`` concurrently merge,
    whereas a plain channel would raise ``InvalidUpdateError``.

``errors``
    Genuinely written by several different nodes along the run (a failed
    retrieval, an LLM extraction that came back empty...). This is the field that
    actually accumulates *across nodes* in the current sequential graph.

Everything else is a plain key with last-write-wins semantics. That is safe here
**only because the edges are strictly sequential**: no two nodes ever write the
same plain key. The moment a node fans out, any key two branches both write must
either become an accumulator or be split into per-branch keys.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.generation.agentic.graph.schemas import (
    BudgetHit,
    Component,
    GraphEstimate,
    Requirement,
)


class EstimationGraphState(TypedDict, total=False):
    """Channels of the estimation graph.

    ``total=False`` because a node only ever returns the subset of keys it
    produced; the graph is invoked with just ``transcript`` populated.
    """

    # --- input ------------------------------------------------------------- #
    transcript: str
    # Identifier of this estimation. Doubles as the checkpointer's ``thread_id``
    # and as the correlation key stamped on every node span, so a Logfire trace
    # and the checkpoint rows for the same run join on it.
    estimation_id: str

    # --- accumulators (explicit reducers) ---------------------------------- #
    budgets: Annotated[list[BudgetHit], operator.add]
    errors: Annotated[list[str], operator.add]

    # --- last-write-wins channels ------------------------------------------ #
    requirements: list[Requirement]
    components: list[Component]
    estimate: GraphEstimate
