"""Pydantic contracts for the Session 13 estimation graph.

These are the *payloads* that travel inside the graph state. The state itself
(``state.py``) is a ``TypedDict`` because LangGraph needs plain dict channels it
can merge with reducers; the values it carries are ordinary Pydantic models so
every node still gets validation at its boundary.

Naming note: this is a THIRD estimate shape in the codebase, and that is
deliberate rather than sloppy:

* ``domain.schemas.estimation.EstimationResult`` — Session 4, euros/weeks/phases.
* ``generation.rag.schemas.Estimate`` — Session 9, engineer-days with mandatory
  ``SourceCitation``s.
* ``GraphEstimate`` (here) — Session 13, engineer-hours with a ``status``.

The graph consolidates hours the same way the Session 12 agent did (it reuses the
very same deterministic ``calculate_estimate``), so it emits the agent's light
shape plus the ``status`` field the business backend reads.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The outcome of ``validate_and_consolidate``, and the only field the business
# backend needs to branch on:
#   ok                   — the estimate passed every guardrail; use the numbers.
#   needs_review         — an estimate exists but a guardrail flagged something
#                          (a component with no historical reference, hours far
#                          outside the range its references imply, ...).
#   insufficient_context — retrieval found nothing to ground on, so there are no
#                          numbers to report. Totals stay at 0 and the caller is
#                          expected to ask for more information rather than to
#                          treat this as a cheap project.
EstimateStatus = Literal["ok", "needs_review", "insufficient_context"]

ComponentCategory = Literal[
    "backend",
    "frontend",
    "mobile",
    "integration",
    "data",
    "infrastructure",
    "security",
    "other",
]


class Requirement(BaseModel):
    """One atomic thing the client asked for, lifted out of the transcript."""

    id: str = Field(description="Stable short id, e.g. 'R1'. Referenced by components.")
    text: str = Field(min_length=1, description="The requirement in one concise sentence.")


class RequirementList(BaseModel):
    """Structured-output wrapper for the ``extract_requirements`` node.

    Instructor needs a model to fill; a bare ``list[Requirement]`` is not a valid
    response model, hence the wrapper.
    """

    requirements: list[Requirement] = Field(default_factory=list)


class Component(BaseModel):
    """A group of requirements that gets costed as a single unit of work."""

    name: str = Field(min_length=1, description="Short technical name, e.g. 'Payments backend'.")
    category: ComponentCategory
    requirement_ids: list[str] = Field(
        default_factory=list, description="Ids of the requirements this component covers."
    )
    search_query: str = Field(
        min_length=1,
        description=(
            "Focused English query used to retrieve historical budgets for THIS "
            "component alone — not the whole project."
        ),
    )


class ComponentList(BaseModel):
    """Structured-output wrapper for the ``classify_components`` node."""

    components: list[Component] = Field(default_factory=list)


class BudgetHit(BaseModel):
    """One historical budget item retrieved for one component.

    Carries ``component`` so the flat, accumulated ``budgets`` list can be grouped
    back by component in ``generate_estimate``. Keeping the list flat is what lets
    it be an ``operator.add`` channel — see ``state.py``.
    """

    component: str = Field(description="Name of the component this hit was retrieved for.")
    chunk_id: int
    content_preview: str
    sector: str | None = None
    budget_id: str | None = None
    estimated_hours: float | None = None
    distance: float


class EstimatedComponent(BaseModel):
    """One costed line of the final estimate."""

    name: str
    category: ComponentCategory = "other"
    estimated_hours: float = Field(ge=0)
    reference_count: int = Field(ge=0, description="How many historical items grounded this line.")
    cited_chunk_ids: list[int] = Field(default_factory=list)
    unbudgeted: bool = Field(
        default=False, description="True when nothing historical was found to anchor this line."
    )


class GraphEstimate(BaseModel):
    """The graph's final structured estimate — the service's output contract.

    ``status`` is the field the business backend branches on; everything else is
    detail it may render but does not need to understand.
    """

    status: EstimateStatus
    components: list[EstimatedComponent] = Field(default_factory=list)
    total_hours: float = Field(default=0.0, ge=0)
    issues: list[str] = Field(
        default_factory=list, description="Guardrail findings from validate_and_consolidate."
    )
    errors: list[str] = Field(
        default_factory=list, description="Non-fatal failures accumulated during the run."
    )
