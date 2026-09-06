"""The five nodes of the Session 13 estimation graph.

Every node is a **pure function of the state**: it reads what it needs, does its
work, and returns a *partial* update. Two rules the graph depends on:

1. **Never mutate the incoming state.** Build a new value and return it.
2. **Never return the whole state.** Returning an accumulated channel (``budgets``,
   ``errors``) re-applies its reducer and silently doubles the list. Return only
   the keys the node actually produced.

Each node reuses logic that already exists rather than reimplementing it:

===========================  ==================================================
Node                         Reuses
===========================  ==================================================
``extract_requirements``     ``LLMWrapper.complete_structured`` (foundation)
``classify_components``      ``LLMWrapper.complete_structured`` (foundation)
``search_budgets``           ``agent_tools.default_retrieval_backend`` -> the
                             S9/S10 hybrid ``retrieve()`` pipeline
``generate_estimate``        ``agent_tools.calculate_estimate`` (deterministic)
``validate_and_consolidate`` ``agent_tools.validate_estimate`` (S4 guardrails)
===========================  ==================================================

Failures are **accumulated, not raised**. A node that cannot do its job appends to
``errors`` and returns whatever partial result it has, so one unavailable
component never kills a run that could still produce a useful estimate. The only
thing that changes as a result is the final ``status``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import structlog

from app.config import get_settings
from app.generation.agentic.agent_schemas import SearchBudgetsArgs
from app.generation.agentic.agent_tools import (
    RetrievalBackend,
    calculate_estimate,
    default_retrieval_backend,
    validate_estimate,
)
from app.generation.agentic.graph.schemas import (
    BudgetHit,
    Component,
    ComponentList,
    EstimatedComponent,
    GraphEstimate,
    Requirement,
    RequirementList,
)
from app.generation.agentic.graph.state import EstimationGraphState

log = structlog.get_logger()

_EXTRACT_SYSTEM_PROMPT = (
    "You are a software-delivery analyst reading a raw client meeting transcript. "
    "Extract the concrete, buildable requirements the client asked for. One "
    "requirement per distinct piece of functionality. Ignore small talk, budget "
    "haggling, scheduling and anecdotes. Normalise to concise technical English "
    "regardless of the transcript language. Give each requirement a short stable "
    "id (R1, R2, ...). Never invent functionality the transcript does not support."
)

_CLASSIFY_SYSTEM_PROMPT = (
    "You group software requirements into the components that will be costed and "
    "built as units of work. Merge requirements that belong to the same deliverable "
    "(e.g. several API endpoints of one service = one backend component); keep "
    "genuinely different deliverables apart (a mobile app is not the backend it "
    "talks to). For each component write a focused English search_query describing "
    "THAT component alone - it is used to retrieve historical budgets for it, so a "
    "query describing the whole project would retrieve nothing useful. Every "
    "requirement id must end up in exactly one component."
)


def _model_for_extraction() -> str:
    """Model used by the two transcript-understanding nodes.

    Reuses ``REFORMULATION_MODEL`` - the knob that already means "the small model
    that turns a messy transcript into structure" - instead of adding another
    setting for the same job. It is in ``AVAILABLE_MODELS``, so it is switchable
    at runtime from the Ajustes tab like the rest.
    """
    return get_settings().REFORMULATION_MODEL


# --------------------------------------------------------------------------- #
# 1. extract_requirements                                                     #
# --------------------------------------------------------------------------- #
async def extract_requirements(state: EstimationGraphState) -> dict[str, Any]:
    """Transcript -> list of requirements."""
    from app.dependencies import get_llm_wrapper

    transcript = (state.get("transcript") or "").strip()
    if not transcript:
        return {"requirements": [], "errors": ["extract_requirements: empty transcript."]}

    settings = get_settings()
    try:
        result, _meta = await asyncio.to_thread(
            get_llm_wrapper().complete_structured,
            system_prompt=_EXTRACT_SYSTEM_PROMPT,
            user_message=transcript,
            response_model=RequirementList,
            model_override=_model_for_extraction(),
            # Reasoning models spend tokens before emitting the structured answer;
            # the wrapper's small default would truncate a long transcript.
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - accumulate, never kill the run.
        log.warning("graph_extract_requirements_failed", error=str(exc)[:200])
        return {
            "requirements": [],
            "errors": [f"extract_requirements: {type(exc).__name__}: {str(exc)[:200]}"],
        }

    requirements = result.requirements
    log.info("graph_extract_requirements", count=len(requirements))
    update: dict[str, Any] = {"requirements": requirements}
    if not requirements:
        update["errors"] = ["extract_requirements: no requirements found in the transcript."]
    return update


# --------------------------------------------------------------------------- #
# 2. classify_components                                                      #
# --------------------------------------------------------------------------- #
async def classify_components(state: EstimationGraphState) -> dict[str, Any]:
    """Requirements -> components, each with a category and its own search query."""
    from app.dependencies import get_llm_wrapper

    requirements: list[Requirement] = state.get("requirements") or []
    if not requirements:
        # Nothing upstream to group. extract_requirements already recorded why.
        return {"components": []}

    listing = "\n".join(f"{r.id}: {r.text}" for r in requirements)
    settings = get_settings()
    try:
        result, _meta = await asyncio.to_thread(
            get_llm_wrapper().complete_structured,
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
            user_message=f"Requirements:\n{listing}",
            response_model=ComponentList,
            model_override=_model_for_extraction(),
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("graph_classify_components_failed", error=str(exc)[:200])
        return {
            "components": [],
            "errors": [f"classify_components: {type(exc).__name__}: {str(exc)[:200]}"],
        }

    components = result.components
    log.info("graph_classify_components", count=len(components))
    update: dict[str, Any] = {"components": components}
    if not components:
        update["errors"] = ["classify_components: requirements produced no components."]
    return update


# --------------------------------------------------------------------------- #
# 3. search_budgets                                                           #
# --------------------------------------------------------------------------- #
async def search_budgets(
    state: EstimationGraphState,
    *,
    backend: RetrievalBackend | None = None,
) -> dict[str, Any]:
    """For each component, retrieve reference budgets - sequentially, for now.

    The loop is deliberately serial in this first step: it is the baseline the
    next iteration parallelises. Because ``budgets`` is an ``operator.add``
    channel, turning this into a fan-out later needs no change to the state.

    ``backend`` is injectable so tests and the offline student stub can stand in
    for the real retrieval pipeline (the same seam the Session 12 agent uses).
    """
    retrieval: RetrievalBackend = backend or default_retrieval_backend
    components: list[Component] = state.get("components") or []
    if not components:
        return {}

    hits: list[BudgetHit] = []
    errors: list[str] = []
    for component in components:
        try:
            items = await retrieval(SearchBudgetsArgs(query=component.search_query))
        except Exception as exc:  # noqa: BLE001 - one bad component must not stop the rest.
            log.warning(
                "graph_search_budgets_failed",
                component=component.name,
                error=str(exc)[:200],
            )
            errors.append(
                f"search_budgets[{component.name}]: {type(exc).__name__}: {str(exc)[:200]}"
            )
            continue

        for item in items:
            hits.append(
                BudgetHit(
                    component=component.name,
                    chunk_id=item["id"],
                    content_preview=item.get("content_preview", ""),
                    sector=item.get("sector"),
                    budget_id=item.get("budget_id"),
                    estimated_hours=item.get("estimated_hours"),
                    distance=item.get("distance", 0.0),
                )
            )
        if not items:
            log.info("graph_search_budgets_empty", component=component.name)

    log.info("graph_search_budgets", components=len(components), hits=len(hits))
    # Only the keys this node produced. Both are accumulator channels, so an empty
    # result is an absent key rather than an empty list.
    update: dict[str, Any] = {}
    if hits:
        update["budgets"] = hits
    if errors:
        update["errors"] = errors
    return update


# --------------------------------------------------------------------------- #
# 4. generate_estimate                                                        #
# --------------------------------------------------------------------------- #
async def generate_estimate(state: EstimationGraphState) -> dict[str, Any]:
    """Consolidate the retrieved budgets into an estimate. Deterministic, no LLM.

    Groups the flat ``budgets`` accumulator back by component, then hands the
    reference hours to the Session 12 ``calculate_estimate`` tool - median plus a
    fixed contingency buffer - so the graph and the agent cost work identically.
    """
    components: list[Component] = state.get("components") or []
    budgets: list[BudgetHit] = state.get("budgets") or []

    by_component: dict[str, list[BudgetHit]] = defaultdict(list)
    for hit in budgets:
        by_component[hit.component].append(hit)

    payload = {
        "components": [
            {
                "name": component.name,
                "reference_amounts": [
                    hit.estimated_hours
                    for hit in by_component.get(component.name, [])
                    if hit.estimated_hours is not None
                ],
            }
            for component in components
        ]
    }

    if not payload["components"]:
        return {"estimate": GraphEstimate(status="insufficient_context", total_hours=0.0)}

    computed = calculate_estimate(payload)
    category_of = {component.name: component.category for component in components}

    lines = [
        EstimatedComponent(
            name=line["name"],
            category=category_of.get(line["name"], "other"),
            estimated_hours=line["estimated_hours"],
            reference_count=line["reference_count"],
            cited_chunk_ids=[hit.chunk_id for hit in by_component.get(line["name"], [])],
            unbudgeted=line["unbudgeted"],
        )
        for line in computed["components"]
    ]

    log.info(
        "graph_generate_estimate",
        components=len(lines),
        total_hours=computed["total_hours"],
    )
    # status is provisional here - validate_and_consolidate owns the final value.
    return {
        "estimate": GraphEstimate(
            status="ok",
            components=lines,
            total_hours=computed["total_hours"],
        )
    }


# --------------------------------------------------------------------------- #
# 5. validate_and_consolidate                                                 #
# --------------------------------------------------------------------------- #
async def validate_and_consolidate(state: EstimationGraphState) -> dict[str, Any]:
    """Run the guardrails and fix the output ``status``.

    This is the only node that decides ``status``, so there is exactly one place
    to look when the business backend gets an answer it did not expect.
    """
    estimate: GraphEstimate | None = state.get("estimate")
    errors: list[str] = list(state.get("errors") or [])
    budgets: list[BudgetHit] = state.get("budgets") or []

    if estimate is None or not estimate.components:
        return {
            "estimate": GraphEstimate(
                status="insufficient_context",
                errors=errors,
                issues=["No components were costed."],
            )
        }

    references: dict[str, list[float]] = defaultdict(list)
    for hit in budgets:
        if hit.estimated_hours is not None:
            references[hit.component].append(hit.estimated_hours)

    verdict = validate_estimate(
        {
            "components": [
                {
                    "name": line.name,
                    "estimated_hours": line.estimated_hours,
                    "reference_amounts": references.get(line.name, []),
                }
                for line in estimate.components
            ],
            "total_hours": estimate.total_hours,
        }
    )

    # Nothing at all was grounded -> the honest answer is "I cannot estimate this",
    # not a total of 0h that reads like a cheap project.
    grounded = any(not line.unbudgeted for line in estimate.components)
    if not grounded:
        status = "insufficient_context"
    elif verdict["issues"] or errors:
        status = "needs_review"
    else:
        status = "ok"

    log.info(
        "graph_validate_and_consolidate",
        status=status,
        issues=len(verdict["issues"]),
        errors=len(errors),
    )
    return {
        "estimate": estimate.model_copy(
            update={"status": status, "issues": verdict["issues"], "errors": errors}
        )
    }
