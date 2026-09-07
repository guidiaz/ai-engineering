#!/usr/bin/env python3
"""Session 13 - run the estimation graph over a transcript.

Where ``run_agent_s12.py`` drove a hand-written loop that decided its own control
flow, this runs the same work as an explicit five-node graph:

    START -> extract_requirements -> classify_components -> search_budgets
          -> generate_estimate -> validate_and_consolidate -> END

It configures Logfire (one span per node, nested under one run span), prints the
topology, then the per-node state deltas as they are produced (via
``astream(stream_mode="updates")``), then the final ``GraphEstimate`` with its
``status``.

    # 1) offline: the S12 student stub stands in for retrieval (NO database)
    uv run python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt --stub

    # 2) the real run, against the S9/S10 retrieval pipeline
    docker compose exec estimator python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt

    # 3) with Postgres checkpointing, under a known estimation id
    docker compose exec estimator python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt \
        --checkpoint --estimation-id demo-001

    # 4) prove resume works: interrupt mid-run, then continue the same thread
    docker compose exec estimator python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt --demo-resume

The real run needs the stack up and the historical-task corpus ingested
(``scripts/build_task_corpus.py --ingest``); ``--stub`` needs neither.
``--checkpoint`` / ``--demo-resume`` need the project's Postgres reachable.

Logfire: spans go to the console always, and additionally to the Logfire cloud
UI when a ``LOGFIRE_TOKEN`` is set (``send_to_logfire="if-token-present"``). No
account is needed to see the trace.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logfire  # noqa: E402

from app.generation.agentic.agent_schemas import SearchBudgetsArgs  # noqa: E402
from app.generation.agentic.graph import compile_estimation_graph  # noqa: E402
from app.generation.agentic.graph.builder import (  # noqa: E402
    NODE_SEQUENCE,
    build_estimation_graph,
)
from app.generation.agentic.graph.checkpointing import postgres_checkpointer  # noqa: E402
from app.generation.agentic.graph.schemas import GraphEstimate  # noqa: E402

# The Session 12 kit already ships an offline retrieval stub; reuse it rather
# than shipping a second copy in the session-13 folder.
STUB_PATH = REPO_ROOT / "exercises" / "session-12" / "reference_retrieval.py"

SEPARATOR = "=" * 78

# The node the resume demo stops before. Anything after the first LLM calls works;
# this one is chosen because everything before it is the expensive part.
RESUME_INTERRUPT_NODE = "generate_estimate"


def _configure_logfire() -> None:
    """Console spans always; cloud too when a token is present."""
    logfire.configure(
        service_name="estimator-graph-s13",
        send_to_logfire="if-token-present",
        console=logfire.ConsoleOptions(colors="never"),
    )
    # NOTE: logfire.instrument_litellm() is deliberately NOT called. It was tried
    # and produced ZERO spans here: the LLM nodes reach LiteLLM through Instructor
    # (LLMWrapper.complete_structured), which that instrumentation does not hook.
    # The structlog `llm_structured_call_started` / `llm_structured_call_completed`
    # events already report model and latency, and they appear inside the node
    # span they belong to. Re-add the instrumentation only if a future node calls
    # litellm directly.


def _load_stub_backend():
    """Load the S12 safety-net retrieval stub and adapt it to a RetrievalBackend."""
    spec = importlib.util.spec_from_file_location("s13_reference_retrieval", STUB_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load stub retrieval from {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def stub_backend(args: SearchBudgetsArgs) -> list[dict]:
        filters = args.filters.model_dump() if args.filters else None
        return module.search_budgets_stub(args.query, filters)

    return stub_backend


def _render_topology(compiled) -> str:
    graph = compiled.get_graph()
    lines = [SEPARATOR, "GRAPH TOPOLOGY", SEPARATOR]
    try:
        # Needs grandalf, which is not a project dependency; fall back below.
        return "\n".join(lines + [graph.draw_ascii()])
    except ImportError:
        pass

    # graph.edges comes back unordered, which reads badly for a sequential graph.
    # Walk the declared sequence instead so the chain is shown in flow order, and
    # only then list anything the walk did not cover (a future branch would show
    # up here rather than being silently dropped).
    edges = {(edge.source, edge.target) for edge in graph.edges}
    chain = ["__start__", *NODE_SEQUENCE, "__end__"]
    walked: set[tuple[str, str]] = set()
    for source, target in zip(chain, chain[1:]):
        if (source, target) in edges:
            lines.append(f"  {source}  ->  {target}")
            walked.add((source, target))
    for source, target in sorted(edges - walked):
        lines.append(f"  {source}  ->  {target}   (not on the main chain)")
    return "\n".join(lines)


def _render_estimate(estimate: GraphEstimate) -> str:
    lines = [SEPARATOR, f"FINAL ESTIMATE   status={estimate.status}", SEPARATOR]
    if not estimate.components:
        lines.append("  (no components were costed)")
    for component in estimate.components:
        cited = ", ".join(str(c) for c in component.cited_chunk_ids) or "none"
        flag = "  [UNBUDGETED]" if component.unbudgeted else ""
        lines.append(
            f"  - {component.name} ({component.category}): {component.estimated_hours}h{flag}"
        )
        lines.append(f"      refs: {component.reference_count}   sources: {cited}")
    lines.append("")
    lines.append(f"  TOTAL: {estimate.total_hours}h")
    if estimate.issues:
        lines.append("  issues:")
        lines.extend(f"    - {issue}" for issue in estimate.issues)
    if estimate.errors:
        lines.append("  errors:")
        lines.extend(f"    - {error}" for error in estimate.errors)
    return "\n".join(lines)


def _summarise_delta(node: str, update: dict) -> str:
    """One readable line per node, showing what that node added to the state."""
    parts: list[str] = []
    for key, value in (update or {}).items():
        if isinstance(value, list):
            parts.append(f"{key}+={len(value)}")
        elif isinstance(value, GraphEstimate):
            parts.append(f"estimate(status={value.status}, total={value.total_hours}h)")
        else:
            parts.append(key)
    return f"  {node:26} {'  '.join(parts) or '(no update)'}"


async def _stream(compiled, payload, config, collected: list[str], *, label: str) -> dict:
    """Drive the graph, printing and collecting one line per completed node.

    Wrapped in a root span so the five node spans nest under one trace, the same
    hierarchy ``run_estimation_graph`` produces. The script streams rather than
    calling that helper because it needs the per-node updates as they arrive.
    """
    final_state: dict = {}
    thread_id = config["configurable"]["thread_id"]
    with logfire.span(label, estimation_id=thread_id) as span:
        async for chunk in compiled.astream(payload, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                line = _summarise_delta(node, update)
                print(line)
                collected.append(line)
                final_state.update(update or {})
        estimate = final_state.get("estimate")
        if estimate is not None:
            span.set_attribute("status", estimate.status)
            span.set_attribute("total_hours", estimate.total_hours)
    return final_state


async def _count_checkpoints(saver, thread_id: str) -> int:
    """How many checkpoints Postgres holds for this thread."""
    return sum([1 async for _ in saver.alist({"configurable": {"thread_id": thread_id}})])


# --------------------------------------------------------------------------- #
# Resume demo                                                                 #
# --------------------------------------------------------------------------- #
async def _demo_resume(transcript: str, backend, estimation_id: str) -> int:
    """Interrupt a run mid-graph, then continue the SAME thread and show replay.

    Re-invoking a thread that already reached END would re-run the graph from the
    start, not resume it — resume is about picking up an *unfinished* run. So the
    first pass compiles with ``interrupt_before`` to stop deliberately, exactly as
    a crash or a human approval gate would leave it.
    """
    lines: list[str] = []
    config = {"configurable": {"thread_id": estimation_id}}

    async with postgres_checkpointer() as saver:
        print(SEPARATOR)
        print(f"PASS 1 - runs until it is interrupted before {RESUME_INTERRUPT_NODE!r}")
        print(SEPARATOR)
        # Same nodes and same checkpointer as a normal compile; the only
        # difference is that this one stops before RESUME_INTERRUPT_NODE.
        interrupted = build_estimation_graph(backend=backend).compile(
            checkpointer=saver, interrupt_before=[RESUME_INTERRUPT_NODE]
        )
        first = await _stream(
            interrupted,
            {"transcript": transcript, "estimation_id": estimation_id},
            config,
            lines,
            label="estimation_graph.pass1",
        )
        ran_first = [line.split()[0] for line in lines]
        components_before = len(first.get("components") or [])
        budgets_before = len(first.get("budgets") or [])
        checkpoints_after_first = await _count_checkpoints(saver, estimation_id)

        print()
        print(SEPARATOR)
        print("PASS 2 - resumes the same thread_id, input=None")
        print(SEPARATOR)
        resume_lines: list[str] = []
        resumed = compile_estimation_graph(backend=backend, checkpointer=saver)
        # input=None means "continue from the checkpoint", not "start over".
        final = await _stream(resumed, None, config, resume_lines, label="estimation_graph.resume")
        ran_second = [line.split()[0] for line in resume_lines]
        checkpoints_after_second = await _count_checkpoints(saver, estimation_id)

        state = await resumed.aget_state(config)
        components_after = len(state.values.get("components") or [])
        budgets_after = len(state.values.get("budgets") or [])

    estimate = final.get("estimate")
    print()
    print(SEPARATOR)
    print("RESUME EVIDENCE")
    print(SEPARATOR)
    print(f"  thread_id (estimation_id) : {estimation_id}")
    print(f"  pass 1 executed nodes     : {', '.join(ran_first) or '(none)'}")
    print(f"  pass 2 executed nodes     : {', '.join(ran_second) or '(none)'}")
    print(f"  checkpoints after pass 1  : {checkpoints_after_first}")
    print(f"  checkpoints after pass 2  : {checkpoints_after_second}")
    print(f"  components  before/after  : {components_before} / {components_after}")
    print(f"  budgets     before/after  : {budgets_before} / {budgets_after}")

    if "__interrupt__" not in ran_first:
        # Pass 1 never reached the interrupt, so there is nothing to resume. That
        # is an upstream problem (an LLM hiccup leaving no components, a retrieval
        # outage), not a broken checkpointer — say so instead of failing silently.
        print()
        print(
            f"  NOTE: pass 1 never reached {RESUME_INTERRUPT_NODE!r}, so this run"
            " cannot demonstrate resume."
        )
        print("        Re-run it; this says nothing about the checkpointer.")
        return 1

    replayed_not_rerun = not (set(ran_second) & set(ran_first))
    state_preserved = components_before == components_after and budgets_before == budgets_after
    print()
    print(f"  pass 2 re-ran none of pass 1's nodes : {replayed_not_rerun}")
    print(f"  state carried across the two passes  : {state_preserved}")
    print(
        "  -> extract_requirements is an LLM call, so identical counts after a "
        "restart\n     are only possible because the state was replayed, not re-inferred."
    )
    if estimate is not None:
        print()
        print(_render_estimate(estimate))
    return 0 if (replayed_not_rerun and state_preserved) else 1


# --------------------------------------------------------------------------- #
# Normal run                                                                  #
# --------------------------------------------------------------------------- #
async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    backend = _load_stub_backend() if args.stub else None
    transcript = transcript_path.read_text(encoding="utf-8")
    estimation_id = args.estimation_id or str(uuid4())

    if args.demo_resume:
        return await _demo_resume(transcript, backend, estimation_id)

    print(f"transcript    : {transcript_path}")
    print(f"retrieval     : {'stub (offline)' if args.stub else 'retrieve() pipeline'}")
    print(f"estimation_id : {estimation_id}")
    print(f"checkpointer  : {'postgres' if args.checkpoint else 'none'}")
    print()

    config = {"configurable": {"thread_id": estimation_id}}
    update_lines = [SEPARATOR, "NODE UPDATES", SEPARATOR]
    checkpoints = None

    if args.checkpoint:
        async with postgres_checkpointer() as saver:
            compiled = compile_estimation_graph(backend=backend, checkpointer=saver)
            topology = _render_topology(compiled)
            print(topology)
            print()
            print("\n".join(update_lines))
            final_state = await _stream(
                compiled,
                {"transcript": transcript, "estimation_id": estimation_id},
                config,
                update_lines,
                label="estimation_graph",
            )
            checkpoints = await _count_checkpoints(saver, estimation_id)
    else:
        compiled = compile_estimation_graph(backend=backend)
        topology = _render_topology(compiled)
        print(topology)
        print()
        print("\n".join(update_lines))
        final_state = await _stream(
            compiled,
            {"transcript": transcript, "estimation_id": estimation_id},
            config,
            update_lines,
            label="estimation_graph",
        )

    print()
    if checkpoints is not None:
        print(f"checkpoints written for thread {estimation_id}: {checkpoints}")
        print()

    estimate = final_state.get("estimate")
    if estimate is None:
        estimate = GraphEstimate(
            status="insufficient_context",
            issues=["The graph finished without producing an estimate."],
        )
    rendered = _render_estimate(estimate)
    print(rendered)

    if args.out:
        sections = [
            f"transcript    : {transcript_path}",
            f"retrieval     : {'stub (offline)' if args.stub else 'retrieve() pipeline'}",
            f"estimation_id : {estimation_id}",
            f"checkpointer  : {'postgres' if args.checkpoint else 'none'}",
            "",
            topology,
            "",
            "\n".join(update_lines),
            "",
        ]
        if checkpoints is not None:
            sections += [f"checkpoints written for thread {estimation_id}: {checkpoints}", ""]
        sections.append(rendered)
        Path(args.out).write_text("\n".join(sections) + "\n", encoding="utf-8")
        print(f"\n(run written to {args.out})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Session 13 estimation graph.")
    parser.add_argument("transcript", help="Path to a transcript text file.")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline S12 retrieval stub instead of the real pipeline.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Persist state to the project's Postgres, keyed by the estimation id.",
    )
    parser.add_argument(
        "--estimation-id",
        help="Identifier for this estimation; used as the checkpointer thread_id. "
        "Reuse it to continue an interrupted run. Minted when omitted.",
    )
    parser.add_argument(
        "--demo-resume",
        action="store_true",
        help="Interrupt a checkpointed run mid-graph, then resume the same thread "
        "and report which nodes were replayed rather than re-executed.",
    )
    parser.add_argument("--out", help="Write the rendered run to this file.")
    args = parser.parse_args()
    _configure_logfire()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
