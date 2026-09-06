#!/usr/bin/env python3
"""Session 13 - run the estimation graph over a transcript.

Where ``run_agent_s12.py`` drove a hand-written loop that decided its own control
flow, this runs the same work as an explicit five-node graph:

    START -> extract_requirements -> classify_components -> search_budgets
          -> generate_estimate -> validate_and_consolidate -> END

It prints the topology, then the per-node state deltas as they are produced
(via ``astream(stream_mode="updates")``), then the final ``GraphEstimate`` with
its ``status``.

    # 1) offline: the S12 student stub stands in for retrieval (NO database)
    uv run python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt --stub

    # 2) the real run, against the S9/S10 retrieval pipeline
    docker compose exec estimator python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt

    # 3) write the deliverable
    docker compose exec estimator python scripts/run_graph_s13.py \
        exercises/session-12/sample_transcript_complex.txt \
        --out exercises/session-13/example_run.txt

The real run needs the stack up and the historical-task corpus ingested
(``scripts/build_task_corpus.py --ingest``); ``--stub`` needs neither.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.generation.agentic.agent_schemas import SearchBudgetsArgs  # noqa: E402
from app.generation.agentic.graph import compile_estimation_graph  # noqa: E402
from app.generation.agentic.graph.builder import NODE_SEQUENCE  # noqa: E402
from app.generation.agentic.graph.schemas import GraphEstimate  # noqa: E402

# The Session 12 kit already ships an offline retrieval stub; reuse it rather
# than shipping a second copy in the session-13 folder.
STUB_PATH = REPO_ROOT / "exercises" / "session-12" / "reference_retrieval.py"

SEPARATOR = "=" * 78


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


async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    backend = _load_stub_backend() if args.stub else None
    transcript = transcript_path.read_text(encoding="utf-8")
    compiled = compile_estimation_graph(backend=backend)

    print(f"transcript : {transcript_path}")
    print(f"retrieval  : {'stub (offline)' if args.stub else 'retrieve() pipeline'}")
    print()
    print(_render_topology(compiled))
    print()

    update_lines = [SEPARATOR, "NODE UPDATES", SEPARATOR]
    print("\n".join(update_lines))
    final_state: dict = {}
    # stream_mode="updates" yields {node_name: partial_update} per completed node,
    # which is exactly the "what did this node contribute" view the exercise wants.
    async for chunk in compiled.astream({"transcript": transcript}, stream_mode="updates"):
        for node, update in chunk.items():
            line = _summarise_delta(node, update)
            print(line)
            update_lines.append(line)
            final_state.update(update or {})
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
            f"transcript : {transcript_path}",
            f"retrieval  : {'stub (offline)' if args.stub else 'retrieve() pipeline'}",
            "",
            _render_topology(compiled),
            "",
            "\n".join(update_lines),
            "",
            rendered,
        ]
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
    parser.add_argument("--out", help="Write the rendered run to this file.")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
