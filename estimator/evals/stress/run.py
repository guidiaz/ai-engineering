"""Multi-turn stress runner: drive scenarios x attachment-size x repeats and
score each turn with the three stress metrics, writing one CSV row per turn.

Transport
---------
Default is in-process via ``httpx.ASGITransport`` (no server to start, same
ergonomics as ``evals/run.py``'s ``TestClient``). ``--http BASE`` targets a
running server instead. Either way the per-turn observability and the memory
snapshot are read back from ``GET /sessions/{id}`` — the service stashes the
latest ``turn_observed`` record on the session and exposes a bucketed
``memory_snapshot``, so the runner never has to scrape the estimator's stdout.

Cost warning
------------
Every turn is a real LLM call (estimation + summariser + metadata extractor).
The full matrix is large: 4 scenarios x 5 attachment sizes x N turns x repeats.
Use ``--dry-run`` to print the matrix first, and ``--scenarios`` /
``--attachment-sizes`` / ``--max-turns`` / ``--repeats`` to scope a run.

Usage::

    uv run python -m evals.stress.run --dry-run
    uv run python -m evals.stress.run --scenarios growing --attachment-sizes 0 --max-turns 6
    uv run python -m evals.stress.run --http http://localhost:8000 --out report.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import httpx

from evals.stress import scenarios
from evals.stress.fixtures.build_pdfs import ensure_fixtures
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)

# 0 KB means "no attachment" (absence), modelled by sending no file.
ATTACHMENT_SIZES: tuple[int, ...] = (0, 5, 20, 50, 100)

# A sensible project_type per scenario (the endpoint requires one).
PROJECT_TYPE: dict[str, str] = {
    "growing": "web_saas",
    "pivot": "mobile_app",
    "contradiction_literal": "internal_tool",
    "contradiction_anchored": "internal_tool",
}

CSV_FIELDS: list[str] = [
    "scenario",
    "attachment_kb",
    "repeat",
    "turn_index",
    "project_type",
    "status",
    "latency_ms",
    "cost_usd",
    "cost_total_usd",
    "tokens_in",
    "tokens_out",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "last_resolved_tier",
    "latency_pass",
    "latency_score",
    "cost_pass",
    "cost_score",
    "drift_pass",
    "drift_score",
    "drift_details",
    "fact",
]


# Keys (the text before the first ':') whose value is single-valued, so a later
# turn changing the value supersedes the earlier one — a real contradiction.
# Everything else accumulates: "feature: ..." piles up turn after turn, and tech
# facts ("stack includes ...") have no colon and accumulate too. Without this
# allowlist, every "feature:" fact would falsely supersede the previous one.
SINGLE_VALUED_KEYS = frozenset({"project name", "budget locked"})


def _partition_facts(facts: list[str]) -> tuple[list[str], list[str]]:
    """Split chronological facts into ``(expected_now, superseded)``.

    A fact whose key is in ``SINGLE_VALUED_KEYS`` and whose value changed in a
    later turn is a contradiction: the earlier value is superseded (should be
    *gone*), the latest is expected. Same-value restatements, colon-less facts,
    and list-valued keys (e.g. ``feature``) accumulate and are never superseded.
    """
    latest_by_key: dict[str, tuple[str, str]] = {}
    superseded: list[str] = []
    for fact in facts:
        if ":" not in fact:
            continue
        key, _, value = fact.partition(":")
        key, value = key.strip().lower(), value.strip().lower()
        if key in SINGLE_VALUED_KEYS:
            prev = latest_by_key.get(key)
            if prev is not None and prev[1] != value:
                superseded.append(prev[0])
            latest_by_key[key] = (fact, value)

    superseded_set = set(superseded)
    expected = list(dict.fromkeys(f for f in facts if f not in superseded_set))
    forbidden = list(dict.fromkeys(superseded))
    return expected, forbidden


def _turns_for(scenario_name: str, max_turns: int | None) -> list[scenarios.ScenarioTurn]:
    profile = scenarios.PROFILES[scenario_name]
    return scenarios.first_n_turns(profile, max_turns) if max_turns else profile


async def _run_conversation(
    client: httpx.AsyncClient,
    *,
    scenario_name: str,
    turns: list[scenarios.ScenarioTurn],
    attachment_kb: int,
    repeat: int,
    attachment_path: Path | None,
    latency_metric: LatencyBudgetMetric,
    cost_metric: CostBudgetMetric,
    drift_metric: MemoryDriftMetric,
    writer: csv.DictWriter,
) -> None:
    project_type = PROJECT_TYPE[scenario_name]
    resp = await client.post("/sessions")
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    observations: list[dict] = []
    facts_so_far: list[str] = []

    for turn_index, transcript, fact in turns:
        data = {
            "transcript": transcript,
            "project_type": project_type,
            "detail_level": "medium",
            "output_format": "phases_table",
        }
        files = None
        handle = None
        if attachment_path is not None:
            handle = attachment_path.open("rb")
            files = {"attachments": (attachment_path.name, handle, "application/pdf")}
        try:
            post = await client.post(f"/sessions/{session_id}/estimate", data=data, files=files)
        finally:
            if handle is not None:
                handle.close()

        if post.status_code != 200:
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "attachment_kb": attachment_kb,
                    "repeat": repeat,
                    "turn_index": turn_index,
                    "project_type": project_type,
                    "status": post.status_code,
                    "drift_details": str(post.text)[:200],
                    "fact": fact,
                }
            )
            # The conversation depends on history; a failed turn poisons the rest.
            print(
                f"  ! {scenario_name} kb={attachment_kb} rep={repeat} "
                f"turn {turn_index}: HTTP {post.status_code} — aborting conversation",
                file=sys.stderr,
            )
            return

        info = (await client.get(f"/sessions/{session_id}")).json()
        last_turn = info.get("last_turn") or {}
        snapshot = info.get("memory_snapshot") or {}

        observations.append(last_turn)
        facts_so_far.append(fact)
        expected, forbidden = _partition_facts(facts_so_far)

        latency = latency_metric.evaluate([last_turn])
        cost = cost_metric.evaluate(observations)
        drift = drift_metric.evaluate(expected, snapshot, forbidden)
        cost_total = sum(float(o.get("cost_usd", 0.0) or 0.0) for o in observations)

        writer.writerow(
            {
                "scenario": scenario_name,
                "attachment_kb": attachment_kb,
                "repeat": repeat,
                "turn_index": turn_index,
                "project_type": project_type,
                "status": 200,
                "latency_ms": last_turn.get("latency_ms"),
                "cost_usd": last_turn.get("cost_usd"),
                "cost_total_usd": round(cost_total, 6),
                "tokens_in": last_turn.get("tokens_in"),
                "tokens_out": last_turn.get("tokens_out"),
                "enriched_transcript_chars": last_turn.get("enriched_transcript_chars"),
                "attachments_total_chars": last_turn.get("attachments_total_chars"),
                "messages_in_window": last_turn.get("messages_in_window"),
                "anchors_count": last_turn.get("anchors_count"),
                "summary_chars": last_turn.get("summary_chars"),
                "last_resolved_tier": last_turn.get("last_resolved_tier"),
                "latency_pass": latency.passed,
                "latency_score": round(latency.score, 4),
                "cost_pass": cost.passed,
                "cost_score": round(cost.score, 4),
                "drift_pass": drift.passed,
                "drift_score": round(drift.score, 4),
                "drift_details": drift.details,
                "fact": fact,
            }
        )
        print(
            f"  {scenario_name} kb={attachment_kb} rep={repeat} "
            f"turn {turn_index:>2}: lat={'OK' if latency.passed else 'X'} "
            f"cost={'OK' if cost.passed else 'X'} "
            f"drift={'OK' if drift.passed else 'X'} ({drift.score:.2f})"
        )


async def _drive(args: argparse.Namespace) -> int:
    scenario_names = args.scenarios or list(scenarios.PROFILES)
    sizes = args.attachment_sizes or list(ATTACHMENT_SIZES)
    fixtures = ensure_fixtures()

    latency_metric = LatencyBudgetMetric(budget_ms=args.latency_budget_ms)
    cost_metric = CostBudgetMetric(budget_usd=args.cost_budget_usd)
    drift_metric = MemoryDriftMetric()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        if args.http:
            transport_cm = _http_client(args.http)
        else:
            transport_cm = _in_process_client(args.max_turns_store)

        async with transport_cm as client:
            for scenario_name in scenario_names:
                turns = _turns_for(scenario_name, args.max_turns)
                for attachment_kb in sizes:
                    attachment_path = fixtures[attachment_kb] if attachment_kb else None
                    for repeat in range(args.repeats):
                        print(
                            f"> {scenario_name} | attachment={attachment_kb}kb | "
                            f"repeat={repeat + 1}/{args.repeats} | turns={len(turns)}"
                        )
                        await _run_conversation(
                            client,
                            scenario_name=scenario_name,
                            turns=turns,
                            attachment_kb=attachment_kb,
                            repeat=repeat,
                            attachment_path=attachment_path,
                            latency_metric=latency_metric,
                            cost_metric=cost_metric,
                            drift_metric=drift_metric,
                            writer=writer,
                        )
    print(f"\nCSV written to {args.out}")
    return 0


def _http_client(base_url: str):
    return httpx.AsyncClient(base_url=base_url, timeout=180.0)


def _in_process_client(max_turns: int):
    """In-process ASGI client with the same session-store override pattern as
    evals/run.py: a singleton store so POST /sessions and the estimate calls
    land in the same place, and the app lifespan run so logging is configured.
    """
    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions.store import SessionStore

    eval_store = SessionStore(max_turns=max_turns)
    app.dependency_overrides[get_session_store] = lambda: eval_store

    class _Ctx:
        async def __aenter__(self):
            self._lifespan = app.router.lifespan_context(app)
            await self._lifespan.__aenter__()
            self._client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://stress",
                timeout=180.0,
            )
            return await self._client.__aenter__()

        async def __aexit__(self, *exc):
            await self._client.__aexit__(*exc)
            await self._lifespan.__aexit__(*exc)
            app.dependency_overrides.pop(get_session_store, None)

    return _Ctx()


def _print_dry_run(args: argparse.Namespace) -> int:
    scenario_names = args.scenarios or list(scenarios.PROFILES)
    sizes = args.attachment_sizes or list(ATTACHMENT_SIZES)
    total_turns = 0
    print("Dry run — planned matrix (no LLM calls):")
    for scenario_name in scenario_names:
        n = len(_turns_for(scenario_name, args.max_turns))
        for attachment_kb in sizes:
            for repeat in range(args.repeats):
                total_turns += n
        print(f"  {scenario_name}: {n} turns x {len(sizes)} sizes x {args.repeats} repeats")
    print(
        f"\nTotal: {len(scenario_names)} scenarios x {len(sizes)} sizes x "
        f"{args.repeats} repeats = {total_turns} turns (each = 1 real estimation call "
        f"plus summariser/metadata calls)."
    )
    return 0


def _csv_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _csv_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=_csv_str_list,
        default=None,
        help=f"Comma list; default all of {list(scenarios.PROFILES)}",
    )
    parser.add_argument(
        "--attachment-sizes",
        type=_csv_int_list,
        default=None,
        help=f"Comma list of KB sizes; default {list(ATTACHMENT_SIZES)} (0 = no attachment)",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Cap turns per conversation (first N); default = full profile (20).",
    )
    parser.add_argument(
        "--max-turns-store",
        type=int,
        default=6,
        dest="max_turns_store",
        help="Sliding-window cap for the in-process session store (default 6).",
    )
    parser.add_argument(
        "--http", default=None, help="Target a running server instead of in-process."
    )
    parser.add_argument("--latency-budget-ms", type=float, default=60_000.0)
    parser.add_argument("--cost-budget-usd", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "stress_results.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        return _print_dry_run(args)
    return asyncio.run(_drive(args))


if __name__ == "__main__":
    raise SystemExit(main())
