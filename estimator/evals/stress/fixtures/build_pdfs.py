"""Generate deterministic PDF attachment fixtures for the stress suite.

Produces ``attach_5kb.pdf``, ``attach_20kb.pdf``, ``attach_50kb.pdf`` and
``attach_100kb.pdf`` next to this file. There is deliberately no ``attach_0kb``
fixture — "0 KB" in the stress matrix means *no attachment at all*, which the
runner models by simply not attaching a file.

The PDFs are NOT committed. They are deterministic, so the runner regenerates
any that are missing (``ensure_fixtures``); only this generator is versioned.

Determinism notes
-----------------
- ``set_compression(False)``: the page stream is stored verbatim, so file size
  grows linearly with text (easy, monotonic sizing) and there is no zlib
  version variance between machines.
- A frozen ``creation_date`` and fixed document metadata — fpdf2 otherwise
  stamps ``datetime.now()`` into the PDF, which would change the bytes (and the
  trailer ``/ID`` derived from them) on every run.
- Core font (Helvetica) + ASCII-only text: no embedded font program, and no
  Unicode-encoding surprises with the built-in fonts. pypdf extracts the text
  cleanly, which is what the attachment pipeline reads.

Sizes are approximate: each file is the smallest whole-item document that
reaches its target, so it lands at or just above the target (overshoot is at
most one requirement entry).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

FIXTURES_DIR = Path(__file__).resolve().parent
TARGETS_KB: tuple[int, ...] = (5, 20, 50, 100)
_KB = 1024

# Frozen so regeneration is byte-for-byte reproducible.
_FIXED_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)

_FONT = "Helvetica"
_FONT_SIZE = 11
_LINE_HEIGHT = 6  # mm

# The corpus is a synthetic-but-realistic Software Requirements Specification —
# this is the attachment text the multi-turn scenarios feed into the estimator.
# It is intentionally neutral with respect to the scenario projects (Nimbus /
# Aurora / Helios): a generic spec document acts as realistic context noise so
# the suite can isolate the effect of attachment *size*, without injecting any of
# the tracked facts the MemoryDriftMetric watches for.
_TITLE = "Project Requirements Specification"

# Fixed front matter, always present even in the smallest fixture.
_PREAMBLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Document Control",
        (
            "This document records the agreed scope and requirements for the engagement and "
            "supersedes any prior draft circulated to the parties.",
            "It is a synthetic specification used as an attachment in automated estimation "
            "tests; the content is representative, not contractual.",
        ),
    ),
    (
        "Executive Summary",
        (
            "The client requires a multi-tenant web platform that centralises their core "
            "operational workflow and replaces a patchwork of spreadsheets and manual steps.",
            "The platform must support self-service onboarding, role-based collaboration, and "
            "exportable reporting from the first release.",
        ),
    ),
    (
        "Scope and Objectives",
        (
            "In scope: account management, the core domain workflow, reporting, and the "
            "third-party integrations enumerated in the sections that follow.",
            "Out of scope: bespoke hardware, on-premise installation, and migration of legacy "
            "data beyond the documented import format.",
        ),
    ),
    (
        "Stakeholders",
        (
            "Primary stakeholders are the product owner, the delivery team, and the client's "
            "operations and compliance leads.",
            "Final acceptance is signed off by the product owner against the acceptance "
            "criteria recorded with each requirement below.",
        ),
    ),
)

# Section themes the variable body cycles through, with their requirement-id prefix.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Functional Requirements", "FR"),
    ("Non-Functional Requirements", "NFR"),
    ("Security and Compliance", "SEC"),
    ("Integrations", "INT"),
    ("Data and Reporting", "DAT"),
    ("Operations and Support", "OPS"),
)
_ITEMS_PER_SECTION = 8

# Pool of realistic, ASCII-only "shall" statements. Requirement bodies are drawn
# from this pool by index, so the text varies while staying a pure function of i.
_REQUIREMENT_SENTENCES: tuple[str, ...] = (
    "The system shall authenticate every user before granting access to any protected resource.",
    "The system shall isolate tenant data so that one organisation can never read another's records.",
    "The system shall record every state-changing action in an append-only audit log with actor and timestamp.",
    "The system shall allow users to export their reports to CSV and PDF on demand.",
    "The system shall enforce role-based access control across administrators, editors and read-only viewers.",
    "The system shall version its public API and deprecate endpoints with no less than ninety days notice.",
    "The system shall rate limit requests per tenant to protect shared backend capacity.",
    "The system shall deliver outbound webhooks for domain events with at-least-once semantics.",
    "The system shall present its interface in English and Spanish, with additional locales pluggable.",
    "The system shall reconcile external data sources on a schedule and retry transient failures with backoff.",
    "The system shall encrypt sensitive fields at rest and all traffic in transit.",
    "The system shall honour data retention and deletion policies in line with applicable regulation.",
    "The system shall load primary dashboards within two seconds at the ninety-fifth percentile.",
    "The system shall remain available at or above the agreed monthly uptime objective.",
    "The system shall emit structured logs, metrics and traces for every request path.",
    "The system shall provide an administrative view for support staff to inspect tenant status.",
    "The system shall validate all inbound payloads and reject malformed requests with a clear error.",
    "The system shall paginate and filter large result sets to keep responses bounded.",
    "The system shall back up persistent data daily and verify restorability on a regular cadence.",
    "The system shall expose a health endpoint suitable for automated liveness and readiness probes.",
)


def _requirement(i: int) -> str:
    """Return the i-th requirement line: a stable, ASCII, spec-style statement."""
    section_idx = i // _ITEMS_PER_SECTION
    _, prefix = _SECTIONS[section_idx % len(_SECTIONS)]
    sentence = _REQUIREMENT_SENTENCES[(i * 5 + 2) % len(_REQUIREMENT_SENTENCES)]
    return f"{prefix}-{i + 1:04d}  {sentence}"


def _section_heading(i: int) -> str:
    """Heading for the section that requirement ``i`` opens (only at boundaries)."""
    section_idx = i // _ITEMS_PER_SECTION
    base, _ = _SECTIONS[section_idx % len(_SECTIONS)]
    wrap = section_idx // len(_SECTIONS)
    return base if wrap == 0 else f"{base} (continued, part {wrap + 1})"


def _heading(pdf: FPDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font(_FONT, style="B", size=12)
    pdf.multi_cell(0, 6, text)
    pdf.ln(1)
    pdf.set_font(_FONT, size=_FONT_SIZE)


def _body(pdf: FPDF, text: str) -> None:
    pdf.set_font(_FONT, size=_FONT_SIZE)
    pdf.multi_cell(0, _LINE_HEIGHT, text)
    pdf.ln(1)


def _render(n_items: int, label: str) -> bytes:
    """Render the spec with ``n_items`` requirement entries to PDF bytes.

    The fixed front matter is always emitted; ``n_items`` (the only knob the
    sizing search turns) controls how many requirement lines follow, which is
    what scales the file to its target size.
    """
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_compression(False)
    pdf.set_title(_TITLE)
    pdf.set_author("evals.stress")
    pdf.set_subject("synthetic attachment corpus for the multi-turn stress suite")
    pdf.set_creator("evals.stress.fixtures.build_pdfs")
    pdf.set_producer("evals.stress.fixtures.build_pdfs")
    pdf.creation_date = _FIXED_DATE
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font(_FONT, style="B", size=16)
    pdf.multi_cell(0, 8, _TITLE)
    pdf.ln(1)
    pdf.set_font(_FONT, size=9)
    pdf.multi_cell(0, 5, f"Synthetic attachment fixture: {label}")
    pdf.ln(1)

    for heading, paragraphs in _PREAMBLE:
        _heading(pdf, heading)
        for paragraph in paragraphs:
            _body(pdf, paragraph)

    for i in range(n_items):
        if i % _ITEMS_PER_SECTION == 0:
            _heading(pdf, _section_heading(i))
        _body(pdf, _requirement(i))

    return bytes(pdf.output())


def _build_to_size(target_bytes: int, label: str) -> bytes:
    """Smallest whole-item document whose size is >= ``target_bytes``.

    File size is monotonic in the requirement-item count, so we bracket with doubling
    and then binary-search the boundary — deterministic and fast.
    """
    hi = 1
    while len(_render(hi, label)) < target_bytes:
        hi *= 2
    lo = hi // 2  # invariant: size(lo) < target <= size(hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(_render(mid, label)) >= target_bytes:
            hi = mid
        else:
            lo = mid
    return _render(hi, label)


def _path_for(kb: int, out_dir: Path) -> Path:
    return out_dir / f"attach_{kb}kb.pdf"


def build_all(out_dir: Path = FIXTURES_DIR) -> dict[int, Path]:
    """(Re)build every fixture and return ``{kb: path}``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    for kb in TARGETS_KB:
        path = _path_for(kb, out_dir)
        path.write_bytes(_build_to_size(kb * _KB, path.name))
        results[kb] = path
    return results


def ensure_fixtures(out_dir: Path = FIXTURES_DIR) -> dict[int, Path]:
    """Build only the fixtures that are missing; return ``{kb: path}``.

    Intended for the runner: cheap to call every run, rebuilds nothing that is
    already on disk (the output is deterministic, so an existing file is correct).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    for kb in TARGETS_KB:
        path = _path_for(kb, out_dir)
        if not path.exists():
            path.write_bytes(_build_to_size(kb * _KB, path.name))
        results[kb] = path
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIXTURES_DIR,
        help="Directory to write the fixtures into (default: alongside this script).",
    )
    args = parser.parse_args()
    for kb, path in build_all(args.out_dir).items():
        size = path.stat().st_size
        print(f"{path.name:<18} target={kb * _KB:>7} B  actual={size:>7} B  ({size / _KB:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
