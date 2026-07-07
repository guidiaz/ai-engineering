"""Structural chunking for historical budgets.

One budget component maps to one embeddable chunk. Parent budget context is
prepended to each component so isolated line items stay traceable to client and
sector.
"""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

_EMBEDDING_MODEL = "text-embedding-3-small"


class JSONStructuralChunker:
    """Chunks budgets by JSON structure: one :class:`BudgetComponent` per chunk."""

    def __init__(self) -> None:
        self._encoder = tiktoken.encoding_for_model(_EMBEDDING_MODEL)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                text = self._build_text(budget, component)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata={
                            "budget_id": budget.budget_id,
                            "component_id": component.component_id,
                            "client_sector": budget.client_metadata.sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "complexity": component.complexity,
                            "estimated_hours": component.estimated_hours,
                        },
                        token_count=len(self._encoder.encode(text)),
                    )
                )
        return chunks

    @staticmethod
    def _build_text(budget: Budget, component: BudgetComponent) -> str:
        tech_stack = ", ".join(component.tech_stack)
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | "
            f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
            f"\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {tech_stack}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
