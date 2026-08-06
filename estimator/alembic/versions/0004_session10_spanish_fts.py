"""Session 10 — switch chunk full-text search to the Spanish configuration.

Revision ID: 0004_session10_spanish_fts
Revises: 0003_session10_fts
Create Date: 2026-08-06 00:00:00

Supersedes 0003_session10_fts's choice of the ``english`` text-search
configuration. The exercise's target corpus is Spanish budget descriptions, so
``content_tsv`` must be stemmed and stop-worded with ``spanish`` instead —
matched by the ORM model's ``Computed(...)`` expression and the repository's
``plainto_tsquery`` call, kept consistent with this migration on purpose.

PostgreSQL has no ``ALTER COLUMN ... SET EXPRESSION`` for generated columns:
the expression is immutable once created, so the only way to change it is to
drop the column (and its dependent GIN index) and re-add it with the new
expression. Postgres recomputes ``content_tsv`` for every existing row as part
of the ``ADD COLUMN``, so no separate backfill step is needed.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0004_session10_spanish_fts"
down_revision: Union[str, None] = "0003_session10_fts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Generated columns can't be altered in place — drop and re-add with the
    # new expression (raw DDL: no first-class Alembic/SQLAlchemy helper for a
    # STORED generated column).
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED"
    )
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    # Restore 0003's english-configured column.
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )
