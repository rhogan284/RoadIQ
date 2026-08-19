"""Apply SQL migrations idempotently, tracked in schema_migrations."""
from __future__ import annotations

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply any unapplied migration files in filename order. Returns those applied."""
    with conn.cursor() as cur:
        cur.execute(BOOTSTRAP)
        cur.execute("SELECT version FROM schema_migrations")
        already = {row[0] for row in cur.fetchall()}

    applied: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = path.stem
        if version in already:
            continue
        with conn.cursor() as cur:
            cur.execute(path.read_text())
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s) "
                "ON CONFLICT DO NOTHING",
                (version,),
            )
        applied.append(version)
    return applied


def main() -> None:
    from edgecv.config import Settings

    settings = Settings.from_env()
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        applied = apply_migrations(conn)
    print(f"applied: {applied or 'nothing (up to date)'}")


if __name__ == "__main__":
    main()
