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

# Arbitrary constant, shared by every caller: identifies "the migration run" as a
# single lockable resource. Any int64 works; this one has no other meaning.
MIGRATION_LOCK_KEY = 847_291_002


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply any unapplied migration files in filename order. Returns those applied.

    Two services (writer, feed-sim) both call this at startup against a possibly
    virgin database. `CREATE TABLE IF NOT EXISTS` is idempotent in *effect* but not
    *atomic* against a concurrent session doing the same thing: two callers can both
    decide the table doesn't exist yet and both try to create it, and one loses a
    UniqueViolation on the `pg_type` catalog (`pg_type_typname_nsp_index`) rather than
    silently no-op'ing. A session-level advisory lock serialises the whole run so only
    one caller is ever inside it at a time; the second caller then sees the
    already-applied versions in `schema_migrations` and does nothing.

    `pg_advisory_xact_lock` would be the natural choice, but every caller here
    connects with autocommit=True, which commits (and releases an xact lock) after
    every single statement -- it would give no protection at all. Session-level
    `pg_advisory_lock` plus an explicit `pg_advisory_unlock` in `finally` is what
    actually holds across the whole function on an autocommit connection.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
    try:
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
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))


def main() -> None:
    from edgecv.config import Settings

    settings = Settings.from_env()
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        applied = apply_migrations(conn)
    print(f"applied: {applied or 'nothing (up to date)'}")


if __name__ == "__main__":
    main()
