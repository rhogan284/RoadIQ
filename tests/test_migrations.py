import pytest
from edgecv.db.migrate import apply_migrations

pytestmark = pytest.mark.integration

# Per amendment A3, the run table was renamed (a "run" is one drive, not a
# factory shift). The four geospatial tables from amendment A5 (segments,
# defect instances, instance reviews, segment condition) are now included:
# A5 has landed in 001_initial.sql.
EXPECTED_TABLES = {
    "detectors", "survey_runs", "frames", "snippets",
    "inferences", "detections", "ground_truth", "bench_runs",
    "segments", "defect_instances", "instance_reviews", "segment_condition",
}

def test_apply_creates_all_tables(pg):
    apply_migrations(pg)
    with pg.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        tables = {row[0] for row in cur.fetchall()}
    assert EXPECTED_TABLES <= tables

def test_frames_is_partitioned(pg):
    apply_migrations(pg)
    with pg.cursor() as cur:
        cur.execute("SELECT relkind FROM pg_class WHERE relname='frames'")
        assert cur.fetchone()[0] == "p"     # 'p' = partitioned table

def test_apply_is_idempotent(pg):
    apply_migrations(pg)
    apply_migrations(pg)          # must not raise
    with pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM schema_migrations WHERE version='001_initial'")
        assert cur.fetchone()[0] == 1

def test_run_coverage_view_exists(pg):
    apply_migrations(pg)
    with pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_views WHERE viewname='run_coverage_1min'")
        assert cur.fetchone()[0] == 1

def test_run_coverage_view_does_not_fan_out_across_detectors(clean_db):
    """Regression for the bug where the view joined frames to inferences before
    aggregating: inferences is UNIQUE(run_id, seq, detector_id), so one frame
    processed by two detectors fanned out to two joined rows and every count
    was computed after the fan-out. One frame processed by two detectors must
    still count as ONE frame in frames_ingested and frames_without_fix."""
    run_id = "66666666-6666-6666-6666-666666666666"
    captured_at = "2026-08-19T10:00:00+10:00"
    with clean_db.cursor() as cur:
        cur.execute(
            "INSERT INTO survey_runs (run_id, authority_id, started_at, source_kind, "
            "source_ref, target_fps, transport) VALUES "
            "(%s, 'demo-council', %s, 'synthetic', 'generated', 10, 'reference')",
            (run_id, captured_at),
        )
        cur.execute(
            "INSERT INTO frames (run_id, seq, captured_at, enqueued_at, width, "
            "height, source_ref, sha256) VALUES "
            "(%s, 1, %s, %s, 128, 128, 'fixtures/f.png', %s)",
            (run_id, captured_at, captured_at, "a" * 64),
        )
        detector_ids = []
        for name in ("threshold", "edge"):
            cur.execute(
                "INSERT INTO detectors (name, version, params, params_hash) "
                "VALUES (%s, '1.0.0', '{}', %s) RETURNING detector_id",
                (name, name),
            )
            detector_ids.append(cur.fetchone()[0])
        for detector_id in detector_ids:
            cur.execute(
                "INSERT INTO inferences (run_id, seq, captured_at, detector_id, "
                "worker_id, started_at, latency_ms, status) VALUES "
                "(%s, 1, %s, %s, 'w1', %s, 5.0, 'ok')",
                (run_id, captured_at, detector_id, captured_at),
            )
        cur.execute(
            "SELECT frames_ingested, frames_processed, frames_without_fix "
            "FROM run_coverage_1min WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    assert row == (1, 1, 1), f"expected one frame counted once, got {row}"

def test_concurrent_apply_migrations_does_not_race(pg):
    """Regression: writer and feed-sim both call apply_migrations() at startup
    against a possibly-virgin database. CREATE TABLE IF NOT EXISTS is idempotent
    in effect but not atomic against a concurrent session doing the same thing --
    before the advisory lock in migrate.py, concurrent callers reproducibly hit
    UniqueViolation on the pg_type catalog (pg_type_typname_nsp_index). Runs
    against a fresh, disposable database so the virgin-schema race is actually
    exercised, not an already-migrated one."""
    import concurrent.futures as cf
    import os
    import uuid

    import psycopg

    from edgecv.db.migrate import apply_migrations

    base_dsn = os.environ.get("PG_DSN", "postgresql://edgecv:edgecv@localhost:5432/edgecv")
    db_name = f"migrate_race_{uuid.uuid4().hex[:12]}"
    race_dsn = f"{base_dsn.rsplit('/', 1)[0]}/{db_name}"

    with pg.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{db_name}"')
    try:
        def go(_n):
            with psycopg.connect(race_dsn, autocommit=True) as conn:
                return apply_migrations(conn)

        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(go, range(4)))  # propagates any worker exception

        with psycopg.connect(race_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM schema_migrations WHERE version='001_initial'")
            assert cur.fetchone()[0] == 1

        # Exactly one of the four callers should have seen an unapplied migration.
        assert sum(1 for r in results if r) == 1
    finally:
        with pg.cursor() as cur:
            cur.execute(f'DROP DATABASE "{db_name}"')
