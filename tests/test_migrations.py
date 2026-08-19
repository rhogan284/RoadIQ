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
