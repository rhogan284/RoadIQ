import os
import psycopg
import pytest
import redis as redis_lib

PG_DSN = os.environ.get("PG_DSN", "postgresql://edgecv:edgecv@localhost:5432/edgecv")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def pg():
    """Connection to a live Postgres. Requires `make up`."""
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        yield conn


@pytest.fixture
def clean_db(pg):
    from edgecv.db.migrate import apply_migrations
    apply_migrations(pg)
    with pg.cursor() as cur:
        cur.execute("TRUNCATE detections, inferences, snippets, frames, "
                    "detectors, survey_runs, ground_truth, bench_runs")
    return pg


@pytest.fixture
def rds():
    client = redis_lib.from_url(REDIS_URL, decode_responses=False)
    client.flushdb()
    yield client
    client.flushdb()
