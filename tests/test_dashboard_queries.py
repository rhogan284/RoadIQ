"""Tests for the dashboard's query functions, kept free of Streamlit so they
stay testable without a browser.

`run_options` is a plain SELECT with no branching -- not exercised here.
`coverage_series` is the one that could silently lie, so it is the one test
this file keeps.
"""
from datetime import datetime, timezone

import pytest

from edgecv.dashboard.app import coverage_series
from edgecv.db.repository import Repository
from tests.test_repository import _result

pytestmark = pytest.mark.integration

RUN_ID = "88888888-8888-8888-8888-888888888888"


@pytest.fixture
def pg_conn(clean_db):
    return clean_db


@pytest.fixture
def repo(pg_conn):
    return Repository(pg_conn)


def test_coverage_counts_frames_that_were_never_processed(pg_conn, repo):
    """A dropped frame must still appear in frames_ingested — otherwise
    dropping frames silently improves the coverage number, which is the
    opposite of what the panel is for."""
    repo.write_results([_result(run_id=RUN_ID, seq=1)])
    # A frame that reached ingestion but never got an inference result at all
    # (e.g. the worker crashed before writing one). `frames` is the
    # denominator, so it must still be counted.
    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO frames (run_id, seq, captured_at, enqueued_at, width, "
            "height, source_ref, sha256) VALUES "
            "(%s, 2, %s, %s, 128, 128, 'fixtures/dropped.png', %s)",
            (RUN_ID, datetime(2026, 8, 19, 10, 1, tzinfo=timezone.utc),
             datetime(2026, 8, 19, 10, 1, tzinfo=timezone.utc), "e" * 64),
        )

    rows = coverage_series(pg_conn, RUN_ID)
    assert sum(r["frames_ingested"] for r in rows) == 2
    assert sum(r["frames_processed"] for r in rows) == 1
