"""Per-run reads shared by the read API, the observability panel and the evaluator."""
from __future__ import annotations

import psycopg

from edgecv.bench.coverage import CoverageReport, coverage_from_rows


def _assert_run_exists(conn: psycopg.Connection, run_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM survey_runs WHERE run_id = %s", (run_id,))
        if cur.fetchone() is None:
            raise LookupError(f"no survey run {run_id!r}")


def latest_run_id(conn: psycopg.Connection) -> str:
    """The most recently started survey run, so the evaluator can be run with no
    arguments straight after a demo."""
    with conn.cursor() as cur:
        cur.execute("SELECT run_id::text FROM survey_runs "
                    "ORDER BY started_at DESC, run_id DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            raise LookupError("no survey runs recorded -- run the feed first")
        return row[0]


def run_coverage(conn: psycopg.Connection, run_id: str) -> CoverageReport:
    """Metres of road assessed vs lost for one run (success criterion 2).

    Reads every frame row including the ones with no fix, because the two kinds
    of absence mean opposite things — see `coverage.py`. A missing seq is road
    nobody assessed; a null lat on a present row is road we assessed but cannot
    place.
    """
    _assert_run_exists(conn, run_id)
    with conn.cursor() as cur:
        cur.execute("SELECT seq, lat, lon FROM frames WHERE run_id = %s "
                    "ORDER BY seq", (run_id,))
        rows = [(int(seq), None if lat is None else float(lat),
                 None if lon is None else float(lon))
                for seq, lat, lon in cur.fetchall()]
    return coverage_from_rows(rows)
