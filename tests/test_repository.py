"""Round-trip tests for the batched writer.

`_result` is exported for reuse by tests/test_dashboard_queries.py, which
seeds coverage-view fixtures through the same writer.
"""
from datetime import datetime, timezone

import pytest

from edgecv.contracts.detection import BBox, Detection, DetectorInfo, InferenceResult
from edgecv.db.repository import Repository

pytestmark = pytest.mark.integration


def _result(**kw):
    base = dict(
        run_id="11111111-1111-1111-1111-111111111111",
        seq=3,
        captured_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        width=256, height=256,
        source_ref="fixtures/defect_001.png",
        frame_sha256="b" * 64,
        detector=DetectorInfo(name="threshold", version="1.0.0", params={"k": 3}),
        worker_id="worker-1",
        started_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        latency_ms=12.5,
        status="ok",
        error=None,
        detections=[Detection(defect_class="scratch", confidence=0.9,
                              bbox=BBox(x=1, y=2, w=10, h=10), severity="minor")],
        snippet_sha256s=["c" * 64],
        thumbnail_sha256="d" * 64,
    )
    base.update(kw)
    return InferenceResult(**base)


@pytest.fixture
def pg_conn(clean_db):
    return clean_db


@pytest.fixture
def repo(pg_conn):
    return Repository(pg_conn)


def test_position_round_trips_through_the_writer(pg_conn, repo):
    repo.write_results([
        _result(seq=501, lat=-33.8688, lon=151.2093, speed_mps=16.7),
        _result(seq=502),                      # fixture frame, no GPS
    ])
    with pg_conn.cursor() as cur:
        cur.execute("SELECT seq, lat, lon FROM frames WHERE seq IN (501, 502) "
                    "ORDER BY seq")
        rows = cur.fetchall()
    assert float(rows[0][1]) == pytest.approx(-33.8688)
    assert rows[1][1] is None and rows[1][2] is None
