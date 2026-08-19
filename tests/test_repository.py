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
        detections=[Detection(defect_class="pothole", confidence=0.9,
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
    stats = repo.write_results([
        _result(seq=501, lat=-33.8688, lon=151.2093, speed_mps=16.7),
        _result(seq=502),                      # fixture frame, no GPS
    ])
    assert stats.frames == 2
    with pg_conn.cursor() as cur:
        cur.execute("SELECT seq, lat, lon FROM frames WHERE seq IN (501, 502) "
                    "ORDER BY seq")
        rows = cur.fetchall()
    assert float(rows[0][1]) == pytest.approx(-33.8688)
    assert rows[1][1] is None and rows[1][2] is None


def test_detections_get_a_populated_snippet_id(pg_conn, repo):
    """snippet_id must never be NULL on a written detection: a later
    end-to-end test asserts count(*) FROM detections WHERE snippet_id IS NULL
    == 0."""
    stats = repo.write_results([_result(seq=600)])
    assert stats.detections == 1
    assert stats.skipped_duplicates == 0
    with pg_conn.cursor() as cur:
        cur.execute("SELECT snippet_id FROM detections")
        snippet_ids = [r[0] for r in cur.fetchall()]
    assert snippet_ids and all(s is not None for s in snippet_ids)


def test_replay_after_a_crash_mid_batch_does_not_lose_detections(pg_conn, repo, monkeypatch):
    """Regression for a Critical defect: write_results ran without a
    transaction, so on an autocommit=True connection the inference row could
    commit before its detections did. A crash in that exact gap meant the
    inference was permanently "already recorded", so the next replay's
    `ON CONFLICT DO NOTHING` skipped straight past the detections that were
    never written, losing them for good.

    This forces that crash for real, inside an actual write_results() call
    (raising right after the inference row would have committed, before any
    detection/snippet is written), then replays the identical batch and
    asserts the detection comes back. A version of this test that instead
    pre-commits a "half-written" frame+inference by hand via separate,
    already-autocommitted SQL statements does NOT exercise the fix at all:
    that corruption exists before write_results is ever called, so no
    transaction inside write_results can undo it. The only way to prove the
    transaction wrapper matters is to crash while it is open."""
    result = _result(seq=700)

    def boom(self, cur, sha256, kind, **kw):
        raise RuntimeError("simulated crash before any detection/snippet write")

    monkeypatch.setattr(Repository, "_snippet_id", boom)
    with pytest.raises(RuntimeError):
        repo.write_results([result])
    monkeypatch.undo()

    # The stream redelivers the same frame; the worker replays the identical
    # batch for real this time.
    stats = repo.write_results([result])

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM detections d JOIN inferences i "
            "USING (inference_id) WHERE i.run_id = %s AND i.seq = %s",
            (result.run_id, result.seq),
        )
        recovered = cur.fetchone()[0]

    assert recovered == 1, f"detection lost on replay after a crash mid-batch: {stats}"
