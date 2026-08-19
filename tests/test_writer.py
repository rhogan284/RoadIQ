from datetime import datetime, timezone
import pytest
from edgecv.contracts.detection import BBox, Detection, DetectorInfo, InferenceResult
from edgecv.writer.main import drain_once

pytestmark = pytest.mark.integration

RUN_ID = "11111111-1111-1111-1111-111111111111"
T0 = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)


def result(seq: int, n_detections: int = 0) -> InferenceResult:
    dets = [Detection(defect_class="pothole", confidence=0.9,
                      bbox=BBox(x=5, y=5, w=20, h=20), severity="major")
            for _ in range(n_detections)]
    return InferenceResult(
        run_id=RUN_ID, seq=seq, captured_at=T0, width=128, height=128,
        source_ref=f"img_{seq}.png", frame_sha256=f"f{seq}".ljust(64, "0"),
        detector=DetectorInfo(name="threshold", version="1.0.0", params={"k": 3}),
        worker_id="w1", started_at=T0, latency_ms=5.0, status="ok", error=None,
        detections=dets,
        snippet_sha256s=[f"c{seq}{i}".ljust(64, "0") for i in range(n_detections)],
        thumbnail_sha256=f"t{seq}".ljust(64, "0") if dets else None,
    )


def count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


@pytest.fixture
def seeded(clean_db):
    from edgecv.db.repository import Repository
    repo = Repository(clean_db)
    repo.upsert_run(run_id=RUN_ID, authority_id="demo-council", started_at=T0,
                    source_kind="synthetic", source_ref="generated",
                    target_fps=15, prevalence=0.1, transport="reference")
    return repo


def test_drains_results_into_postgres(rds, clean_db, seeded):
    for seq in range(5):
        rds.xadd("results", {"json": result(seq, 1).to_json()})
    stats = drain_once(rds, seeded, stream="results", group="writers",
                       consumer="w", batch=10)
    assert stats.frames == 5
    assert count(clean_db, "detections") == 5


def test_acks_so_entries_are_not_reprocessed(rds, clean_db, seeded):
    rds.xadd("results", {"json": result(1, 1).to_json()})
    drain_once(rds, seeded, stream="results", group="writers", consumer="w", batch=10)
    assert drain_once(rds, seeded, stream="results", group="writers",
                      consumer="w", batch=10).frames == 0


def test_redelivered_result_does_not_duplicate_detections(rds, clean_db, seeded):
    payload = result(1, 2).to_json()
    rds.xadd("results", {"json": payload})
    drain_once(rds, seeded, stream="results", group="writers", consumer="w", batch=10)
    rds.xadd("results", {"json": payload})       # same result delivered twice
    drain_once(rds, seeded, stream="results", group="writers", consumer="w", batch=10)
    assert count(clean_db, "detections") == 2    # NOT 4


def test_empty_stream_is_a_no_op(rds, seeded):
    stats = drain_once(rds, seeded, stream="results", group="writers",
                       consumer="w", batch=10)
    assert stats.frames == 0 and stats.detections == 0


def test_own_pending_entry_from_a_prior_crash_is_still_processed(rds, clean_db, seeded):
    """A crash between XREADGROUP and XACK leaves the entry delivered to this
    consumer but unacked, sitting in its PEL. ">" alone never returns it again --
    drain_once must drain "0" (this consumer's own pending list) first."""
    from edgecv.writer.main import _ensure_group

    rds.xadd("results", {"json": result(1, 1).to_json()})
    _ensure_group(rds, "results", "writers")
    # Simulate the crash: deliver the entry to writer-1 and never ack it.
    delivered = rds.xreadgroup("writers", "writer-1", {"results": ">"}, count=10)
    assert delivered and delivered[0][1]        # sanity: it really was delivered

    stats = drain_once(rds, seeded, stream="results", group="writers",
                       consumer="writer-1", batch=10)
    assert stats.frames == 1
    assert count(clean_db, "detections") == 1
