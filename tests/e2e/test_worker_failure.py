"""Kill a worker mid-run. Assert nothing is lost, nothing is duplicated, and the bus
health readings the dashboard shows tell the truth about it.

A worker that reads frames and dies without acking leaves them PENDING. XAUTOCLAIM lets
a survivor take them over. Reprocessing is therefore expected -- and harmless, because
inferences carries UNIQUE(run_id, seq, detector_id).

"Killing a worker" is simulated at the consumer-group level rather than by
signal-killing a real `edgecv.worker.main` process: a `doomed` consumer reads frames and
never acks them, which is exactly the state Redis is left in whether the process died
from SIGKILL, a segfault, or simply hung -- the consumer group cannot tell the
difference, and neither does the recovery path.

reclaim() makes a single XAUTOCLAIM call and does not loop its own cursor, and the
worker only calls it when a normal read() comes back empty. The tests mirror that
gating (drain reads to empty, THEN loop reclaim until it stops making progress --
`progressed`, not "returned deliveries", because a page that only purges destroyed
entries recovers nothing yet has still advanced), and use a finite feed published in
full before any consumer starts.

These tests read from `stream="frames"`, `group="workers"` -- the same stream and group
the containerised workers consume from -- so the app services must be STOPPED (`make
test` does it). A confusing count mismatch here is very likely a live stack racing the
test's consumers, not a real bug.
"""
from __future__ import annotations

import datetime as dt

from edgecv.blobstore.store import BlobStore
from edgecv.bus.consumer import FrameConsumer
from edgecv.bus.observe import group_health, stuck_entries
from edgecv.db.repository import Repository
from edgecv.feedsim.main import run_feed
from edgecv.worker.main import process_one
from edgecv.writer.main import drain_once

KILL_AFTER = 10


def count(conn, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def _register(repo: Repository, run_id: str, manifest) -> None:
    repo.upsert_run(run_id=run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="dataset-replay", source_ref=str(manifest),
                    target_fps=1000, prevalence=None, transport="reference")


def test_no_frames_lost_when_a_worker_dies(rds, clean_db, rdd_subset, detector, tmp_path):
    manifest = rdd_subset(15, 15)
    n_frames = 30
    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")

    stats = run_feed(rds, manifest=manifest, run_id=None, fps=1000,
                     maxlen=10_000, seed=3, stream="frames")
    assert stats.published == n_frames
    _register(repo, stats.run_id, manifest)

    # --- doomed worker: reads KILL_AFTER frames, acks NONE, then "dies" ---
    doomed = FrameConsumer(rds, stream="frames", group="workers",
                           consumer="doomed", block_ms=100)
    doomed.ensure_group()
    read_before_death = 0
    while read_before_death < KILL_AFTER:
        batch = doomed.read(count=5)
        if not batch:
            break
        read_before_death += len(batch)
    assert read_before_death >= KILL_AFTER

    # What the observability panel sees: the dead worker's frames are stuck work.
    health = group_health(rds, stream_name="frames", group="workers")
    assert health.pending == read_before_death
    assert health.lag == n_frames - read_before_death
    stuck = stuck_entries(rds, stream_name="frames", group="workers", min_idle_ms=0,
                          count=100)
    assert {s.consumer for s in stuck} == {"doomed"}
    assert len(stuck) == read_before_death

    # --- survivor drains the rest, then reclaims the dead worker's frames ---
    survivor = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="survivor", block_ms=100)
    survivor.ensure_group()

    def handle(batch) -> int:
        for entry_id, envelope in batch:
            rds.xadd("results", {"json": process_one(
                envelope, blobstore=blobstore, worker_id="survivor",
                detector=detector).to_json()})
            survivor.ack(entry_id)
        return len(batch)

    handled = 0
    while batch := survivor.read(count=10):
        handled += handle(batch)

    reclaimed_total = 0
    while True:
        outcome = survivor.reclaim(min_idle_ms=0, count=10)
        if not outcome.progressed:
            break
        reclaimed_total += handle(outcome.deliveries)

    assert reclaimed_total == read_before_death      # every orphan recovered
    assert handled + reclaimed_total == n_frames
    # Nothing was destroyed here, only orphaned. Asserted so the two failure modes
    # stay distinguishable -- the whole point of reading XAUTOCLAIM's purged-id list.
    assert survivor.lost_in_flight == 0
    health = group_health(rds, stream_name="frames", group="workers")
    assert (health.pending, health.lag) == (0, 0)

    while drain_once(rds, repo, stream="results", group="writers",
                     consumer="wr1", batch=100).frames:
        pass

    # No loss.
    assert count(clean_db, "SELECT count(*) FROM frames") == n_frames
    assert count(clean_db, "SELECT count(*) FROM inferences") == n_frames
    # No duplicates -- guaranteed by UNIQUE(run_id, seq, detector_id).
    assert count(clean_db,
                 "SELECT count(*) FROM (SELECT run_id, seq, detector_id "
                 "FROM inferences GROUP BY 1,2,3 HAVING count(*) > 1) dupes") == 0


def test_redelivered_results_add_no_extra_rows(rds, clean_db, rdd_subset, detector,
                                               tmp_path):
    manifest = rdd_subset(0, 10)
    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")

    stats = run_feed(rds, manifest=manifest, run_id=None, fps=1000, maxlen=1000,
                     seed=9, stream="frames")
    _register(repo, stats.run_id, manifest)

    consumer = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="w1", block_ms=100)
    consumer.ensure_group()
    envelopes = []
    while batch := consumer.read(count=10):
        for entry_id, envelope in batch:
            envelopes.append(envelope)
            consumer.ack(entry_id)
    assert len(envelopes) == 10

    results = [process_one(e, blobstore=blobstore, worker_id="w1", detector=detector)
               for e in envelopes]
    repo.write_results(results)
    first = count(clean_db, "SELECT count(*) FROM detections")
    assert first > 0, "YOLOv12s found nothing on 10 labelled damage images"

    repo.write_results(results)          # exact same work, delivered again
    assert count(clean_db, "SELECT count(*) FROM detections") == first
    assert count(clean_db, "SELECT count(*) FROM frames") == 10


def test_drain_reclaimed_does_not_strand_orphans_behind_destroyed_entries(
        rds, rdd_subset, detector, tmp_path, monkeypatch):
    """A page of destroyed entries must not end the drain.

    XAUTOCLAIM pages through the PEL RECLAIM_COUNT entries at a time. If the first page
    is entirely entries whose stream entry is gone, it recovers no work -- and a loop
    that stops there abandons every live orphan queued behind it until the next empty
    read, RECLAIM_IDLE_MS later at best.

    Here the first 10 of 25 pending frames are destroyed, so the first XAUTOCLAIM page
    is all tombstones and the remaining 15 are recoverable. Asserting 15 is asserting
    that the drain kept going.
    """
    from edgecv.worker import main as worker_main

    monkeypatch.setattr(worker_main, "RECLAIM_IDLE_MS", 0)
    blobstore = BlobStore(root=tmp_path / "blobs")

    n_total, n_destroyed = 25, worker_main.RECLAIM_COUNT
    run_feed(rds, manifest=rdd_subset(20, 5), run_id=None, fps=1000, maxlen=10_000,
             seed=5, stream="frames")

    doomed = FrameConsumer(rds, stream="frames", group="workers",
                           consumer="doomed", block_ms=100)
    doomed.ensure_group()
    entry_ids = []
    while len(entry_ids) < n_total:
        batch = doomed.read(count=n_total)
        if not batch:
            break
        entry_ids += [entry_id for entry_id, _e in batch]
    assert len(entry_ids) == n_total

    # Destroy the oldest 10 out from under the PEL. This is what Redis's own MAXLEN
    # trimming does under backpressure: it evicts the OLDEST first.
    rds.xdel("frames", *entry_ids[:n_destroyed])
    assert rds.xpending("frames", "workers")["pending"] == n_total

    survivor = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="survivor", block_ms=100)
    survivor.ensure_group()
    processed = worker_main._drain_reclaimed(
        survivor, blobstore=blobstore, worker_id="survivor", client=rds,
        results_stream="results", detector=detector)

    assert processed == n_total - n_destroyed    # the 15 behind the tombstones
    assert survivor.lost_in_flight == n_destroyed
    assert int(rds.get(worker_main.LOST_IN_FLIGHT_KEY)) == n_destroyed
    assert rds.xpending("frames", "workers")["pending"] == 0
