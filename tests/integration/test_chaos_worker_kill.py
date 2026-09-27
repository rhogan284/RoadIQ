"""Kill a worker mid-run. Assert nothing is lost and nothing is duplicated.

A worker that reads frames and dies without acking leaves them PENDING. XAUTOCLAIM lets
a survivor take them over. Reprocessing is therefore expected -- and harmless, because
inferences carries UNIQUE(run_id, seq, detector_id).

"Killing a worker" is simulated here at the consumer-group level rather than by
spawning and signal-killing a real `edgecv.worker.main` process: a `doomed` consumer
reads frames and never acks them, which is exactly the state Redis is left in whether
the process died from SIGKILL, a segfault, or simply hung -- the consumer group cannot
tell the difference, and neither does the recovery path. This is what
FrameConsumer.reclaim() is built to detect and correct.

Read the "Critical knowledge" section of the dispatch brief before changing this file:
reclaim() makes a single XAUTOCLAIM call and does not loop its own cursor, and the
worker only calls it when a normal read() comes back empty. Both tests below mirror
that gating explicitly (drain reads to empty, THEN loop reclaim until it stops
making progress -- `progressed`, not "returned deliveries", because a page that only
purges destroyed entries recovers nothing yet has still advanced) rather than
polling reclaim() on a timer, and both use a finite feed published in full before any
consumer starts, so the "read comes back empty" condition is reached quickly and
deterministically.

These tests own Redis outright and require the docker-compose app services
(worker, writer, feedsim, dashboard) to be STOPPED before running -- redis and
postgres should stay up. Two independent reasons, either one sufficient on its
own: (1) the `rds` fixture flushes the whole Redis DB on setup and teardown, so
running these tests destroys a live stack's stream and consumer group; (2) this
file reads from `stream="frames"`, `group="workers"` -- the same stream and
consumer group the real containerised workers consume from -- so a live stack
races the `doomed`/`survivor` consumers here for the same frames and breaks the
exact counts this file asserts on (a real containerised worker can "steal" a
frame the test expects `survivor` to reclaim). `make itest` stops the app
services for you; if you invoke pytest directly instead, stop them yourself
first (`docker compose stop worker writer feedsim dashboard`) or a confusing
count mismatch here is very likely a stale live stack, not a real bug.
"""
from __future__ import annotations

import pytest

from edgecv.blobstore.store import BlobStore
from edgecv.bus.consumer import FrameConsumer
from edgecv.db.repository import Repository
from edgecv.feedsim.main import run_feed
from edgecv.worker.main import process_one
from edgecv.writer.main import drain_once

pytestmark = pytest.mark.integration

N_FRAMES = 60
KILL_AFTER = 20


@pytest.fixture
def fixtures(tmp_path_factory):
    from scripts.make_fixtures import generate
    out = tmp_path_factory.mktemp("chaos")
    generate(out, n_clean=20, n_defect=10, size=64, seed=77)
    return out


def count(conn, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_no_frames_lost_when_a_worker_dies(rds, clean_db, fixtures, tmp_path):
    import datetime as dt

    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")

    stats = run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
                     n_frames=N_FRAMES, fps=1000, prevalence=0.2,
                     maxlen=10_000, seed=3, stream="frames")
    # Repository.upsert_run takes `authority_id`, not `line_id` -- see
    # dispatch-9-report.md for the full list of library-text/code mismatches.
    repo.upsert_run(run_id=stats.run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="synthetic", source_ref="generated",
                    target_fps=1000, prevalence=0.2, transport="reference")

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
    assert rds.xpending("frames", "workers")["pending"] == read_before_death

    # --- survivor drains the rest, then reclaims the dead worker's frames ---
    survivor = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="survivor", block_ms=100)
    survivor.ensure_group()

    def handle(batch) -> int:
        for entry_id, envelope in batch:
            rds.xadd("results", {"json": process_one(
                envelope, blobstore=blobstore, worker_id="survivor").to_json()})
            survivor.ack(entry_id)
        return len(batch)

    handled = 0
    while True:
        batch = survivor.read(count=10)
        if not batch:
            break
        handled += handle(batch)

    reclaimed_total = 0
    while True:
        outcome = survivor.reclaim(min_idle_ms=0, count=10)
        if not outcome.progressed:
            break
        reclaimed_total += handle(outcome.deliveries)

    assert reclaimed_total == read_before_death      # every orphan recovered
    assert handled + reclaimed_total == N_FRAMES
    # Nothing was destroyed in this scenario, only orphaned. The worker died;
    # the stream did not lose an entry. Asserted so the two failure modes stay
    # distinguishable -- the whole point of reading XAUTOCLAIM's purged-id list.
    assert survivor.lost_in_flight == 0

    while drain_once(rds, repo, stream="results", group="writers",
                     consumer="wr1", batch=100).frames:
        pass

    # No loss.
    assert count(clean_db, "SELECT count(*) FROM frames") == N_FRAMES
    assert count(clean_db, "SELECT count(*) FROM inferences") == N_FRAMES
    # No duplicates -- guaranteed by UNIQUE(run_id, seq, detector_id).
    assert count(clean_db,
                 "SELECT count(*) FROM (SELECT run_id, seq, detector_id "
                 "FROM inferences GROUP BY 1,2,3 HAVING count(*) > 1) dupes") == 0
    assert rds.xpending("frames", "workers")["pending"] == 0


def test_reprocessing_a_frame_twice_adds_no_extra_detections(rds, clean_db,
                                                             fixtures, tmp_path):
    import datetime as dt

    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")

    stats = run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
                     n_frames=10, fps=1000, prevalence=1.0, maxlen=1000,
                     seed=9, stream="frames")
    repo.upsert_run(run_id=stats.run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="synthetic", source_ref="generated",
                    target_fps=1000, prevalence=1.0, transport="reference")

    consumer = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="w1", block_ms=100)
    consumer.ensure_group()
    envelopes = []
    while len(envelopes) < 10:
        batch = consumer.read(count=10)
        if not batch:
            break
        for entry_id, envelope in batch:
            envelopes.append(envelope)
            consumer.ack(entry_id)

    results = [process_one(e, blobstore=blobstore, worker_id="w1")
               for e in envelopes]
    repo.write_results(results)
    first = count(clean_db, "SELECT count(*) FROM detections")

    repo.write_results(results)          # exact same work, delivered again
    assert count(clean_db, "SELECT count(*) FROM detections") == first
    assert count(clean_db, "SELECT count(*) FROM frames") == 10


def test_drain_reclaimed_does_not_strand_orphans_behind_destroyed_entries(
        rds, fixtures, tmp_path, monkeypatch):
    """A page of destroyed entries must not end the drain.

    XAUTOCLAIM pages through the PEL RECLAIM_COUNT entries at a time. If the
    first page is entirely entries whose stream entry is gone, it recovers no
    work -- and a loop that stops there abandons every live orphan queued behind
    it until the next empty read, RECLAIM_IDLE_MS later at best.

    Here the first 10 of 25 pending frames are destroyed, so the first
    XAUTOCLAIM page is all tombstones and the remaining 15 are recoverable.
    Asserting 15 is asserting that the drain kept going.
    """
    from edgecv.worker import main as worker_main

    monkeypatch.setattr(worker_main, "RECLAIM_IDLE_MS", 0)
    blobstore = BlobStore(root=tmp_path / "blobs")

    n_total, n_destroyed = 25, worker_main.RECLAIM_COUNT
    run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
             n_frames=n_total, fps=1000, prevalence=0.2, maxlen=10_000,
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

    # Destroy the oldest 10 out from under the PEL. This is what Redis's own
    # MAXLEN trimming does under backpressure: it evicts the OLDEST first.
    rds.xdel("frames", *entry_ids[:n_destroyed])
    assert rds.xpending("frames", "workers")["pending"] == n_total

    survivor = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="survivor", block_ms=100)
    survivor.ensure_group()
    processed = worker_main._drain_reclaimed(
        survivor, blobstore=blobstore, worker_id="survivor", client=rds,
        results_stream="results")

    assert processed == n_total - n_destroyed    # the 15 behind the tombstones
    assert survivor.lost_in_flight == n_destroyed
    assert int(rds.get(worker_main.LOST_IN_FLIGHT_KEY)) == n_destroyed
    assert rds.xpending("frames", "workers")["pending"] == 0
