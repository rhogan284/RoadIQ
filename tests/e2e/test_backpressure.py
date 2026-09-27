"""Overload the bounded bus. Assert the dropped frames are counted, not hidden.

The frames stream is bounded (FRAMES_MAXLEN): when the workers fall behind, feed-sim's
producer refuses frames and counts them. Those frames are road nobody assessed. The
read API must report them as dropped frames and as metres not assessed (success
criterion 2) -- and still read 100 % "accounted for", because every frame's fate is
known.

The workers here pause while feed-sim publishes, then catch up once, mid-feed. That
leaves the drops in the MIDDLE of the run -- a gap between two landed frames, which is
what coverage measures in metres. Drops only at the end would be a shorter run, not a
gap.
"""
from __future__ import annotations

import datetime as dt

from edgecv.bench.collect import run_coverage
from edgecv.blobstore.store import BlobStore
from edgecv.bus.consumer import FrameConsumer
from edgecv.db.repository import Repository
from edgecv.feedsim.main import run_feed
from edgecv.feedsim.route import RouteTrack
from edgecv.roads import DEFAULT_NETWORK
from edgecv.worker.main import process_one
from edgecv.writer.main import drain_once

N_FRAMES = 40
MAXLEN = 10
CATCH_UP_AT = 25          # the workers empty the bus just before frame 25 is offered
FPS = 8.0


def test_dropped_frames_are_reported_as_road_not_assessed(rds, clean_db, rdd_subset,
                                                          detector, tmp_path, api):
    manifest = rdd_subset(20, 20)
    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")
    run_id = "00000000-0000-4000-8000-0000000000bb"
    repo.upsert_run(run_id=run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="dataset-replay", source_ref=str(manifest),
                    target_fps=FPS, prevalence=None, transport="reference")

    consumer = FrameConsumer(rds, stream="frames", group="workers", consumer="w1",
                             block_ms=100)
    consumer.ensure_group()
    taken = []

    def take_all() -> None:
        """Read and ack everything on the bus (ack XDELs, freeing the space)."""
        while batch := consumer.read(count=MAXLEN):
            for entry_id, envelope in batch:
                taken.append(envelope)
                consumer.ack(entry_id)

    offered = 0

    def control() -> str:
        # Called before every frame: the one hook inside the feed loop.
        nonlocal offered
        if offered == CATCH_UP_AT:
            take_all()
        offered += 1
        return "run"

    track = RouteTrack.from_network(DEFAULT_NETWORK, n_frames=N_FRAMES, seed=1,
                                    speed_mps=13.89, fps=FPS)
    stats = run_feed(rds, manifest=manifest, run_id=run_id, fps=1000, maxlen=MAXLEN,
                     seed=1, stream="frames", track=track, control=control)
    take_all()

    # seq 0-9 fill the bus, 10-24 are refused, 25-34 fill it again, 35-39 are refused.
    assert (stats.offered, stats.published, stats.dropped) == (N_FRAMES, 20, 20)
    assert sorted(e.seq for e in taken) == list(range(10)) + list(range(25, 35))
    repo.finish_run(run_id, dt.datetime.now(dt.timezone.utc),
                    config={"frames_offered": stats.offered,
                            "frames_dropped": stats.dropped,
                            "raw_bytes_offered": stats.raw_bytes})

    for envelope in taken:
        rds.xadd("results", {"json": process_one(
            envelope, blobstore=blobstore, worker_id="w1", detector=detector).to_json()})
    while drain_once(rds, repo, stream="results", group="writers", consumer="wr1",
                     batch=50).frames:
        pass

    # The middle gap is measured in metres of road, between the frames either side.
    coverage = run_coverage(clean_db, run_id)
    assert coverage.frames == 20
    gap = next(g for g in coverage.gaps if g.after_seq == 9)
    assert (gap.before_seq, gap.missing_frames) == (25, 15)
    assert gap.metres > 0
    assert coverage.gap_m >= gap.metres

    summary = api.get(f"/api/runs/{run_id}/summary").json()
    assert summary["frames_offered"] == N_FRAMES
    assert summary["frames_ingested"] == 20
    assert summary["frames_dropped"] == 20
    assert summary["frames_in_flight"] == 0
    assert summary["frames_accounted_pct"] == 100.0
    assert summary["gap_m"] == round(coverage.gap_m, 1)
