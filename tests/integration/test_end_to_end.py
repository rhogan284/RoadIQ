"""Feed -> bus -> worker -> writer -> Postgres, with exact assertions.

Uses generated fixtures whose ground truth is known, so counts are exact rather than
approximate.

The worker/writer functions are called in-process here rather than through the
docker-compose containers: `process_one` and `drain_once` are plain functions, and
the blob store is a `BlobStore(root=tmp_path / "blobs")` created fresh in each test.
That sidesteps the host-vs-container blob-path reachability question raised in the
dispatch brief entirely -- there is no container writing crops to a volume that the
test then has to reach from the host, because nothing here runs in a container. If a
future test wants to assert on files written by the containerised `worker` service,
it would need to read them from the `blobs` named volume (e.g. via
`docker compose cp` or a bind mount), not from a host path built from `Settings`.
"""
from __future__ import annotations

import datetime as dt
import random

import pytest

from edgecv.blobstore.store import BlobStore
from edgecv.bus.consumer import FrameConsumer
from edgecv.db.repository import Repository
from edgecv.feedsim.main import load_pools, run_feed
from edgecv.feedsim.prevalence import PrevalenceSampler
from edgecv.worker.main import process_one
from edgecv.writer.main import drain_once

pytestmark = pytest.mark.integration

N_FRAMES = 200
PREVALENCE = 0.10
SEED = 4242


@pytest.fixture
def fixtures(tmp_path_factory):
    from scripts.make_fixtures import generate
    out = tmp_path_factory.mktemp("e2e")
    generate(out, n_clean=40, n_defect=20, size=128, seed=SEED)
    return out


def expected_defect_count(fixtures) -> int:
    """Recompute the seeded sampler's draws -- the feed is deterministic."""
    clean, defect = load_pools(fixtures / "manifest.json")
    sampler = PrevalenceSampler(clean, defect, prevalence=PREVALENCE,
                                rng=random.Random(SEED))
    return sum(1 for _ in range(N_FRAMES) if sampler.next()[1])


def count(conn, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_pipeline_end_to_end(rds, clean_db, fixtures, tmp_path):
    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")

    # 1. feed
    stats = run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
                     n_frames=N_FRAMES, fps=1000, prevalence=PREVALENCE,
                     maxlen=10_000, seed=SEED, stream="frames")
    assert stats.offered == N_FRAMES
    assert stats.dropped == 0
    # Repository.upsert_run takes `authority_id`, not `line_id` -- a run is one
    # drive for one road authority, not a shift on a factory line.
    repo.upsert_run(run_id=stats.run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="synthetic", source_ref="generated",
                    target_fps=1000, prevalence=PREVALENCE, transport="reference")

    # 2. infer
    consumer = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="w1", block_ms=100)
    consumer.ensure_group()
    processed = 0
    while processed < N_FRAMES:
        batch = consumer.read(count=50)
        if not batch:
            break
        for entry_id, envelope in batch:
            outcome = process_one(envelope, blobstore=blobstore, worker_id="w1")
            rds.xadd("results", {"json": outcome.to_json()})
            consumer.ack(entry_id)
            processed += 1
    assert processed == N_FRAMES

    # 3. write
    written = 0
    while written < N_FRAMES:
        drained = drain_once(rds, repo, stream="results", group="writers",
                             consumer="wr1", batch=50)
        if drained.frames == 0:
            break
        written += drained.frames
    assert written == N_FRAMES

    # 4. assert
    assert count(clean_db, "SELECT count(*) FROM frames") == N_FRAMES
    assert count(clean_db, "SELECT count(*) FROM inferences") == N_FRAMES
    assert count(clean_db,
                 "SELECT count(*) FROM inferences WHERE status='ok'") == N_FRAMES

    n_defect_frames = expected_defect_count(fixtures)
    flagged = count(clean_db,
                    "SELECT count(DISTINCT i.inference_id) FROM inferences i "
                    "JOIN detections d ON d.inference_id = i.inference_id")
    assert flagged == n_defect_frames

    # Every detection has a stored crop.
    assert count(clean_db,
                 "SELECT count(*) FROM detections WHERE snippet_id IS NULL") == 0

    # Dedup: distinct crops cannot exceed the number of distinct source images.
    distinct_crops = count(clean_db,
                           "SELECT count(*) FROM snippets WHERE kind='crop'")
    assert 0 < distinct_crops <= 20


def test_rollup_reports_the_configured_prevalence(rds, clean_db, fixtures, tmp_path):
    """The dashboard's coverage panel is fed by `run_coverage_1min`, a plain VIEW
    computed on read (see the writer module docstring and db/migrations/001_initial.sql)
    -- there is no materialised rollup and therefore no refresh step. The library
    text this test was transcribed from called a `Repository.refresh_rollup()` and
    an `edgecv.dashboard.app.defect_rate_series()` that do not exist anywhere in this
    codebase; the real, tested query function is `coverage_series`.
    """
    from edgecv.dashboard.app import coverage_series

    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")

    stats = run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
                     n_frames=N_FRAMES, fps=1000, prevalence=PREVALENCE,
                     maxlen=10_000, seed=SEED, stream="frames")
    repo.upsert_run(run_id=stats.run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="synthetic", source_ref="generated",
                    target_fps=1000, prevalence=PREVALENCE, transport="reference")

    consumer = FrameConsumer(rds, stream="frames", group="workers",
                             consumer="w1", block_ms=100)
    consumer.ensure_group()
    for _ in range(N_FRAMES):
        batch = consumer.read(count=50)
        if not batch:
            break
        for entry_id, envelope in batch:
            rds.xadd("results", {"json": process_one(
                envelope, blobstore=blobstore, worker_id="w1").to_json()})
            consumer.ack(entry_id)
    while drain_once(rds, repo, stream="results", group="writers",
                     consumer="wr1", batch=100).frames:
        pass

    rows = coverage_series(clean_db, stats.run_id)
    total = sum(r["frames_ingested"] for r in rows)
    flagged = sum(r["frames_flagged"] for r in rows)
    assert total == N_FRAMES
    # Sampling variance at n=200: the observed rate should be near the configured one.
    assert abs(flagged / total - PREVALENCE) < 0.05
