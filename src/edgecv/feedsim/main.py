"""feed-sim: replay images as a paced, prevalence-controlled frame feed."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import psycopg
import redis

from edgecv.bus.producer import FrameProducer
from edgecv.config import Settings
from edgecv.contracts.frame import FrameEnvelope
from edgecv.db.migrate import apply_migrations
from edgecv.db.repository import Repository
from edgecv.feedsim.prevalence import PrevalenceSampler, pace_deadlines


@dataclass(frozen=True, slots=True)
class FeedStats:
    run_id: str
    offered: int
    published: int
    dropped: int


def load_pools(manifest_path: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads(Path(manifest_path).read_text())
    return ([entry["path"] for entry in manifest["clean"]],
            [entry["path"] for entry in manifest["defect"]])


def run_feed(client: redis.Redis, *, manifest: Path, run_id: str | None,
             n_frames: int, fps: float, prevalence: float, maxlen: int,
             seed: int, stream: str, transport: str = "reference") -> FeedStats:
    run_id = run_id or str(uuid.uuid4())
    clean, defect = load_pools(manifest)
    sampler = PrevalenceSampler(clean, defect, prevalence=prevalence,
                                rng=random.Random(seed))
    producer = FrameProducer(client, stream=stream, maxlen=maxlen)

    published = 0
    for seq, deadline in enumerate(
        pace_deadlines(start=time.monotonic(), fps=fps, n=n_frames)
    ):
        path, _is_defect = sampler.next()
        data = Path(path).read_bytes()
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]

        envelope = FrameEnvelope(
            run_id=run_id, seq=seq,
            captured_at=datetime.now(timezone.utc),
            width=int(width), height=int(height),
            source_ref=path,
            sha256=hashlib.sha256(data).hexdigest(),
            transport=transport,
            payload=data if transport == "inline" else None,
            path=path if transport == "reference" else None,
        )
        if producer.publish(envelope) is not None:
            published += 1

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    return FeedStats(run_id=run_id, offered=producer.offered,
                     published=published, dropped=producer.dropped)


def main() -> None:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description="Simulated road-survey frame feed")
    ap.add_argument("--manifest", type=Path,
                    default=Path("tests/fixtures/generated/manifest.json"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--prevalence", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--authority-id", default="demo-council")
    ap.add_argument("--transport", choices=["inline", "reference"],
                    default="reference")
    args = ap.parse_args()

    run_id = args.run_id or str(uuid.uuid4())
    client = redis.from_url(settings.redis_url, decode_responses=False)

    # feedsim is the only component that knows run_id, target_fps, prevalence,
    # transport and the source, so it registers the run -- nothing else in the
    # running system writes survey_runs. This puts a Postgres write on the
    # "edge" side of the frames-stream seam described in README.md; see the
    # note there for why that's an accepted trade for this milestone.
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        apply_migrations(conn)
        repo = Repository(conn)
        repo.upsert_run(run_id=run_id, authority_id=args.authority_id,
                        started_at=datetime.now(timezone.utc),
                        source_kind="synthetic", source_ref=str(args.manifest),
                        target_fps=args.fps, prevalence=args.prevalence,
                        transport=args.transport)

        stats = run_feed(client, manifest=args.manifest, run_id=run_id,
                         n_frames=args.frames, fps=args.fps,
                         prevalence=args.prevalence, maxlen=settings.frames_maxlen,
                         seed=args.seed, stream=settings.frames_stream,
                         transport=args.transport)

        repo.finish_run(stats.run_id, datetime.now(timezone.utc),
                       config={"frames_offered": stats.offered,
                               "frames_dropped": stats.dropped})

    print(f"run_id={stats.run_id} offered={stats.offered} "
          f"published={stats.published} dropped={stats.dropped}")


if __name__ == "__main__":
    main()
