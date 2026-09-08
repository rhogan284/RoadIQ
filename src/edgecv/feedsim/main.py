"""feed-sim: replay images as a paced, prevalence-controlled frame feed.

Frames carry a synthetic GPS fix by default (`--no-gps` to opt out). Before
this, every replayed frame reached Postgres with `lat IS NULL`, which left the
bytes-per-kilometre milestone with no denominator -- there was no distance to
divide by. The track is generated here rather than baked into the fixture
manifest because position belongs to the drive, not to the picture: the same
pool of images replayed at a different speed must cover a different distance.
"""
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
from edgecv.feedsim.gpstrack import DEFAULT_ACCURACY_M, Fix, SyntheticTrack
from edgecv.feedsim.prevalence import PrevalenceSampler, pace_deadlines

#: Sydney CBD, George and Market. An arbitrary but plausible survey origin.
DEFAULT_START = (-33.8688, 151.2093)
#: 50 km/h in m/s -- an urban survey speed.
DEFAULT_SPEED_MPS = 13.89
DEFAULT_BEARING_DEG = 90.0


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


def build_envelope(*, run_id: str, seq: int, path: str, data: bytes,
                   width: int, height: int, transport: str,
                   fix: Fix | None = None) -> FrameEnvelope:
    """One frame envelope, with the position fields filled in when a fix exists.

    Optional fields stay None when there is no fix rather than being sent as
    zeros: contract 1 omits absent fields from the wire entirely, and
    `run_coverage_1min` counts `frames_without_fix` off exactly that null.
    """
    return FrameEnvelope(
        run_id=run_id, seq=seq,
        captured_at=datetime.now(timezone.utc),
        width=int(width), height=int(height),
        source_ref=path,
        sha256=hashlib.sha256(data).hexdigest(),
        transport=transport,
        payload=data if transport == "inline" else None,
        path=path if transport == "reference" else None,
        lat=fix.lat if fix else None,
        lon=fix.lon if fix else None,
        heading_deg=fix.heading_deg if fix else None,
        speed_mps=fix.speed_mps if fix else None,
        gps_accuracy_m=fix.gps_accuracy_m if fix else None,
    )


def run_feed(client: redis.Redis, *, manifest: Path, run_id: str | None,
             n_frames: int, fps: float, prevalence: float, maxlen: int,
             seed: int, stream: str, transport: str = "reference",
             track: SyntheticTrack | None = None) -> FeedStats:
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

        envelope = build_envelope(
            run_id=run_id, seq=seq, path=path, data=data,
            width=width, height=height, transport=transport,
            fix=track.fix_for(seq) if track else None,
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
    ap.add_argument("--no-gps", dest="gps", action="store_false", default=True,
                    help="replay without a position fix (the pre-milestone-2 "
                         "behaviour; leaves lat/lon NULL and makes "
                         "bytes-per-km unmeasurable)")
    ap.add_argument("--start-lat", type=float, default=DEFAULT_START[0])
    ap.add_argument("--start-lon", type=float, default=DEFAULT_START[1])
    ap.add_argument("--bearing", type=float, default=DEFAULT_BEARING_DEG)
    ap.add_argument("--speed-mps", type=float, default=DEFAULT_SPEED_MPS)
    ap.add_argument("--gps-accuracy-m", type=float, default=DEFAULT_ACCURACY_M)
    args = ap.parse_args()

    track = SyntheticTrack(
        start=(args.start_lat, args.start_lon), bearing_deg=args.bearing,
        speed_mps=args.speed_mps, fps=args.fps, accuracy_m=args.gps_accuracy_m,
    ) if args.gps else None

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
                         transport=args.transport, track=track)

        config: dict = {"frames_offered": stats.offered,
                        "frames_dropped": stats.dropped}
        if track is not None:
            # Written so the denominator of the bytes-per-km figure can be
            # audited and the run reproduced, not just trusted.
            config["gps_track"] = {
                "kind": "synthetic-rhumb", "start_lat": args.start_lat,
                "start_lon": args.start_lon, "bearing_deg": args.bearing,
                "speed_mps": args.speed_mps, "fps": args.fps,
                "accuracy_m": args.gps_accuracy_m,
                "expected_distance_m": round(track.distance_m(stats.published), 3),
            }
        repo.finish_run(stats.run_id, datetime.now(timezone.utc), config=config)

    gps = f" distance_m={track.distance_m(stats.published):.1f}" if track else \
          " gps=off"
    print(f"run_id={stats.run_id} offered={stats.offered} "
          f"published={stats.published} dropped={stats.dropped}{gps}")


if __name__ == "__main__":
    main()
