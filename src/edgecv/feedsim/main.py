"""feed-sim: replay a dataset as a paced frame feed along the road network.

Every manifest image is sent exactly once, in a seeded shuffle, with a GPS fix from a
drive over the street network. The track is generated here rather than baked into the
manifest because position belongs to the drive, not to the picture: the same images
replayed at a different speed must cover a different distance.
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
from edgecv.feedsim.route import DEFAULT_ACCURACY_M, Fix, RouteTrack
from edgecv.roads import DEFAULT_NETWORK

#: 50 km/h in m/s -- an urban survey speed.
DEFAULT_SPEED_MPS = 13.89
DEFAULT_MANIFEST = Path("data/rdd2022/manifest.json")


@dataclass(frozen=True, slots=True)
class FeedStats:
    run_id: str
    offered: int
    published: int
    dropped: int
    #: Sum of the replayed files' sizes — what a naive pipeline would have uploaded.
    raw_bytes: int = 0
    cancelled: bool = False


#: Operator control for a live run, set by the read API: "pause", "run" or "cancel".
#: A Redis key rather than a signal because the dashboard and feed-sim share nothing else.
def control_key(run_id: str) -> str:
    return f"control:{run_id}"


#: What feed-sim is doing now, for the dashboard: "running", "paused", "cancelled", "done".
def state_key(run_id: str) -> str:
    return f"control:{run_id}:state"


def redis_control(client: redis.Redis, run_id: str):
    """A control callable for run_feed backed by the run's Redis key."""
    def read() -> str:
        raw = client.get(control_key(run_id))
        return raw.decode() if raw else "run"
    return read


def load_pools(manifest_path: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads(Path(manifest_path).read_text())
    return ([entry["path"] for entry in manifest["clean"]],
            [entry["path"] for entry in manifest["defect"]])


def is_ordered(manifest_path: Path) -> bool:
    """A video manifest (scripts/extract_video.py) is one drive: replay it in capture
    order. Shuffling is only right for a dataset of unrelated photos."""
    return bool(json.loads(Path(manifest_path).read_text()).get("ordered", False))


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
             fps: float, maxlen: int, seed: int, stream: str,
             transport: str = "reference", track: RouteTrack | None = None,
             control=None, max_frames: int | None = None) -> FeedStats:
    """Replay every manifest image exactly once in a seeded shuffle.

    `control`, if given, is called before every frame and returns "run", "pause" or
    "cancel". Pause holds the vehicle where it is; on resume the pacing restarts from
    now, so there is no burst of catch-up frames (which would read as a speed spike and
    could overflow the bounded stream). Cancel stops the feed; the run keeps what it sent."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    run_id = run_id or str(uuid.uuid4())
    clean, defect = load_pools(manifest)
    paths = clean + defect
    if not is_ordered(manifest):
        random.Random(seed).shuffle(paths)
    # A short demo run takes the first N of the shuffle — a random sample of the
    # split, not its first N files, which would all be one country.
    if max_frames:
        paths = paths[:max_frames]
    producer = FrameProducer(client, stream=stream, maxlen=maxlen)

    def set_state(state: str) -> None:
        if control is not None:
            client.set(state_key(run_id), state)

    published = raw_bytes = 0
    cancelled = False
    interval = 1.0 / fps
    deadline = time.monotonic()
    set_state("running")
    for seq, path in enumerate(paths):
        if control is not None:
            action = control()
            if action == "pause":
                set_state("paused")
                while (action := control()) == "pause":
                    time.sleep(0.2)
                deadline = time.monotonic()        # restart pacing: no catch-up burst
                set_state("running")
            if action == "cancel":
                cancelled = True
                set_state("cancelled")
                break
        data = Path(path).read_bytes()
        raw_bytes += len(data)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]

        envelope = build_envelope(
            run_id=run_id, seq=seq, path=path, data=data,
            width=width, height=height, transport=transport,
            fix=track.fix_for(seq) if track else None,
        )
        if producer.publish(envelope) is not None:
            published += 1

        deadline += interval
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    if not cancelled:
        set_state("done")
    return FeedStats(run_id=run_id, offered=producer.offered,
                     published=published, dropped=producer.dropped,
                     raw_bytes=raw_bytes, cancelled=cancelled)


def main() -> None:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description="Simulated road-survey frame feed")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--authority-id", default="demo-council")
    ap.add_argument("--transport", choices=["inline", "reference"],
                    default="reference")
    ap.add_argument("--route", type=Path, default=DEFAULT_NETWORK,
                    help="road-network GeoJSON to drive along")
    ap.add_argument("--route-mode", choices=["random", "loop", "cover"], default="random",
                    help="random = a new randomised drive over the council streets each "
                         "run, sized to the frame count; loop = the fixed main-road loop; "
                         "cover = every street, with jumps")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="replay only the first N of the shuffle")
    ap.add_argument("--route-seed", type=int, default=None,
                    help="fix the random route (default: a new one every run)")
    ap.add_argument("--source-kind", choices=["dataset-replay", "drive"],
                    default="dataset-replay")
    ap.add_argument("--speed-mps", type=float, default=DEFAULT_SPEED_MPS)
    ap.add_argument("--gps-accuracy-m", type=float, default=DEFAULT_ACCURACY_M)
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest {args.manifest} not found — run `make dataset` first")
    clean, defect = load_pools(args.manifest)
    n_planned = len(clean) + len(defect)
    if args.max_frames:
        n_planned = min(n_planned, args.max_frames)
    route_seed = (args.route_seed if args.route_seed is not None
                  else random.SystemRandom().randrange(1_000_000))
    track = RouteTrack.from_network(args.route, mode=args.route_mode,
                                    n_frames=n_planned, seed=route_seed,
                                    speed_mps=args.speed_mps,
                                    fps=args.fps, accuracy_m=args.gps_accuracy_m)

    run_id = args.run_id or str(uuid.uuid4())
    client = redis.from_url(settings.redis_url, decode_responses=False)

    # feedsim is the only component that knows run_id, target_fps, transport and the
    # source, so it registers the run -- nothing else in the running system writes
    # survey_runs. This puts a Postgres write on the "edge" side of the frames-stream
    # seam described in README.md; see the note there for why that's accepted.
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        apply_migrations(conn)
        repo = Repository(conn)
        repo.upsert_run(run_id=run_id, authority_id=args.authority_id,
                        started_at=datetime.now(timezone.utc),
                        source_kind=args.source_kind, source_ref=str(args.manifest),
                        target_fps=args.fps, prevalence=None,
                        transport=args.transport,
                        # Written BEFORE the first frame, so the dashboard can draw the
                        # whole planned drive while the car is still on it.
                        config={"planned_route": {
                            "mode": args.route_mode, "seed": route_seed,
                            "length_m": round(track.length_m, 1),
                            "speed_mps": args.speed_mps, "fps": args.fps,
                            "points": [[round(la, 6), round(lo, 6)] for la, lo in track.route],
                        }})

        stats = run_feed(client, manifest=args.manifest, run_id=run_id,
                         fps=args.fps, maxlen=settings.frames_maxlen,
                         seed=args.seed, stream=settings.frames_stream,
                         transport=args.transport, track=track,
                         control=redis_control(client, run_id),
                         max_frames=args.max_frames)

        repo.finish_run(stats.run_id, datetime.now(timezone.utc), config={
            "frames_offered": stats.offered,
            "frames_dropped": stats.dropped,
            "raw_bytes_offered": stats.raw_bytes,
            "cancelled": stats.cancelled,
            "gps_track": {
                "kind": "route", "network": str(args.route), "mode": args.route_mode,
                "seed": route_seed,
                "route_length_m": round(track.length_m, 1),
                "speed_mps": args.speed_mps, "fps": args.fps,
                "accuracy_m": args.gps_accuracy_m,
                "expected_distance_m": round(track.distance_m(stats.published), 3),
            },
        })

    print(f"run_id={stats.run_id} offered={stats.offered} "
          f"{'CANCELLED ' if stats.cancelled else ''}"
          f"published={stats.published} dropped={stats.dropped} "
          f"distance_m={track.distance_m(stats.published):.1f}")


if __name__ == "__main__":
    main()
