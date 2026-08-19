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
import redis

from edgecv.bus.producer import FrameProducer
from edgecv.config import Settings
from edgecv.contracts.frame import FrameEnvelope
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
    ap = argparse.ArgumentParser(description="Simulated production-line frame feed")
    ap.add_argument("--manifest", type=Path,
                    default=Path("tests/fixtures/generated/manifest.json"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--prevalence", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--transport", choices=["inline", "reference"],
                    default="reference")
    args = ap.parse_args()

    client = redis.from_url(settings.redis_url, decode_responses=False)
    stats = run_feed(client, manifest=args.manifest, run_id=args.run_id,
                     n_frames=args.frames, fps=args.fps,
                     prevalence=args.prevalence, maxlen=settings.frames_maxlen,
                     seed=args.seed, stream=settings.frames_stream,
                     transport=args.transport)
    print(f"run_id={stats.run_id} offered={stats.offered} "
          f"published={stats.published} dropped={stats.dropped}")


if __name__ == "__main__":
    main()
