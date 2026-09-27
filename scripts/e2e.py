"""Run the RDD2022 test split through the whole system, end to end.

    make e2e            (= make stack, then this)

1. feed-sim (in its container) replays every test image once along the Sydney route.
2. Wait until the writer has landed every published frame and the bus is empty.
3. Segmenter --once for the run (the always-on service would get there too; running it
   once here makes the finish line deterministic).
4. Bench: load ground truth, score the run, write bench_runs.

Runs on the host and talks to Postgres/Redis through the published ports; every
component itself runs in its Compose container.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid

import psycopg
import redis

from edgecv.config import Settings
from edgecv.runner import GT, feed_args, wait_for_drain


def compose(*args: str) -> None:
    cmd = ["docker", "compose", "run", "--rm", "-T", *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--speed-mps", type=float, default=13.89)
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--route-mode", choices=["random", "loop", "cover"], default="random")
    ap.add_argument("--route-seed", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    settings = Settings.from_env()
    run_id = str(uuid.uuid4())
    t0 = time.monotonic()
    compose("feedsim", "python", "-m", "edgecv.feedsim.main", *feed_args(
        run_id=run_id, fps=args.fps, speed_mps=args.speed_mps, route_mode=args.route_mode,
        route_seed=args.route_seed, max_frames=args.max_frames))
    print(f"feed finished in {time.monotonic() - t0:.0f} s; waiting for the pipeline to drain",
          flush=True)
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        wait_for_drain(conn, redis.from_url(settings.redis_url), run_id,
                       timeout_s=args.timeout_s)
    compose("segmenter", "python", "-m", "edgecv.segmenter.main", "--once", "--run-id", run_id)
    compose("bench", "python", "-m", "edgecv.bench.evaluate", "--load-gt", GT,
            "--run-id", run_id)
    print(f"\nE2E complete in {time.monotonic() - t0:.0f} s · run {run_id}\n"
          f"Dashboard: http://localhost:8000   Pipeline panel: http://localhost:8501")


if __name__ == "__main__":
    sys.exit(main())
