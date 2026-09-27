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

from edgecv.bus.observe import group_health
from edgecv.config import Settings

ROUTE = "src/edgecv/roads/sydney_demo.geojson"
MANIFEST = "data/rdd2022/manifest.json"
GT = "data/rdd2022/ground_truth.json"


def compose(*args: str) -> None:
    cmd = ["docker", "compose", "run", "--rm", "-T", *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def wait_for_drain(conn, client, run_id: str, *, timeout_s: float) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT config FROM survey_runs WHERE run_id = %s", (run_id,))
        cfg = cur.fetchone()[0]
    expected = cfg["frames_offered"] - cfg["frames_dropped"]
    deadline = time.monotonic() + timeout_s
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM frames WHERE run_id = %s", (run_id,))
            (landed,) = cur.fetchone()
        health = group_health(client, stream_name="frames", group="workers")
        print(f"  landed {landed}/{expected} · bus lag {health.lag} · pending {health.pending}",
              flush=True)
        if landed >= expected and health.pending == 0:
            return
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out with {landed}/{expected} frames landed")
        time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--speed-mps", type=float, default=13.89)
    ap.add_argument("--timeout-s", type=float, default=900)
    args = ap.parse_args()

    settings = Settings.from_env()
    run_id = str(uuid.uuid4())
    t0 = time.monotonic()
    compose("feedsim", "python", "-m", "edgecv.feedsim.main", "--manifest", MANIFEST,
            "--order", "all", "--route", ROUTE, "--source-kind", "dataset-replay",
            "--fps", str(args.fps), "--speed-mps", str(args.speed_mps), "--run-id", run_id)
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
