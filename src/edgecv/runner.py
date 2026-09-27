"""One survey run, end to end, from inside a container — what the dashboard's "New run"
button starts.

    python -m edgecv.runner --run-id R [--fps 8] [--max-frames N] [--route-mode random]

The same sequence as `make e2e` (scripts/e2e.py), but each step is a plain `python -m`
process instead of `docker compose run`, because the read API that launches it has no
Docker socket. Every component still runs as its own process with its own code:
feed-sim replays → the always-on workers and writer drain → segmenter --once → bench.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psycopg
import redis

from edgecv.bus.observe import group_health
from edgecv.config import Settings

ROUTE = "src/edgecv/roads/sydney_demo.geojson"
MANIFEST = "data/rdd2022/manifest.json"
GT = "data/rdd2022/ground_truth.json"


def wait_for_drain(conn, client, run_id: str, *, timeout_s: float,
                   stream: str = "frames") -> None:
    """Block until the writer has landed every frame feed-sim published for the run and
    nothing is left in flight on the bus."""
    with conn.cursor() as cur:
        cur.execute("SELECT config FROM survey_runs WHERE run_id = %s", (run_id,))
        cfg = cur.fetchone()[0]
    expected = cfg["frames_offered"] - cfg["frames_dropped"]
    deadline = time.monotonic() + timeout_s
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM frames WHERE run_id = %s", (run_id,))
            (landed,) = cur.fetchone()
        health = group_health(client, stream_name=stream, group="workers")
        print(f"  landed {landed}/{expected} · bus lag {health.lag} · pending {health.pending}",
              flush=True)
        if landed >= expected and health.pending == 0:
            return
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out with {landed}/{expected} frames landed")
        time.sleep(5)


def feed_args(*, run_id: str, fps: float, speed_mps: float, route_mode: str,
              route_seed: int | None, max_frames: int | None) -> list[str]:
    args = ["--manifest", MANIFEST, "--order", "all", "--route", ROUTE,
            "--route-mode", route_mode, "--source-kind", "dataset-replay",
            "--fps", str(fps), "--speed-mps", str(speed_mps), "--run-id", run_id]
    if route_seed is not None:
        args += ["--route-seed", str(route_seed)]
    if max_frames:
        args += ["--max-frames", str(max_frames)]
    return args


def _step(*module_and_args: str) -> None:
    cmd = [sys.executable, "-m", *module_and_args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="One e2e survey run, in-container")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--speed-mps", type=float, default=13.89)
    ap.add_argument("--route-mode", choices=["random", "loop", "cover"], default="random")
    ap.add_argument("--route-seed", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=900)
    args = ap.parse_args()

    settings = Settings.from_env()
    t0 = time.monotonic()
    _step("edgecv.feedsim.main", *feed_args(
        run_id=args.run_id, fps=args.fps, speed_mps=args.speed_mps,
        route_mode=args.route_mode, route_seed=args.route_seed, max_frames=args.max_frames))
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        wait_for_drain(conn, redis.from_url(settings.redis_url), args.run_id,
                       timeout_s=args.timeout_s, stream=settings.frames_stream)
    _step("edgecv.segmenter.main", "--once", "--run-id", args.run_id)
    _step("edgecv.bench.evaluate", "--load-gt", GT, "--run-id", args.run_id)
    print(f"run {args.run_id} complete in {time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
