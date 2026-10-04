"""One survey run, end to end, from inside a container — what the dashboard's "New run"
button starts.

    python -m edgecv.runner --run-id R [--fps 8] [--max-frames N] [--route-mode random]
    python -m edgecv.runner --serve        # the `runner` compose service

`--serve` waits on the Redis list `runner:requests` and runs one request at a time. It is
its own container ON PURPOSE: the first version ran inside the API container, and a
routine API redeploy killed a 12 fps run mid-feed (2026-09-27). Progress goes to the
`runner:status` hash and the `runner:log` list, which the read API reports.

The same sequence as `make e2e` (scripts/e2e.py), but each step is a plain `python -m`
process instead of `docker compose run`, because the read API that launches it has no
Docker socket. Every component still runs as its own process with its own code:
feed-sim replays → the always-on workers and writer drain → segmenter --once → bench.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import psycopg
import redis

from edgecv.bus.observe import group_health
from edgecv.config import Settings

ROUTE = "src/edgecv/roads/lackey_road.geojson"
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


REQUESTS = "runner:requests"
STATUS = "runner:status"
LOG = "runner:log"
LOG_LINES = 60


def _step(*module_and_args: str, sink=None) -> None:
    cmd = [sys.executable, "-m", *module_and_args]
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    if sink is None:
        subprocess.run(cmd, check=True)
        return
    sink(line)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for out in proc.stdout:
        print(out, end="", flush=True)
        sink(out.rstrip())
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_once(req: dict, *, settings: Settings, sink=None, on_phase=lambda _p: None) -> None:
    """Feed → drain → segment → bench for one request (a StartRun body + run_id)."""
    t0 = time.monotonic()
    on_phase("feeding")
    _step("edgecv.feedsim.main", *feed_args(
        run_id=req["run_id"], fps=req.get("fps", 8.0), speed_mps=req.get("speed_mps", 13.89),
        route_mode=req.get("route_mode", "random"), route_seed=req.get("route_seed"),
        max_frames=req.get("max_frames")), sink=sink)
    on_phase("scoring")
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        wait_for_drain(conn, redis.from_url(settings.redis_url), req["run_id"],
                       timeout_s=req.get("timeout_s", 900), stream=settings.frames_stream)
    _step("edgecv.segmenter.main", "--once", "--run-id", req["run_id"], sink=sink)
    _step("edgecv.bench.evaluate", "--load-gt", GT, "--run-id", req["run_id"], sink=sink)
    msg = f"run {req['run_id']} complete in {time.monotonic() - t0:.0f} s"
    print(msg, flush=True)
    if sink:
        sink(msg)


def serve(settings: Settings) -> None:
    client = redis.from_url(settings.redis_url, decode_responses=True)
    print("runner waiting for requests", flush=True)
    while True:
        # Short blocking waits, and a lost socket is retried, not fatal: the first version
        # used a 30 s BLPOP, hit a socket read timeout while idle, and the service died
        # with a request still queued ("Starting run…" forever).
        try:
            item = client.blpop(REQUESTS, timeout=5)
        except (redis.TimeoutError, redis.ConnectionError) as exc:
            print(f"runner: redis wait failed ({exc}); retrying", flush=True)
            time.sleep(2)
            continue
        if item is None:
            continue
        req = json.loads(item[1])
        run_id = req["run_id"]
        client.delete(LOG)

        def sink(line: str) -> None:
            if "landed " in line and "bus lag" in line:
                return                                   # drain heartbeat: noise
            client.rpush(LOG, line)
            client.ltrim(LOG, -LOG_LINES, -1)

        def on_phase(phase: str) -> None:
            client.hset(STATUS, mapping={"run_id": run_id, "state": phase,
                                         "updated": time.time()})

        client.hset(STATUS, mapping={"run_id": run_id, "state": "starting",
                                     "request": json.dumps(req), "started": time.time(),
                                     "updated": time.time(), "exit_code": ""})
        try:
            run_once(req, settings=settings, sink=sink, on_phase=on_phase)
            client.hset(STATUS, mapping={"state": "done", "exit_code": 0, "updated": time.time()})
        except (subprocess.CalledProcessError, SystemExit, psycopg.Error, redis.RedisError) as exc:
            sink(f"FAILED: {exc}")
            client.hset(STATUS, mapping={"state": "failed", "exit_code": 1,
                                         "updated": time.time()})


def main() -> None:
    ap = argparse.ArgumentParser(description="One e2e survey run, in-container")
    ap.add_argument("--serve", action="store_true",
                    help="run requests from the runner:requests queue, forever")
    ap.add_argument("--run-id")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--speed-mps", type=float, default=13.89)
    ap.add_argument("--route-mode", choices=["random", "loop", "cover"], default="random")
    ap.add_argument("--route-seed", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--timeout-s", type=float, default=900)
    args = ap.parse_args()

    settings = Settings.from_env()
    if args.serve:
        serve(settings)
        return
    if not args.run_id:
        ap.error("--run-id is required unless --serve")
    run_once({"run_id": args.run_id, "fps": args.fps, "speed_mps": args.speed_mps,
              "route_mode": args.route_mode, "route_seed": args.route_seed,
              "max_frames": args.max_frames, "timeout_s": args.timeout_s},
             settings=settings)


if __name__ == "__main__":
    main()
