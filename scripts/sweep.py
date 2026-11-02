"""Benchmark sweep: latency, throughput ceiling, resource use, storage vs prevalence, chaos.

    make bench                       # = all four grids below, into bench-out/<date>/
    uv run python -m scripts.sweep baseline throughput prevalence chaos --out bench-out/x

Each grid point is one survey run through the real Compose stack:

  1. recreate the workers at the point's replica count, buffer depth and blob root
  2. (optional) a short warm-up feed on images the run does NOT use, so model
     start-up cost is not charged to the run
  3. feed-sim replays the point's images at the point's rate
  4. a background sampler records `docker stats` for every container each second,
     and another records bus lag / pending / frames landed
  5. drain → segmenter --once → bench evaluate → resources written into bench_runs
  6. edgecv.bench.export writes the run's CSVs; scripts/charts.py draws them

Grids (spec §8, cut to fit the time left):

  baseline     whole RDD2022 test split, 2 workers, 8 fps, buffer 1000 — the
               product as demoed. Latency, accuracy, PR curve, confusion, resources.
  throughput   workers {1,2,3} x offered fps {4,8,12,16}, 600 frames each, buffer
               200. The workers x fps surface is the throughput ceiling; the small
               buffer makes running past it show up as drops within 600 frames.
  prevalence   10 / 33 / 66 % damaged frames, 600 frames, 2 workers, 8 fps. Bytes
               stored per km as road damage rises (success criterion 1).
  chaos        1,200 frames, 2 workers, 8 fps, one worker SIGKILLed 40 s in.
               Bus pending over time shows XAUTOCLAIM recovering its frames.

Every run gets its own blob root, so bytes are attributable to it, and the store is
measured just before the run so warm-up crops are subtracted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = os.environ.setdefault("COMPOSE_PROJECT_NAME", "roadiq-bench")
# Shervin's v4 weights, not the public Hugging Face set the compose file defaults to:
# the public weights may have been trained on these very test images (C4 §3.4.1), so
# their accuracy here would be inflated. Latency is the same either way — same
# architecture, same input size. Override YOLO_WEIGHTS to benchmark the other set.
os.environ.setdefault("YOLO_WEIGHTS", "src/edgecv/detectors/yolo12s_rdd2022_continued.pt")

import psycopg  # noqa: E402
import redis  # noqa: E402

from edgecv.bench.export import export_run  # noqa: E402
from edgecv.bench.resources import DockerStatsSampler, summarise, write_samples  # noqa: E402
from edgecv.bus.observe import group_health  # noqa: E402
from edgecv.config import Settings  # noqa: E402
from edgecv.runner import GT, MANIFEST, feed_args, wait_for_drain  # noqa: E402

BENCH_DATA = Path("data/bench")
ROUTE_SEED = 42           # one simulated route for every run, so distance is comparable
FEED_SEED = 0             # feed-sim's shuffle seed (its own default)
WARMUP_FRAMES = 16
CHAOS_KILL_AFTER_S = 40
SPEED_MPS = 13.89         # 50 km/h


# --- shell ---------------------------------------------------------------------------

def compose(*args: str, env: dict | None = None, capture: bool = False) -> str:
    cmd = ["docker", "compose", *args]
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, check=True, text=True,
                         env=os.environ | {k: str(v) for k, v in (env or {}).items()},
                         stdout=subprocess.PIPE if capture else None)
    return res.stdout if capture else ""


def store_bytes(blob_root: str) -> int:
    out = compose("run", "--rm", "-T", "-e", f"BLOB_ROOT={blob_root}", "bench", "python", "-c",
                  "import os; from pathlib import Path; "
                  "from edgecv.blobstore.store import BlobStore; "
                  "print(BlobStore(root=Path(os.environ['BLOB_ROOT'])).total_bytes())",
                  capture=True)
    return int(out.strip().splitlines()[-1])


# --- manifests -----------------------------------------------------------------------

def _full() -> dict:
    return json.loads(Path(MANIFEST).read_text())


def _shuffled_full() -> list[str]:
    """The exact order feed-sim replays the full manifest in (feedsim.run_feed)."""
    full = _full()
    paths = [e["path"] for e in full["clean"]] + [e["path"] for e in full["defect"]]
    random.Random(FEED_SEED).shuffle(paths)
    return paths


def build_manifests(max_run_frames: int, prevalences: list[float], n: int) -> dict:
    """Warm-up and prevalence manifests from images no measured run of the full
    manifest uses (the first `max_run_frames` of feed-sim's shuffle)."""
    BENCH_DATA.mkdir(parents=True, exist_ok=True)
    full = _full()
    defect = {e["path"] for e in full["defect"]}
    reserved = set(_shuffled_full()[:max_run_frames])
    spare = [p for p in _shuffled_full()[max_run_frames:]]
    rng = random.Random(7)
    warm = [p for p in spare if p in defect][:WARMUP_FRAMES]
    taken = reserved | set(warm)
    spare_clean = [p for p in spare if p not in defect and p not in taken]
    spare_defect = [p for p in spare if p in defect and p not in taken]
    out = {"warmup": BENCH_DATA / "warmup.json"}
    out["warmup"].write_text(json.dumps({"source": "bench warm-up", "clean": [],
                                         "defect": [{"path": p} for p in warm]}))
    for prev in prevalences:
        k = round(prev * n)
        path = BENCH_DATA / f"prevalence-{round(prev * 100):02d}.json"
        path.write_text(json.dumps({
            "source": f"bench prevalence {prev:.0%}",
            "clean": [{"path": p} for p in rng.sample(spare_clean, n - k)],
            "defect": [{"path": p} for p in rng.sample(spare_defect, k)]}))
        out[prev] = path
    return out


# --- grids ---------------------------------------------------------------------------

def grid(names: list[str], args) -> list[dict]:
    points = []
    if "baseline" in names:
        points.append({"grid": "baseline", "label": "baseline", "workers": 2, "fps": 8.0,
                       "maxlen": 1000, "frames": None, "manifest": MANIFEST,
                       "warmup": False, "prevalence": None})
    if "throughput" in names:
        for w in args.workers_list:
            for fps in args.fps_list:
                points.append({"grid": "throughput", "label": f"tp-w{w}-fps{fps:g}",
                               "workers": w, "fps": fps, "maxlen": args.sweep_maxlen,
                               "frames": args.frames, "manifest": MANIFEST,
                               "warmup": True, "prevalence": None})
    if "prevalence" in names:
        for prev in args.prevalences:
            points.append({"grid": "prevalence", "label": f"prev-{round(prev * 100):02d}",
                           "workers": 2, "fps": 8.0, "maxlen": 1000, "frames": None,
                           "manifest": None, "warmup": True, "prevalence": prev})
    if "chaos" in names:
        points.append({"grid": "chaos", "label": "chaos-kill-1-of-2", "workers": 2,
                       "fps": 8.0, "maxlen": 1000, "frames": args.chaos_frames,
                       "manifest": MANIFEST, "warmup": True, "prevalence": None,
                       "kill_after_s": CHAOS_KILL_AFTER_S})
    return points


# --- samplers ------------------------------------------------------------------------

class BusSampler:
    """Once a second: frames-stream lag and pending, results backlog, frames landed."""

    def __init__(self, settings: Settings, run_id: str):
        self.settings, self.run_id = settings, run_id
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "BusSampler":
        self._thread.start()
        return self

    def _loop(self) -> None:
        client = redis.from_url(self.settings.redis_url)
        with psycopg.connect(self.settings.pg_dsn, autocommit=True) as conn:
            while not self._stop.is_set():
                t = time.time()
                try:
                    h = group_health(client, stream_name=self.settings.frames_stream,
                                     group="workers")
                    results_len = client.xlen(self.settings.results_stream)
                    with conn.cursor() as cur:
                        cur.execute("SELECT count(*) FROM frames WHERE run_id = %s",
                                    (self.run_id,))
                        (landed,) = cur.fetchone()
                    self.rows.append({"t": round(t, 3), "lag": h.lag, "pending": h.pending,
                                      "stream_len": h.stream_length,
                                      "results_len": results_len, "landed": landed})
                except (redis.RedisError, psycopg.Error) as exc:
                    print(f"  bus sample failed: {exc}", flush=True)
                self._stop.wait(max(0.0, 1.0 - (time.time() - t)))

    def stop(self) -> list[dict]:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- one point -----------------------------------------------------------------------

def set_workers(n: int, env: dict) -> None:
    since = datetime.now(timezone.utc).isoformat()
    # --no-deps: never recreate redis/postgres between points.
    compose("up", "-d", "--no-build", "--no-deps", "--force-recreate",
            "--scale", f"worker={n}", "worker", env=env)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        logs = compose("logs", "--since", since, "worker", capture=True)
        if logs.count("started detector=") >= n:
            return
        time.sleep(2)
    raise SystemExit(f"workers did not report ready: {n} expected")


def feed(run_id: str, *, fps: float, frames: int | None, manifest: str, env: dict) -> None:
    compose("run", "--rm", "-T", "feedsim", "python", "-m", "edgecv.feedsim.main",
            *feed_args(run_id=run_id, fps=fps, speed_mps=SPEED_MPS, route_mode="random",
                       route_seed=ROUTE_SEED, max_frames=frames, manifest=manifest),
            env=env)


def run_point(pt: dict, out_root: Path, stamp: str, settings: Settings,
              manifests: dict) -> dict:
    label = pt["label"]
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)
    blob_root = f"/blobs/bench-{stamp}/{label}"
    env = {"WORKERS": pt["workers"], "BLOB_ROOT": blob_root, "FRAMES_MAXLEN": pt["maxlen"]}
    manifest = pt["manifest"] or manifests[pt["prevalence"]].as_posix()
    print(f"\n=== {label}: {pt['workers']} workers · {pt['fps']:g} fps · "
          f"buffer {pt['maxlen']} · {pt['frames'] or 'all'} frames · {manifest}", flush=True)

    set_workers(pt["workers"], env)
    conn = psycopg.connect(settings.pg_dsn, autocommit=True)
    rclient = redis.from_url(settings.redis_url)
    if pt["warmup"]:
        warm_id = str(uuid.uuid4())
        feed(warm_id, fps=4, frames=None, manifest=manifests["warmup"].as_posix(), env=env)
        wait_for_drain(conn, rclient, warm_id, timeout_s=300,
                       stream=settings.frames_stream)
    bytes_before = store_bytes(blob_root)

    run_id = str(uuid.uuid4())
    stats = DockerStatsSampler(PROJECT).start()
    bus = BusSampler(settings, run_id).start()
    time.sleep(2)                               # a couple of idle samples first
    t_feed0 = time.time()
    kill = {}
    if pt.get("kill_after_s"):
        def killer():
            time.sleep(pt["kill_after_s"])
            name = f"{PROJECT}-worker-1"
            kill["t"] = time.time()
            kill["container"] = name
            subprocess.run(["docker", "kill", name], check=False)
            print(f"  CHAOS: killed {name} at +{kill['t'] - t_feed0:.1f} s", flush=True)
        threading.Thread(target=killer, daemon=True).start()
    feed(run_id, fps=pt["fps"], frames=pt["frames"], manifest=manifest, env=env)
    t_feed1 = time.time()
    wait_for_drain(conn, rclient, run_id, timeout_s=1800, stream=settings.frames_stream)
    t_drain = time.time()
    time.sleep(2)
    samples = stats.stop()
    bus_rows = bus.stop()

    compose("run", "--rm", "-T", "segmenter", "python", "-m", "edgecv.segmenter.main",
            "--once", "--run-id", run_id, env=env)
    compose("run", "--rm", "-T", "bench", "python", "-m", "edgecv.bench.evaluate",
            "--load-gt", GT, "--run-id", run_id, env=env)

    res = summarise(samples, t0=t_feed0, t1=t_drain)
    worker = res.get("worker", {})
    with conn.cursor() as cur:
        cur.execute("SELECT bench_run_id, bytes_stored FROM bench_runs b "
                    "JOIN survey_runs r USING (bench_run_id) WHERE r.run_id = %s", (run_id,))
        bench_id, bytes_after = cur.fetchone()
        run_bytes = int(bytes_after) - bytes_before
        cur.execute(
            "UPDATE bench_runs SET peak_rss_mb = %s, cpu_pct = %s, bytes_stored = %s, "
            "label = %s, grid_point = grid_point || %s::jsonb WHERE bench_run_id = %s",
            (worker.get("mem_peak_mb"), res.get("pipeline", {}).get("cpu_mean_pct"),
             run_bytes, f"sweep {label}",
             json.dumps({"sweep": pt | {"manifest": manifest}, "resources": res,
                         "bytes_store_total": int(bytes_after),
                         "bytes_before_run": bytes_before}),
             bench_id))
    summary = export_run(conn, run_id, out)
    conn.close()

    t0 = samples[0]["t"] if samples else t_feed0
    write_samples(out / "resources.csv", samples)
    _write_rows(out / "bus.csv", bus_rows)
    (out / "point.json").write_text(json.dumps({
        "point": pt | {"manifest": manifest}, "run_id": run_id, "env": env,
        "t_sampling_start": t0, "t_feed_start": t_feed0, "t_feed_end": t_feed1,
        "t_drained": t_drain, "kill": kill, "resources": res}, indent=2))

    st = summary["stages"]
    acc = summary["accuracy"] or {}
    return {"grid": pt["grid"], "label": label, "run_id": run_id,
            "workers": pt["workers"], "offered_fps": pt["fps"], "buffer": pt["maxlen"],
            "prevalence": pt["prevalence"],
            "frames_offered": summary["frames_offered"],
            "frames_dropped": summary["frames_dropped"],
            "frames_landed": summary["frames_landed"],
            "drop_pct": round(100 * (summary["frames_dropped"] or 0)
                              / max(1, summary["frames_offered"] or 1), 2),
            "achieved_fps": summary["achieved_fps"],
            "e2e_p50_ms": st["e2e_ms"]["p50"], "e2e_p95_ms": st["e2e_ms"]["p95"],
            "e2e_p99_ms": st["e2e_ms"]["p99"],
            "queue_p50_ms": st["queue_ms"]["p50"], "queue_p95_ms": st["queue_ms"]["p95"],
            "infer_p50_ms": st["infer_ms"]["p50"], "infer_p95_ms": st["infer_ms"]["p95"],
            "write_p50_ms": st["write_ms"]["p50"], "write_p95_ms": st["write_ms"]["p95"],
            "precision": acc.get("precision"), "recall": acc.get("recall"),
            "f1": acc.get("f1"), "map50_scored": summary["map50_scored"],
            "bytes_raw": summary["bytes"]["raw_offered"], "bytes_stored": run_bytes,
            "reduction_x": (round(summary["bytes"]["raw_offered"] / run_bytes, 2)
                            if run_bytes and summary["bytes"]["raw_offered"] else None),
            "assessed_km": round(summary["coverage"]["assessed_m"] / 1000, 3),
            "coverage_pct": summary["coverage"]["coverage_pct"],
            "defect_frame_share": summary["defect_frame_share"],
            "pipeline_cpu_mean_pct": res.get("pipeline", {}).get("cpu_mean_pct"),
            "worker_cpu_mean_pct": worker.get("cpu_mean_pct"),
            "worker_mem_peak_mb": worker.get("mem_peak_mb"),
            "pipeline_mem_peak_mb": res.get("pipeline", {}).get("mem_total_peak_mb"),
            "chaos_kill_at_s": round(kill["t"] - t_feed0, 1) if kill else None}


def hardware() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], text=True,
                                  capture_output=True).stdout.strip()
        except OSError:
            return None
    info = json.loads(subprocess.run(["docker", "info", "--format", "{{json .}}"],
                                     text=True, capture_output=True).stdout or "{}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True,
                            capture_output=True).stdout.strip()
    mem = sysctl("hw.memsize")
    return {"host_cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "host_cores": sysctl("hw.ncpu"),
            "host_mem_gb": round(int(mem) / 1024 ** 3, 1) if mem else None,
            "host_os": platform.platform(),
            "docker_vm_cpus": info.get("NCPU"),
            "docker_vm_mem_gb": round(info.get("MemTotal", 0) / 1024 ** 3, 2),
            "docker_version": info.get("ServerVersion"),
            "detector": os.environ.get("DETECTOR", "yolo12s"),
            "weights": os.environ.get("YOLO_WEIGHTS", "weights/yolo12s_RDD2022_best.pt"),
            "worker_threads": int(os.environ.get("WORKER_THREADS", 6)),
            "gpu": "none (CPU-only torch)", "commit": commit,
            "compose_project": PROJECT}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grids", nargs="+",
                    choices=["baseline", "throughput", "prevalence", "chaos"])
    ap.add_argument("--out", type=Path,
                    default=Path("bench-out") / datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--frames", type=int, default=600, help="frames per throughput point")
    ap.add_argument("--chaos-frames", type=int, default=1200)
    ap.add_argument("--workers-list", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 2, 3])
    ap.add_argument("--fps-list", type=lambda s: [float(x) for x in s.split(",")],
                    default=[4.0, 8.0, 12.0, 16.0])
    ap.add_argument("--prevalences", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.10, 0.33, 0.66])
    ap.add_argument("--sweep-maxlen", type=int, default=200,
                    help="stream buffer depth for the throughput grid")
    args = ap.parse_args()

    settings = Settings.from_env()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifests = build_manifests(max(args.frames, args.chaos_frames), args.prevalences,
                                n=args.frames)
    (args.out / "hardware.json").write_text(json.dumps(hardware(), indent=2))

    compose("up", "-d", "--no-build", "redis", "postgres", "writer", "segmenter", "api",
            env={"WORKERS": 0})
    index = args.out / "runs.csv"
    rows = list(csv.DictReader(index.open())) if index.exists() else []
    for pt in grid(args.grids, args):
        row = run_point(pt, args.out, stamp, settings, manifests)
        rows = [r for r in rows if r["label"] != row["label"]] + [row]
        _write_rows(index, rows)            # after every point: a crash keeps the rest
        print(f"  → {row['label']}: {row['frames_landed']}/{row['frames_offered']} landed, "
              f"{row['frames_dropped']} dropped, {row['achieved_fps']} fps, "
              f"e2e p95 {row['e2e_p95_ms']} ms, worker CPU {row['worker_cpu_mean_pct']}%",
              flush=True)
    print(f"\nsweep complete → {args.out}")


if __name__ == "__main__":
    sys.exit(main())
