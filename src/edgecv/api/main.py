"""Component 7 — the read API (Contract 3), and the host for the dashboard (component 8).

    uvicorn edgecv.api.main:app --host 0.0.0.0 --port 8000

Read-only against Postgres except one write path: human review decisions, which go to
`instance_reviews` (never to the derived `defect_instances`) and then trigger a re-score of
the run, so a Confirm or Reject changes the segment's condition on the next map refresh.

The API is the only thing the browser talks to: frames and crops are served from here, so
the dashboard never needs a path into the blob store or the dataset.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Literal

import psycopg
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from edgecv.bench.collect import run_coverage
from edgecv.blobstore.store import BlobStore
from edgecv.bus.observe import consumer_health, group_health, stuck_entries
from edgecv.config import Settings
from edgecv.contracts.frame import FrameEnvelope
from edgecv.feedsim.main import control_key, state_key
from edgecv.runner import LOG as RUNNER_LOG, REQUESTS as RUNNER_REQUESTS, STATUS as RUNNER_STATUS
from edgecv.roads import load_ways
from edgecv.segmenter.index import AUTO_ACCEPT_CONF
from edgecv.segmenter.main import HEADING_TOL_DEG, SNAP_LATERAL, SNAP_M, segment_run
from edgecv.worker.main import CROP_FMT

SETTINGS = Settings.from_env()
STATIC = Path(__file__).resolve().parent / "static"
#: Frames are served only from under this directory (the dataset mount), whatever
#: `source_ref` says — a path in the database is not a licence to read the filesystem.
FRAME_ROOT = Path(os.environ.get("FRAME_ROOT", "data")).resolve()

app = FastAPI(title="RoadIQ read API", version="0.1.0")


def db() -> psycopg.Connection:
    return psycopg.connect(SETTINGS.pg_dsn, autocommit=True)


def rows(cur) -> list[dict]:
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# Effective review state. Derived, never stored on defect_instances: the segmenter replaces
# those rows every pass, so a human decision there would be destroyed on the next one.
STATE_SQL = f"""coalesce(r.review_state,
    CASE WHEN di.peak_confidence >= {AUTO_ACCEPT_CONF} THEN 'auto-accepted' ELSE 'pending' END)"""


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/static/index.html")


@app.get("/api/health")
def health() -> dict:
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
    return {"ok": True}


@app.get("/api/runs")
def runs() -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.run_id::text, r.authority_id, r.started_at, r.ended_at, r.source_kind,
                   r.target_fps::float8 AS target_fps,
                   (SELECT count(*) FROM defect_instances di WHERE di.run_id = r.run_id)
                       AS instances
            FROM survey_runs r ORDER BY r.started_at DESC""")
        return rows(cur)


def _run(cur, run_id: str) -> dict:
    cur.execute("SELECT run_id::text, authority_id, started_at, ended_at, source_kind, "
                "target_fps::float8 AS target_fps, config FROM survey_runs WHERE run_id = %s",
                (run_id,))
    found = rows(cur)
    if not found:
        raise HTTPException(404, f"run {run_id} not found")
    return found[0]


@app.get("/api/runs/{run_id}/summary")
def summary(run_id: str) -> dict:
    with db() as conn, conn.cursor() as cur:
        run = _run(cur, run_id)
        cov = run_coverage(conn, run_id)
        cur.execute(f"""
            SELECT count(*) FILTER (WHERE {STATE_SQL} <> 'rejected') AS defects,
                   count(*) FILTER (WHERE {STATE_SQL} = 'pending') AS pending
            FROM defect_instances di
            LEFT JOIN instance_reviews r USING (run_id, cluster_key)
            WHERE di.run_id = %s""", (run_id,))
        defects, pending = cur.fetchone()
        cur.execute("SELECT count(*), count(*) FILTER (WHERE EXISTS (SELECT 1 FROM inferences i "
                    "WHERE i.run_id = f.run_id AND i.seq = f.seq AND i.status = 'ok')) "
                    "FROM frames f WHERE f.run_id = %s", (run_id,))
        ingested, processed = cur.fetchone()
        cur.execute("SELECT count(*) FILTER (WHERE condition_band IN ('good','fair','poor')), "
                    "coalesce(sum(coverage_m), 0) FROM segment_condition WHERE run_id = %s",
                    (run_id,))
        scored, seg_cov = cur.fetchone()
    cfg = run["config"] or {}
    if run["ended_at"] is None:
        # Live: feed-sim writes its totals only when it finishes, so read its running
        # counters off the bus instead. Without this, "offered" fell back to what had
        # landed, and a run backing up at 12 fps still read 100 % accounted for.
        published, dropped = _live_counts(run_id)
        offered = max(published + dropped, ingested)
    else:
        offered = cfg.get("frames_offered") or ingested
        dropped = cfg.get("frames_dropped", 0)
    in_flight = max(0, offered - dropped - ingested)
    stored = _run_crop_bytes(run_id)
    return {
        "run": {k: run[k] for k in ("run_id", "authority_id", "started_at", "ended_at",
                                    "source_kind", "target_fps")},
        "assessed_km": round(cov.assessed_m / 1000, 2),
        "gap_m": round(cov.gap_m, 1),
        "defects": defects, "pending_review": pending,
        "frames_offered": offered, "frames_ingested": ingested,
        "frames_processed": processed,
        "frames_dropped": dropped, "frames_in_flight": in_flight,
        # "Accounted for" = the frame's fate is known: it landed, or it was refused by the
        # bounded bus and counted. Frames still in flight are the only unaccounted ones.
        "frames_accounted_pct": round(100 * (ingested + dropped) / offered, 1) if offered else 0.0,
        # THIS run's crops only. Thumbnails are stored but not linked to a run, and the
        # store is shared across runs, so its total would overstate a short run.
        "bytes_stored": int(stored), "raw_bytes": cfg.get("raw_bytes_offered"),
        "segments_scored": scored, "segment_coverage_km": round(float(seg_cov) / 1000, 2),
    }


def _live_counts(run_id: str) -> tuple[int, int]:
    """(published, dropped) so far, from the producer's per-run counters."""
    try:
        client = redis.from_url(SETTINGS.redis_url)
        pub, drop = client.mget(f"stats:{run_id}:published", f"stats:{run_id}:dropped")
        return int(pub or 0), int(drop or 0)
    except (redis.RedisError, ValueError):
        return 0, 0


def _run_crop_bytes(run_id: str) -> int:
    """Bytes of the distinct crops this run's detections point at, measured on disk:
    `snippets.bytes` is a 0 placeholder the writer cannot fill (repository.py, I2)."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT sn.sha256 FROM inferences i
            JOIN detections d ON d.inference_id = i.inference_id
            JOIN snippets sn ON sn.snippet_id = d.snippet_id
            WHERE i.run_id = %s""", (run_id,))
        shas = [r[0] for r in cur.fetchall()]
    store = BlobStore(root=SETTINGS.blob_root)
    total = 0
    for sha in shas:
        try:
            total += store.path_for(sha, kind="crop", fmt=CROP_FMT).stat().st_size
        except OSError:
            pass
    return total


@app.get("/api/runs/{run_id}/segments.geojson")
def segments_geojson(run_id: str) -> dict:
    """Contract 6: every segment of the run's authority, coloured by this run's band.
    Unsurveyed segments are included with band null — "not surveyed" is information."""
    with db() as conn, conn.cursor() as cur:
        run = _run(cur, run_id)
        cur.execute("""
            SELECT json_build_object('type', 'FeatureCollection', 'features',
                   coalesce(json_agg(json_build_object(
                       'type', 'Feature',
                       'geometry', ST_AsGeoJSON(s.geom, 6)::json,
                       'properties', json_build_object(
                           'segment_id', s.segment_id, 'road_name', s.road_name,
                           'road_ref', s.road_ref, 'length_m', round(s.length_m::numeric, 1),
                           'condition_index', sc.condition_index,
                           'condition_band', sc.condition_band,
                           'counts', sc.counts, 'coverage_m', sc.coverage_m,
                           'frames_assessed', sc.frames_assessed))), '[]'::json))
            FROM segments s
            LEFT JOIN segment_condition sc ON sc.segment_id = s.segment_id AND sc.run_id = %s
            WHERE s.authority_id = %s""", (run_id, run["authority_id"]))
        return cur.fetchone()[0]


@app.get("/api/runs/{run_id}/worklist")
def worklist(run_id: str, limit: int = 25) -> list[dict]:
    """Worst-first, with change against the same segment in the authority's previous run."""
    with db() as conn, conn.cursor() as cur:
        run = _run(cur, run_id)
        cur.execute("""
            WITH prev AS (
                SELECT r.run_id FROM survey_runs r
                WHERE r.authority_id = %(auth)s AND r.started_at < %(started)s
                ORDER BY r.started_at DESC LIMIT 1)
            SELECT s.segment_id, s.road_name, s.road_ref, round(s.length_m::numeric, 1) AS length_m,
                   sc.condition_index::float8 AS condition_index, sc.condition_band,
                   sc.counts, sc.coverage_m,
                   (SELECT coalesce(sum(value::int), 0) FROM jsonb_each_text(sc.counts)
                     WHERE key <> 'pending') AS defects,
                   p.condition_index::float8 AS previous_index, pr.started_at AS previous_at
            FROM segment_condition sc
            JOIN segments s USING (segment_id)
            LEFT JOIN prev ON TRUE
            LEFT JOIN segment_condition p ON p.segment_id = sc.segment_id AND p.run_id = prev.run_id
            LEFT JOIN survey_runs pr ON pr.run_id = prev.run_id
            WHERE sc.run_id = %(run)s AND sc.condition_index IS NOT NULL
            ORDER BY sc.condition_index ASC, s.length_m DESC
            LIMIT %(limit)s""",
            {"auth": run["authority_id"], "started": run["started_at"], "run": run_id,
             "limit": limit})
        out = rows(cur)
    for r in out:
        r["change"] = (round(r["condition_index"] - r["previous_index"], 1)
                       if r["previous_index"] is not None else None)
    return out


_INSTANCE_COLS = f"""
    di.cluster_key, di.segment_id, s.road_name, s.road_ref, di.defect_class,
    r.new_class, di.peak_confidence::float8 AS confidence, di.lat, di.lon, di.along_m,
    di.observation_count, di.first_seen_at, di.severity, {STATE_SQL} AS state,
    r.reviewed_by, r.reviewed_at"""


@app.get("/api/runs/{run_id}/instances")
def instances(run_id: str, state: str | None = None, segment_id: int | None = None,
              limit: int = 50000) -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {_INSTANCE_COLS}
            FROM defect_instances di
            LEFT JOIN instance_reviews r USING (run_id, cluster_key)
            LEFT JOIN segments s ON s.segment_id = di.segment_id
            WHERE di.run_id = %(run)s
              AND (%(state)s::text IS NULL OR {STATE_SQL} = %(state)s)
              AND (%(seg)s::bigint IS NULL OR di.segment_id = %(seg)s)
            ORDER BY (di.defect_class = 'other'), di.peak_confidence DESC
            LIMIT %(limit)s""", {"run": run_id, "state": state, "seg": segment_id,
                                 "limit": limit})
        return rows(cur)


@app.get("/api/instances/{run_id}/{cluster_key}")
def instance(run_id: str, cluster_key: str) -> dict:
    """Everything the evidence panel shows: the instance, its best frame, and every box
    detected on that frame (so the panel can draw accepted and review boxes together)."""
    with db() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {_INSTANCE_COLS}, d.detection_id, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                   d.area_px, sn.sha256 AS crop_sha256, sn.format AS crop_format,
                   s.surface_type, i.seq, i.inference_id, i.latency_ms::float8 AS latency_ms,
                   i.worker_id, f.width, f.height, f.captured_at, f.speed_mps,
                   f.heading_deg, f.gps_accuracy_m, f.source_ref,
                   dt.name AS detector, dt.version AS detector_version
            FROM defect_instances di
            LEFT JOIN instance_reviews r USING (run_id, cluster_key)
            LEFT JOIN segments s ON s.segment_id = di.segment_id
            JOIN detections d ON d.detection_id = di.best_detection_id
            LEFT JOIN snippets sn ON sn.snippet_id = d.snippet_id
            JOIN inferences i ON i.inference_id = d.inference_id
            JOIN detectors dt ON dt.detector_id = i.detector_id
            JOIN frames f ON f.run_id = i.run_id AND f.seq = i.seq AND f.captured_at = i.captured_at
            WHERE di.run_id = %s AND di.cluster_key = %s""", (run_id, cluster_key))
        found = rows(cur)
        if not found:
            raise HTTPException(404, "instance not found")
        inst = found[0]
        # snippets.format is the writer's placeholder; the worker's constant is the truth.
        inst["crop_format"] = CROP_FMT
        cur.execute(f"""
            SELECT d.detection_id, d.defect_class, d.confidence::float8 AS confidence,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, di.cluster_key, {STATE_SQL} AS state
            FROM detections d
            LEFT JOIN defect_instances di ON di.best_detection_id = d.detection_id
            LEFT JOIN instance_reviews r ON r.run_id = di.run_id AND r.cluster_key = di.cluster_key
            WHERE d.inference_id = %s ORDER BY d.confidence DESC""", (inst["inference_id"],))
        inst["frame_boxes"] = rows(cur)
        cur.execute("SELECT count(*) FROM ground_truth WHERE source_ref = %s",
                    (inst["source_ref"],))
        inst["ground_truth_boxes"] = cur.fetchone()[0]
    return inst


@app.get("/api/frames/{run_id}/{seq}/image")
def frame_image(run_id: str, seq: int):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT source_ref FROM frames WHERE run_id = %s AND seq = %s LIMIT 1",
                    (run_id, seq))
        hit = cur.fetchone()
    if not hit:
        raise HTTPException(404, "frame not found")
    path = Path(hit[0]).resolve()
    if not path.is_relative_to(FRAME_ROOT) or not path.is_file():
        raise HTTPException(404, "frame pixels not available (only crops left the vehicle)")
    return FileResponse(path)


@app.get("/api/blobs/{kind}/{sha256}")
def blob(kind: Literal["crop", "thumbnail"], sha256: str, fmt: str = "webp"):
    if len(sha256) != 64 or not all(c in "0123456789abcdef" for c in sha256):
        raise HTTPException(400, "bad sha256")
    path = BlobStore(root=SETTINGS.blob_root).path_for(sha256, kind=kind, fmt=fmt)
    if not path.is_file():
        raise HTTPException(404, "blob not found")
    return FileResponse(path, media_type=f"image/{fmt}")


class Review(BaseModel):
    state: Literal["confirmed", "rejected", "reclassified", "pending"]
    new_class: Literal["D00", "D10", "D20", "D40"] | None = None
    reviewed_by: str = "officer"


@app.post("/api/instances/{run_id}/{cluster_key}/review")
def review(run_id: str, cluster_key: str, body: Review) -> dict:
    if body.state == "reclassified" and not body.new_class:
        raise HTTPException(422, "reclassified needs new_class")
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT segment_id FROM defect_instances WHERE run_id = %s "
                        "AND cluster_key = %s", (run_id, cluster_key))
            hit = cur.fetchone()
            if not hit:
                raise HTTPException(404, "instance not found")
            cur.execute("""
                INSERT INTO instance_reviews (run_id, cluster_key, review_state, new_class,
                                              reviewed_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id, cluster_key) DO UPDATE
                SET review_state = EXCLUDED.review_state, new_class = EXCLUDED.new_class,
                    reviewed_by = EXCLUDED.reviewed_by, reviewed_at = now()""",
                (run_id, cluster_key, body.state, body.new_class, body.reviewed_by))
        segment_run(conn, run_id)       # the decision moves the score now, not next pass
        with conn.cursor() as cur:
            cur.execute("SELECT condition_index::float8, condition_band FROM segment_condition "
                        "WHERE run_id = %s AND segment_id = %s", (run_id, hit[0]))
            cond = cur.fetchone()
    return {"cluster_key": cluster_key, "state": body.state, "segment_id": hit[0],
            "condition_index": cond[0] if cond else None,
            "condition_band": cond[1] if cond else None}


@app.get("/api/runs/{run_id}/bench")
def bench(run_id: str) -> dict | None:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT b.* FROM bench_runs b JOIN survey_runs r ON r.bench_run_id = b.bench_run_id
            WHERE r.run_id = %s""", (run_id,))
        found = rows(cur)
    return found[0] if found else None


@app.get("/api/network")
def network() -> dict:
    """The roads the vehicle can drive that are NOT the council's (state arterials) —
    drawn grey on the map, never segmented or scored (the mock-up's "Not ours")."""
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": w.name, "highway": w.highway},
         "geometry": {"type": "LineString",
                      "coordinates": [[lon, lat] for lat, lon in w.points]}}
        for w in load_ways() if not w.council]}


def _latest_published(run_id: str) -> dict | None:
    """The newest frame feed-sim has put on the bus for this run: where the vehicle IS,
    as opposed to where the pipeline has caught up to. None if the bus has moved on."""
    try:
        client = redis.from_url(SETTINGS.redis_url, decode_responses=False)
        for _id, fields in client.xrevrange(SETTINGS.frames_stream, count=1):
            env = FrameEnvelope.from_fields(fields)
            if env.run_id == run_id and env.lat is not None:
                return {"seq": env.seq, "lat": env.lat, "lon": env.lon,
                        "heading_deg": env.heading_deg, "speed_mps": env.speed_mps,
                        "captured_at": env.captured_at, "source": "bus"}
    except (redis.RedisError, ValueError, AttributeError):
        pass
    return None


def _feed_state(run_id: str) -> str | None:
    try:
        raw = redis.from_url(SETTINGS.redis_url).get(state_key(run_id))
        return raw.decode() if raw else None
    except redis.RedisError:
        return None


class Control(BaseModel):
    action: Literal["pause", "resume", "cancel"]


@app.post("/api/runs/{run_id}/control")
def control(run_id: str, body: Control) -> dict:
    """Pause, resume or cancel a live run. feed-sim reads the key before every frame;
    the workers and writer keep draining whatever is already on the bus."""
    with db() as conn, conn.cursor() as cur:
        run = _run(cur, run_id)
    if run["ended_at"] is not None:
        raise HTTPException(409, "run already finished")
    value = {"pause": "pause", "resume": "run", "cancel": "cancel"}[body.action]
    # Expires so a crashed feed-sim can't leave a stale "pause" for a reused run id.
    redis.from_url(SETTINGS.redis_url).set(control_key(run_id), value, ex=6 * 3600)
    return {"run_id": run_id, "requested": body.action, "feed_state": _feed_state(run_id)}


class StartRun(BaseModel):
    fps: float = 8.0
    max_frames: int | None = None          # None = the whole test split
    route_mode: Literal["random", "loop"] = "random"
    route_seed: int | None = None          # None = a new random route


RUNNER_BUSY = ("starting", "feeding", "scoring")


def _runner_state(client) -> dict:
    return client.hgetall(RUNNER_STATUS) or {}


@app.post("/api/runs/start")
def start_run(body: StartRun) -> dict:
    """Queue a new survey run for the `runner` service (feed-sim → drain → segmenter →
    bench). The API only queues it, so redeploying the API cannot kill a run. Refuses
    while another run is queued, feeding or scoring — from here or from `make e2e`."""
    if not 1 <= body.fps <= 30:
        raise HTTPException(422, "fps must be between 1 and 30")
    if body.max_frames is not None and body.max_frames < 10:
        raise HTTPException(422, "max_frames must be at least 10")
    client = redis.from_url(SETTINGS.redis_url, decode_responses=True)
    state = _runner_state(client)
    if state.get("state") in RUNNER_BUSY or client.llen(RUNNER_REQUESTS):
        raise HTTPException(409, f"run {state.get('run_id', '?')} is still {state.get('state', 'queued')}")
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT run_id::text FROM survey_runs WHERE ended_at IS NULL "
                    "AND started_at > now() - interval '6 hours'")
        open_runs = [r[0] for r in cur.fetchall()]
    feeding = [r for r in open_runs if _feed_state(r) in ("running", "paused")]
    if feeding:
        raise HTTPException(409, f"run {feeding[0]} is still feeding — pause or cancel it first")
    req = {"run_id": str(uuid.uuid4()), **body.model_dump()}
    client.hset(RUNNER_STATUS, mapping={"run_id": req["run_id"], "state": "starting",
                                        "request": json.dumps(req), "started": time.time(),
                                        "updated": time.time(), "exit_code": ""})
    client.rpush(RUNNER_REQUESTS, json.dumps(req))
    return {"run_id": req["run_id"], "request": body.model_dump()}


@app.get("/api/runner")
def runner_status() -> dict:
    """The runner service's current job: state, and the last lines of its log (the drain,
    segment and bench steps after the feed ends are only visible here)."""
    client = redis.from_url(SETTINGS.redis_url, decode_responses=True)
    st = _runner_state(client)
    if not st:
        return {"run_id": None, "alive": False}
    return {"run_id": st.get("run_id"), "state": st.get("state"),
            "alive": st.get("state") in RUNNER_BUSY,
            "exit_code": st.get("exit_code"),
            "request": json.loads(st.get("request") or "{}"),
            "elapsed_s": round(time.time() - float(st.get("started") or time.time())),
            "log_tail": "\n".join(client.lrange(RUNNER_LOG, -12, -1))}


#: Above a healthy frame's processing time, so a stuck entry is a worker that died
#: holding work rather than one that is merely busy.
STUCK_MS = 30_000


@app.get("/api/bus")
def bus_health(stuck_ms: int = STUCK_MS) -> dict:
    """What is still in flight on the frames bus — the half Postgres cannot show, since it
    only holds what already landed. Not per run: the bus is shared by every run.

    Never a 5xx when Redis is down or the group does not exist yet: the panel says
    "not readable" and the rest of the page, which reads Postgres, carries on."""
    stream = SETTINGS.frames_stream
    try:
        client = redis.from_url(SETTINGS.redis_url, decode_responses=False)
        g = group_health(client, stream_name=stream, group="workers")
        consumers = consumer_health(client, stream_name=stream, group="workers")
        stuck = stuck_entries(client, stream_name=stream, group="workers",
                              min_idle_ms=max(0, stuck_ms), count=10)
    except (redis.RedisError, LookupError) as exc:
        return {"readable": False, "error": str(exc), "stream": stream, "group": "workers"}
    return {"readable": True, "stream": g.stream, "group": g.group,
            "stream_length": g.stream_length, "consumers": g.consumers,
            "pending": g.pending,
            # null, never 0: nil lag means Redis cannot compute the backlog, which is a
            # different thing from being caught up (bus/observe.py).
            "lag": g.lag,
            "consumer_rows": [{"name": c.name, "pending": c.pending, "idle_ms": c.idle_ms}
                              for c in consumers],
            "stuck_ms": max(0, stuck_ms),
            "stuck": [{"entry_id": s.entry_id, "consumer": s.consumer, "idle_ms": s.idle_ms,
                       "deliveries": s.delivery_count} for s in stuck]}


@app.get("/api/runs/{run_id}/position")
def position(run_id: str, trail: int = 150) -> dict:
    """Car marker + recent track. The car comes from the bus when the run is live, and
    from the last frame that landed in Postgres otherwise; the trail is always landed
    frames, so the gap between car and trail is the pipeline's lag, visibly."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT seq, lat, lon, heading_deg::float8 AS heading_deg,
                   speed_mps::float8 AS speed_mps, captured_at
            FROM frames WHERE run_id = %s AND lat IS NOT NULL
            ORDER BY seq DESC LIMIT %s""", (run_id, trail))
        landed = rows(cur)
    # Whichever is further along: the bus's newest frame, or the newest landed one. The
    # workers delete frames from the bus once processed, so a drained bus (paused, or
    # caught up) would otherwise snap the car back to an older landed frame.
    bus = _latest_published(run_id)
    last = landed[0] | {"source": "landed"} if landed else None
    car = max((c for c in (bus, last) if c), key=lambda c: c["seq"], default=None)
    return {"car": car, "landed_seq": landed[0]["seq"] if landed else None,
            "feed_state": _feed_state(run_id),
            "trail": [[r["lat"], r["lon"]] for r in reversed(landed)]}


@app.get("/api/runs/{run_id}/route")
def planned_route(run_id: str) -> dict | None:
    """The drive feed-sim planned for this run (written before its first frame), so the
    map can show the road still ahead of the car. None for runs without a route."""
    with db() as conn, conn.cursor() as cur:
        run = _run(cur, run_id)
    return (run["config"] or {}).get("planned_route")


@app.get("/api/runs/{run_id}/track")
def track(run_id: str, every: int = 3) -> dict:
    """The driven path so far, thinned to every `every`-th frame, split into stretches that
    were on a council segment (scored) or not (state roads, turns: never scored). Uses the
    segmenter's own snap, so a dashed stretch on the map is exactly what the score ignored."""
    with db() as conn, conn.cursor() as cur:
        run = _run(cur, run_id)
        cur.execute("SELECT f.seq, f.lat, f.lon, s.segment_id IS NOT NULL AS council "
                    "FROM frames f " + SNAP_LATERAL +
                    " WHERE f.run_id = %(run)s AND f.lat IS NOT NULL "
                    "AND f.seq %% %(every)s = 0 ORDER BY f.seq",
                    {"authority": run["authority_id"], "snap": SNAP_M,
                     "tol": HEADING_TOL_DEG, "run": run_id, "every": max(1, every)})
        pts = cur.fetchall()
    stretches: list[dict] = []
    for seq, lat, lon, council in pts:
        if stretches and stretches[-1]["council"] == council and seq - stretches[-1]["last"] <= every * 2:
            stretches[-1]["points"].append([lat, lon])
            stretches[-1]["last"] = seq
        else:
            # Start each stretch at the previous one's last point so the line has no gaps.
            head = [stretches[-1]["points"][-1]] if stretches and seq - stretches[-1]["last"] <= every * 2 else []
            stretches.append({"council": council, "first": seq, "last": seq,
                              "points": head + [[lat, lon]]})
    return {"stretches": stretches}


@app.get("/api/runs/{run_id}/log")
def frame_log(run_id: str, after_id: int | None = None, limit: int = 200) -> dict:
    """Frames in the order they LANDED (inference_id is the writer's insert order), with
    what the worker found on each. With `after_id`: the oldest `limit` after the cursor,
    so a viewer paging forward never skips a frame. Without: the newest `limit`."""
    limit = min(limit, 1000)
    forward = after_id is not None
    with db() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT i.inference_id, i.seq, i.worker_id, i.status, i.error,
                   i.latency_ms::float8 AS latency_ms, i.started_at, f.captured_at,
                   f.lat, f.lon, f.speed_mps::float8 AS speed_mps, f.source_ref,
                   coalesce((SELECT json_agg(json_build_object(
                               'c', d.defect_class, 'p', round(d.confidence, 2))
                               ORDER BY d.confidence DESC)
                             FROM detections d WHERE d.inference_id = i.inference_id),
                            '[]'::json) AS detections
            FROM inferences i
            JOIN frames f ON f.run_id = i.run_id AND f.seq = i.seq
                         AND f.captured_at = i.captured_at
            WHERE i.run_id = %(run)s
              AND (%(after)s::bigint IS NULL OR i.inference_id > %(after)s)
            ORDER BY i.inference_id {"ASC" if forward else "DESC"} LIMIT %(limit)s""",
            {"run": run_id, "after": after_id, "limit": limit})
        lines = rows(cur)
    if not forward:
        lines.reverse()
    for ln in lines:
        ln["source_ref"] = ln["source_ref"].rsplit("/", 1)[-1]
    return {"lines": lines, "cursor": lines[-1]["inference_id"] if lines else after_id}


app.mount("/static", StaticFiles(directory=STATIC, html=True), name="static")
