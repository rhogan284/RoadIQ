"""Component 7 — the read API (Contract 3), and the host for the dashboard (component 8).

    uvicorn edgecv.api.main:app --host 0.0.0.0 --port 8000

Read-only against Postgres except one write path: human review decisions, which go to
`instance_reviews` (never to the derived `defect_instances`) and then trigger a re-score of
the run, so a Confirm or Reject changes the segment's condition on the next map refresh.

The API is the only thing the browser talks to: frames and crops are served from here, so
the dashboard never needs a path into the blob store or the dataset.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from edgecv.bench.collect import run_coverage
from edgecv.blobstore.store import BlobStore
from edgecv.config import Settings
from edgecv.segmenter.index import AUTO_ACCEPT_CONF
from edgecv.segmenter.main import segment_run
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
    offered = cfg.get("frames_offered") or ingested
    # From the store itself, as the bytes-per-km bench does: `snippets.bytes` is a 0
    # placeholder the writer cannot fill (repository.py, I2).
    stored = BlobStore(root=SETTINGS.blob_root).total_bytes()
    return {
        "run": {k: run[k] for k in ("run_id", "authority_id", "started_at", "ended_at",
                                    "source_kind", "target_fps")},
        "assessed_km": round(cov.assessed_m / 1000, 2),
        "gap_m": round(cov.gap_m, 1),
        "defects": defects, "pending_review": pending,
        "frames_offered": offered, "frames_ingested": ingested,
        "frames_processed": processed,
        "frames_dropped": cfg.get("frames_dropped", 0),
        "frames_accounted_pct": round(100 * processed / offered, 1) if offered else 0.0,
        # The blob store is shared across runs (snippets carry no run linkage), so this is
        # the store's total — exact for a single-run demo database, an upper bound otherwise.
        "bytes_stored": int(stored), "raw_bytes": cfg.get("raw_bytes_offered"),
        "segments_scored": scored, "segment_coverage_km": round(float(seg_cov) / 1000, 2),
    }


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


app.mount("/static", StaticFiles(directory=STATIC, html=True), name="static")
