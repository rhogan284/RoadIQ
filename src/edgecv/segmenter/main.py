"""Segmenter: snap detections to road segments, cluster them into physical defects, and
score every segment the run assessed.

    python -m edgecv.segmenter.main            # always-on: re-segment runs with new results
    python -m edgecv.segmenter.main --once     # every run once, then exit (make e2e)

Idempotent by construction (R-R2): a run's `defect_instances` and `segment_condition` rows
are DERIVED, so each pass deletes and re-inserts all of them for that run in one
transaction. Nothing is merged incrementally, so a redelivered result or a re-run cannot
create a second instance. Human decisions live in `instance_reviews`, keyed by the
deterministic `cluster_key`, so they survive the churn of `instance_id`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import psycopg

from edgecv.config import Settings
from edgecv.db.migrate import apply_migrations
from edgecv.roads import DEFAULT_NETWORK, load_ways, segments
from edgecv.segmenter.index import AUTO_ACCEPT_CONF, score

#: A fix further than this from every segment is left unsnapped, never guessed: a defect
#: placed on the wrong road is worse than one reported as unlocated (spec §6).
SNAP_M = 15.0
#: Along-road merge tolerance for real drives, where consecutive frames overlap (§2a).
DRIVE_MERGE_M = 3.0
#: Consecutive dataset-replay frames are unrelated photos, so merging them would invent
#: physical defects. One detection = one instance there.
REPLAY_MERGE_M = 0.0
MAX_SEQ_GAP = 10


def load_segments(conn: psycopg.Connection, *, authority_id: str,
                  network=DEFAULT_NETWORK) -> int:
    """Insert any segment of the network not already loaded for this authority."""
    with conn.cursor() as cur:
        cur.execute("SELECT road_ref FROM segments WHERE authority_id = %s", (authority_id,))
        have = {r[0] for r in cur.fetchall()}
        new = [s for s in segments(load_ways(network)) if s.road_ref not in have]
        for s in new:
            cur.execute(
                "INSERT INTO segments (authority_id, road_name, road_ref, geom, length_m) "
                "VALUES (%s, %s, %s, ST_GeogFromText(%s), %s)",
                (authority_id, s.road_name, s.road_ref, "SRID=4326;" + s.wkt(), s.length_m))
    return len(new)


#: A frame snaps to a segment only if the vehicle is driving ALONG it: heading within
#: this many degrees of the street's direction (either way). Without it, a frame on
#: Harris St passing a side-street mouth is within SNAP_M of that side street and its
#: defects land on the wrong road — found on the first loop run, 2026-09-27.
HEADING_TOL_DEG = 30.0

# The snap, as a LATERAL subquery over a frame alias `f` (lat, lon, heading_deg). Shared
# with the read API's track endpoint so the map and the scores can never disagree about
# which frames were on a council road. Parameters: %(authority)s, %(snap)s, %(tol)s.
SNAP_LATERAL = """
LEFT JOIN LATERAL (
    SELECT sg.segment_id, loc.frac * sg.length_m AS along_m
    FROM segments sg
    CROSS JOIN LATERAL (
        SELECT ST_LineLocatePoint(sg.geom::geometry,
                                  ST_SetSRID(ST_MakePoint(f.lon, f.lat), 4326)) AS frac
    ) loc
    CROSS JOIN LATERAL (
        -- the street's bearing at the snap point, from two points 5 percent either side
        SELECT degrees(ST_Azimuth(
                   ST_LineInterpolatePoint(sg.geom::geometry, greatest(0, loc.frac - 0.05))::geography,
                   ST_LineInterpolatePoint(sg.geom::geometry, least(1, loc.frac + 0.05))::geography)) AS az
    ) dir
    WHERE sg.authority_id = %(authority)s
      AND ST_DWithin(sg.geom, ST_SetSRID(ST_MakePoint(f.lon, f.lat), 4326)::geography,
                     %(snap)s)
      AND (f.heading_deg IS NULL OR dir.az IS NULL
           OR abs(((dir.az - f.heading_deg + 540)::numeric %% 360) - 180) >= 180 - %(tol)s
           OR abs(((dir.az - f.heading_deg + 540)::numeric %% 360) - 180) <= %(tol)s)
    ORDER BY sg.geom <-> ST_SetSRID(ST_MakePoint(f.lon, f.lat), 4326)::geography
    LIMIT 1
) s ON TRUE
"""

# Frames are read once and joined to detections in Python, not per detection in SQL.
_FRAMES_SQL = """
SELECT f.seq, f.captured_at, f.width, f.height, f.speed_mps,
       s.segment_id, s.along_m, f.lat, f.lon,
       EXISTS (SELECT 1 FROM inferences i WHERE i.run_id = f.run_id AND i.seq = f.seq
                 AND i.captured_at = f.captured_at AND i.status = 'ok') AS processed
FROM frames f
""" + SNAP_LATERAL + """
WHERE f.run_id = %(run)s AND f.lat IS NOT NULL
"""

_DETECTIONS_SQL = """
SELECT d.detection_id, i.seq, d.defect_class, d.confidence::float8, d.severity,
       d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.area_px
FROM inferences i JOIN detections d ON d.inference_id = i.inference_id
WHERE i.run_id = %s
"""


@dataclass(slots=True)
class Obs:
    detection_id: int
    seq: int
    defect_class: str
    confidence: float
    severity: str
    bbox_x: int
    area_px: int
    frame: tuple                    # row from _FRAMES_SQL


def _cluster(obs: list[Obs], merge_m: float) -> list[list[Obs]]:
    """Group same-class detections on one segment that sit within merge_m along the road
    and within MAX_SEQ_GAP frames of each other. merge_m == 0 → one per detection."""
    if merge_m <= 0:
        return [[o] for o in obs]
    obs = sorted(obs, key=lambda o: (o.frame[6], o.seq))
    groups: list[list[Obs]] = []
    for o in obs:
        g = groups[-1] if groups else None
        if g and abs(o.frame[6] - g[-1].frame[6]) <= merge_m and o.seq - g[-1].seq <= MAX_SEQ_GAP:
            g.append(o)
        else:
            groups.append([o])
    return groups


def _cluster_key(run_id: str, defect_class: str, min_seq: int, ordinal: int) -> str:
    return hashlib.md5(f"{run_id}|{defect_class}|{min_seq}|{ordinal}".encode()).hexdigest()


def segment_run(conn: psycopg.Connection, run_id: str) -> dict:
    """Recompute and replace every derived row for one run. Returns a summary."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT authority_id, source_kind, target_fps::float8 FROM survey_runs "
                    "WHERE run_id = %s", (run_id,))
        authority, source_kind, fps = cur.fetchone()
        merge_m = REPLAY_MERGE_M if source_kind == "dataset-replay" else DRIVE_MERGE_M

        cur.execute(_FRAMES_SQL, {"authority": authority, "snap": SNAP_M,
                                  "tol": HEADING_TOL_DEG, "run": run_id})
        frames = {r[0]: r for r in cur.fetchall()}
        cur.execute(_DETECTIONS_SQL, (run_id,))
        per_group: dict[tuple, list[Obs]] = defaultdict(list)
        unlocated = 0
        for (det_id, seq, cls, conf, sev, bx, _by, _bw, _bh, area) in cur.fetchall():
            fr = frames.get(seq)
            if fr is None or fr[5] is None:
                unlocated += 1
                continue
            per_group[(fr[5], cls)].append(Obs(det_id, seq, cls, conf or 0.0, sev, bx, area, fr))

        cur.execute("SELECT cluster_key, review_state, new_class FROM instance_reviews "
                    "WHERE run_id = %s", (run_id,))
        reviews = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

        cur.execute("DELETE FROM defect_instances WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM segment_condition WHERE run_id = %s", (run_id,))

        ordinals: dict[tuple, int] = defaultdict(int)
        area: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        rows = []
        for (segment_id, cls), obs in sorted(per_group.items()):
            for group in _cluster(obs, merge_m):
                group.sort(key=lambda o: (o.seq, o.bbox_x, o.detection_id))
                best = max(group, key=lambda o: (o.confidence, -o.detection_id))
                min_seq = group[0].seq
                # Ordinal disambiguates two same-class boxes first seen on one frame; it is
                # assigned in a deterministic order, so a re-run reproduces the same keys.
                ordinal = ordinals[(cls, min_seq)]
                ordinals[(cls, min_seq)] += 1
                key = _cluster_key(run_id, cls, min_seq, ordinal)
                review_state, new_class = reviews.get(key, (None, None))
                state = review_state or ("auto-accepted" if best.confidence >= AUTO_ACCEPT_CONF
                                         else "pending")
                eff_class = new_class if state == "reclassified" and new_class else cls
                fr = best.frame
                if state in ("auto-accepted", "confirmed", "reclassified") and eff_class != "other":
                    area[segment_id][eff_class] += best.area_px / (fr[2] * fr[3])
                    counts[segment_id][eff_class] += 1
                elif state == "pending":
                    counts[segment_id]["pending"] += 1
                rows.append((key, run_id, segment_id, cls, fr[7], fr[8], fr[6],
                             frames[group[0].seq][1], frames[group[-1].seq][1], len(group),
                             best.confidence, best.severity, best.detection_id))
        if rows:
            cur.executemany(
                "INSERT INTO defect_instances (cluster_key, run_id, segment_id, defect_class, "
                "lat, lon, along_m, first_seen_at, last_seen_at, observation_count, "
                "peak_confidence, severity, best_detection_id) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)

        # Coverage per segment: processed frames on it × the distance one frame advances,
        # capped at the segment's length (a second lap re-surveys, it doesn't lengthen it).
        cur.execute("SELECT segment_id, length_m FROM segments WHERE authority_id = %s",
                    (authority,))
        lengths = dict(cur.fetchall())
        assessed: dict[int, int] = defaultdict(int)
        spacing: dict[int, float] = {}
        for fr in frames.values():
            if fr[5] is not None and fr[9]:
                assessed[fr[5]] += 1
                spacing[fr[5]] = (fr[4] or 0.0) / fps
        now = datetime.now().astimezone()
        cond = []
        for segment_id, n in assessed.items():
            coverage = min(lengths[segment_id], n * spacing[segment_id])
            idx, band = score(area[segment_id], n, coverage)
            cond.append((segment_id, run_id, now, idx, band,
                         json.dumps(dict(counts[segment_id])), n, round(coverage, 1)))
        if cond:
            cur.executemany(
                "INSERT INTO segment_condition (segment_id, run_id, assessed_at, "
                "condition_index, condition_band, counts, frames_assessed, coverage_m) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", cond)
    return {"run_id": run_id, "instances": len(rows), "segments_scored": len(cond),
            "unlocated_detections": unlocated}


def _watermarks(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT run_id::text, max(inference_id) FROM inferences GROUP BY run_id")
        return dict(cur.fetchall())


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmenter (component 11)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--authority-id", default="demo-council")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()
    settings = Settings.from_env()
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        apply_migrations(conn)
        print(f"segments loaded: {load_segments(conn, authority_id=args.authority_id)} new",
              flush=True)
        if args.once:
            runs = [args.run_id] if args.run_id else list(_watermarks(conn))
            for run in runs:
                print(segment_run(conn, run), flush=True)
            return
        seen: dict[str, int] = {}
        while True:
            for run, mark in _watermarks(conn).items():
                if seen.get(run) != mark:
                    print(segment_run(conn, run), flush=True)
                    seen[run] = mark
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
