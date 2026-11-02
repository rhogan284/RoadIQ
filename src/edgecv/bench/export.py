"""Export one run's benchmark data to plain files, so charts never need the database.

    python -m edgecv.bench.export --run-id R --out bench-out/baseline

Writes, into --out:

  frames.csv      one row per processed frame: the latency chain, split by stage
  pr_curve.csv    precision/recall at every confidence threshold, per class
  confusion.csv   ground-truth class x predicted class, IoU >= 0.5
  summary.json    run config, coverage, stage percentiles, accuracy, bytes, AP50

The latency chain, and what each stage measures:

  captured_at ──queue──▶ started_at ──inference──▶ +latency_ms ──write──▶ written_at

  queue      feed-sim built and published the frame → a worker started on it.
             Time spent waiting on the Redis stream. Rises when workers fall behind.
  inference  decode + YOLO + crops + thumbnail, measured inside the worker.
  write      worker finished → the writer's INSERT. Results-stream wait + batching.
  e2e        the sum: camera to queryable row.

`frames.enqueued_at` is NOT used. The writer fills it from the worker's started_at
(repository.py, frames INSERT), so it always equals the start time and any queue
figure taken from it would be zero. feed-sim stamps captured_at immediately before
it publishes, so captured_at is the enqueue time to within a millisecond.

`written_at` comes from migration 003 and is NULL for runs written before it; the
write and e2e stages are then left empty rather than guessed.

All timestamps come from containers on one Docker VM, so they share one clock.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

from edgecv.bench.collect import latest_run_id, run_coverage
from edgecv.bench.evaluate import CLASSES, IOU_MIN, _iou
from edgecv.config import Settings

STAGES = ("queue_ms", "infer_ms", "write_ms", "e2e_ms")
PERCENTILES = (50, 90, 95, 99)

# --- the queries ---------------------------------------------------------------------
# Kept as module constants so the report appendix can quote them exactly.

FRAMES_SQL = """
SELECT f.seq,
       f.source_ref,
       i.worker_id,
       i.status,
       EXTRACT(EPOCH FROM (i.started_at - f.captured_at)) * 1000          AS queue_ms,
       i.latency_ms::float8                                                AS infer_ms,
       EXTRACT(EPOCH FROM (i.written_at - i.started_at)) * 1000
           - i.latency_ms::float8                                          AS write_ms,
       EXTRACT(EPOCH FROM (i.written_at - f.captured_at)) * 1000          AS e2e_ms,
       EXTRACT(EPOCH FROM (f.captured_at - r.started_at))                  AS t_s,
       (SELECT count(*) FROM detections d
         WHERE d.inference_id = i.inference_id)                            AS n_detections
FROM frames f
JOIN inferences i
  ON i.run_id = f.run_id AND i.seq = f.seq AND i.captured_at = f.captured_at
JOIN survey_runs r ON r.run_id = f.run_id
WHERE f.run_id = %s
ORDER BY f.seq
"""

PREDICTIONS_SQL = """
SELECT f.source_ref, f.seq, d.defect_class, d.confidence::float8,
       d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
FROM inferences i
JOIN frames f ON f.run_id = i.run_id AND f.seq = i.seq AND f.captured_at = i.captured_at
JOIN detections d ON d.inference_id = i.inference_id
WHERE i.run_id = %s AND d.defect_class <> 'other'
"""

GROUND_TRUTH_SQL = """
SELECT source_ref, defect_class, bbox_x, bbox_y, bbox_w, bbox_h
FROM ground_truth
WHERE source_ref IN (SELECT DISTINCT source_ref FROM frames WHERE run_id = %s)
"""

THROUGHPUT_SQL = """
SELECT count(*) FILTER (WHERE i.status = 'ok'),
       EXTRACT(EPOCH FROM (max(i.written_at) - min(f.captured_at))),
       EXTRACT(EPOCH FROM (max(f.captured_at) - min(f.captured_at)))
FROM frames f
JOIN inferences i
  ON i.run_id = f.run_id AND i.seq = f.seq AND i.captured_at = f.captured_at
WHERE f.run_id = %s
"""

# (source_ref, seq, class, confidence, x, y, w, h)
Pred = tuple
# (source_ref, class) -> [[x, y, w, h], ...]
GroundTruth = dict


# --- pure functions (unit-tested in tests/unit/test_bench_export.py) -----------------

def match_flags(preds: list[Pred], gt: GroundTruth, cls: str) -> tuple[list[float], list[bool], int]:
    """Greedy match of one class's predictions, highest confidence first — the same
    rule as evaluate.py. Returns (confidences desc, is_tp per prediction, n_gt).

    Greedy-by-confidence means the matches made by the first k predictions are
    exactly the matches you get with the threshold set at the k-th confidence, so a
    prefix sum over these flags IS the precision-recall curve."""
    cls_preds = sorted((p for p in preds if p[2] == cls), key=lambda p: -p[3])
    used: dict[tuple, set] = defaultdict(set)
    confs, flags = [], []
    for ref, seq, _c, conf, *box in cls_preds:
        best, best_i = 0.0, None
        for i, g in enumerate(gt.get((ref, cls), [])):
            if i in used[(ref, seq)]:
                continue
            iou = _iou(box, g)
            if iou > best:
                best, best_i = iou, i
        hit = best_i is not None and best >= IOU_MIN
        if hit:
            used[(ref, seq)].add(best_i)
        confs.append(conf)
        flags.append(hit)
    # Counted once per image, exactly as evaluate.py counts it, so the two agree on
    # recall. Every benchmark run replays each image once, so this is also exact.
    n_gt = sum(len(boxes) for (_ref, c), boxes in gt.items() if c == cls)
    return confs, flags, n_gt


def pr_curve(confs: list[float], flags: list[bool], n_gt: int) -> list[dict]:
    """One point per prediction: the precision and recall with the threshold there."""
    tp = np.cumsum(flags, dtype=float)
    fp = np.cumsum([not f for f in flags], dtype=float)
    out = []
    for k, conf in enumerate(confs):
        out.append({"confidence": conf,
                    "precision": tp[k] / (tp[k] + fp[k]),
                    "recall": tp[k] / n_gt if n_gt else 0.0})
    return out


def average_precision(curve: list[dict]) -> float:
    """All-point interpolated AP (the VOC 2010+ / COCO-style area under the PR curve).
    Precision is made monotone from the right, then integrated over recall."""
    if not curve:
        return 0.0
    recall = np.concatenate([[0.0], [p["recall"] for p in curve], [1.0]])
    precision = np.concatenate([[1.0], [p["precision"] for p in curve], [0.0]])
    # Make precision monotone decreasing, right to left.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def confusion(preds: list[Pred], gt: GroundTruth, *, conf_min: float = 0.0) -> dict:
    """Ground-truth class x predicted class, class-agnostic matching at IoU >= 0.5.

    Per frame, predictions are taken highest confidence first and each claims the
    unused ground-truth box of ANY class it overlaps most. A claimed box counts in
    (gt class, predicted class); an unclaimed box is a miss (gt class, "missed"); a
    prediction that claims nothing is a false alarm ("background", predicted class).
    """
    by_frame: dict[tuple, list] = defaultdict(list)
    for p in preds:
        if p[3] >= conf_min:
            by_frame[(p[0], p[1])].append(p)
    gt_by_ref: dict[str, list] = defaultdict(list)
    for (ref, cls), boxes in gt.items():
        for b in boxes:
            gt_by_ref[ref].append((cls, b))

    counts: dict[tuple, int] = defaultdict(int)
    frames_seen = {key for key in by_frame}
    # Frames with ground truth but no predictions still have misses to count.
    seqs_by_ref: dict[str, set] = defaultdict(set)
    for ref, seq, *_ in preds:
        seqs_by_ref[ref].add(seq)
    for ref in gt_by_ref:
        for seq in seqs_by_ref.get(ref, {None}):
            frames_seen.add((ref, seq))

    for ref, seq in frames_seen:
        truths = gt_by_ref.get(ref, [])
        used: set[int] = set()
        for _r, _s, pcls, _conf, *box in sorted(by_frame.get((ref, seq), []),
                                                 key=lambda p: -p[3]):
            best, best_i = 0.0, None
            for i, (_gcls, g) in enumerate(truths):
                if i in used:
                    continue
                iou = _iou(box, g)
                if iou > best:
                    best, best_i = iou, i
            if best_i is not None and best >= IOU_MIN:
                used.add(best_i)
                counts[(truths[best_i][0], pcls)] += 1
            else:
                counts[("background", pcls)] += 1
        for i, (gcls, _g) in enumerate(truths):
            if i not in used:
                counts[(gcls, "missed")] += 1
    return dict(counts)


def percentiles(values: list[float]) -> dict:
    clean = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if clean.size == 0:
        return {f"p{p}": None for p in PERCENTILES} | {"mean": None, "n": 0}
    return ({f"p{p}": round(float(np.percentile(clean, p)), 1) for p in PERCENTILES}
            | {"mean": round(float(clean.mean()), 1), "max": round(float(clean.max()), 1),
               "n": int(clean.size)})


# --- database → files ------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_run(conn: psycopg.Connection, run_id: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with conn.cursor() as cur:
        cur.execute(FRAMES_SQL, (run_id,))
        cols = [c.name for c in cur.description]
        frames = [dict(zip(cols, (float(v) if hasattr(v, "is_finite") else v for v in row)))
                  for row in cur.fetchall()]
        cur.execute(PREDICTIONS_SQL, (run_id,))
        preds = cur.fetchall()
        cur.execute(GROUND_TRUTH_SQL, (run_id,))
        gt: GroundTruth = defaultdict(list)
        for ref, cls, *box in cur.fetchall():
            gt[(ref, cls)].append(box)
        cur.execute(THROUGHPUT_SQL, (run_id,))
        n_ok, busy_s, feed_s = cur.fetchone()
        cur.execute("SELECT target_fps::float8, config, started_at, ended_at, bench_run_id "
                    "FROM survey_runs WHERE run_id = %s", (run_id,))
        target_fps, config, started_at, ended_at, bench_id = cur.fetchone()
        bench = None
        if bench_id:
            cur.execute("SELECT precision::float8, recall::float8, f1::float8, bytes_stored, "
                        "grid_point FROM bench_runs WHERE bench_run_id = %s", (bench_id,))
            p, r, f1, bytes_stored, grid = cur.fetchone()
            bench = {"bench_run_id": str(bench_id), "precision": p, "recall": r, "f1": f1,
                     "bytes_stored": bytes_stored, "per_class": grid.get("per_class"),
                     "unscored": grid.get("unscored")}

    _write_csv(out / "frames.csv", frames)

    curves, ap = [], {}
    have_gt = {c for (_ref, c) in gt}
    for cls in CLASSES:
        if cls not in have_gt:
            continue
        confs, flags, n_gt = match_flags(preds, gt, cls)
        curve = pr_curve(confs, flags, n_gt)
        ap[cls] = round(average_precision(curve), 4)
        curves += [{"class": cls} | pt for pt in curve]
    _write_csv(out / "pr_curve.csv", curves)

    # `other` is never scored: its predictions are not stored for scoring
    # (PREDICTIONS_SQL) and evaluate.py skips it, so its ground truth would only add
    # misses that no prediction could ever claim.
    matrix = confusion(preds, {k: v for k, v in gt.items() if k[1] != "other"})
    _write_csv(out / "confusion.csv",
               [{"gt_class": g, "pred_class": p, "count": n}
                for (g, p), n in sorted(matrix.items())])

    cov = run_coverage(conn, run_id)
    raw = (config or {}).get("raw_bytes_offered")
    km = cov.assessed_m / 1000 if cov.assessed_m else None
    summary = {
        "run_id": run_id,
        "target_fps": target_fps,
        "frames_offered": (config or {}).get("frames_offered"),
        "frames_dropped": (config or {}).get("frames_dropped"),
        "frames_landed": len(frames),
        "frames_ok": n_ok,
        "achieved_fps": round(n_ok / float(busy_s), 2) if n_ok and busy_s else None,
        "feed_seconds": round(float(feed_s), 1) if feed_s else None,
        "stages": {s: percentiles([f[s] for f in frames]) for s in STAGES},
        "coverage": {"assessed_m": round(cov.assessed_m, 1), "gap_m": round(cov.gap_m, 1),
                     "coverage_pct": round(cov.coverage_pct, 2), "gaps": cov.gap_count,
                     "largest_gap_m": round(cov.largest_gap_m, 1),
                     "missing_frames": cov.missing_frames},
        "bytes": {"raw_offered": raw,
                  "stored": bench["bytes_stored"] if bench else None,
                  "reduction_x": (round(raw / bench["bytes_stored"], 2)
                                  if bench and raw and bench["bytes_stored"] else None),
                  "raw_per_km": round(raw / km) if raw and km else None,
                  "stored_per_km": (round(bench["bytes_stored"] / km)
                                    if bench and km and bench["bytes_stored"] else None)},
        "defect_frame_share": (round(sum(1 for f in frames if f["n_detections"]) / len(frames), 4)
                               if frames else None),
        "accuracy": bench,
        "ap50": ap,
        "map50_scored": round(sum(ap.values()) / len(ap), 4) if ap else None,
        "started_at": started_at.isoformat() if started_at else None,
        "ended_at": ended_at.isoformat() if ended_at else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one run's benchmark data to files")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with psycopg.connect(Settings.from_env().pg_dsn, autocommit=True) as conn:
        run_id = args.run_id or latest_run_id(conn)
        s = export_run(conn, run_id, args.out)
    print(json.dumps({k: s[k] for k in ("run_id", "frames_landed", "achieved_fps")}
                     | {"e2e": s["stages"]["e2e_ms"], "map50": s["map50_scored"]}))


if __name__ == "__main__":
    main()
