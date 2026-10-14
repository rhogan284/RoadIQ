"""Accuracy + latency for one run against RDD2022 ground truth → one `bench_runs` row.

    python -m edgecv.bench.evaluate --load-gt data/rdd2022/ground_truth.json [--run-id R]

Box-level evaluation (spec §7, level one): per class, greedy matching by descending
confidence at IoU ≥ 0.5, one ground-truth box per prediction. Ground truth joins on
`source_ref`, not frame, so the same image resolves to the same GT in every run.
A class is scored only if the run's images carry ground truth for it; any other class is
reported as unscored rather than as all false positives. On the dronefreak mirror that
means D40 (pothole) is unscored — see scripts/fetch_rdd2022.py for why. `other` is never
scored. An image with no GT rows for a scored class makes that class's predictions on it
false positives, which is the point of replaying clean images.
"""
from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from edgecv.bench.collect import latest_run_id
from edgecv.blobstore.store import BlobStore
from edgecv.config import Settings

IOU_MIN = 0.5
CLASSES = ("D00", "D10", "D20", "D40")


def load_ground_truth(conn: psycopg.Connection, path: Path) -> int:
    rows = [(t["source_ref"], b["defect_class"], b["x"], b["y"], b["w"], b["h"])
            for t in json.loads(Path(path).read_text()) for b in t["boxes"]]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO ground_truth (source_ref, defect_class, bbox_x, bbox_y, bbox_w, bbox_h) "
            "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", rows)
    return len(rows)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def evaluate(conn: psycopg.Connection, run_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT f.source_ref FROM frames f WHERE f.run_id = %s", (run_id,))
        refs = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT source_ref, defect_class, bbox_x, bbox_y, bbox_w, bbox_h "
                    "FROM ground_truth WHERE source_ref = ANY(%s)", (refs,))
        gt: dict[tuple, list] = defaultdict(list)
        for ref, cls, *box in cur.fetchall():
            gt[(ref, cls)].append(box)
        # One inference per frame per detector; a source image replayed twice would be
        # scored twice, which is right — it was predicted twice.
        cur.execute("""
            SELECT f.source_ref, f.seq, d.defect_class, d.confidence::float8,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
            FROM inferences i
            JOIN frames f ON f.run_id = i.run_id AND f.seq = i.seq AND f.captured_at = i.captured_at
            JOIN detections d ON d.inference_id = i.inference_id
            WHERE i.run_id = %s AND d.defect_class <> 'other'""", (run_id,))
        preds = cur.fetchall()
        cur.execute("""
            SELECT count(*),
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms),
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),
                   percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms),
                   min(started_at), max(started_at)
            FROM inferences WHERE run_id = %s AND status = 'ok'""", (run_id,))
        n_ok, p50, p95, p99, t0, t1 = cur.fetchone()
        cur.execute("SELECT config, target_fps::float8 FROM survey_runs WHERE run_id = %s",
                    (run_id,))
        config, fps = cur.fetchone()
        cur.execute("SELECT DISTINCT d.name, d.version, d.params FROM inferences i "
                    "JOIN detectors d USING (detector_id) WHERE i.run_id = %s", (run_id,))
        detectors = [{"name": n, "version": v, "weights": (p or {}).get("weights")}
                     for n, v, p in cur.fetchall()]
    # From the store, not snippets.bytes (a 0 placeholder — repository.py, I2).
    bytes_stored = BlobStore(root=Settings.from_env().blob_root).total_bytes()

    per_class, unscored = {}, {}
    tp_all = fp_all = fn_all = 0
    have_gt = {c for (_ref, c) in gt}
    for cls in CLASSES:
        if cls not in have_gt:
            unscored[cls] = {"pred": sum(1 for p in preds if p[2] == cls),
                             "reason": "no ground truth for this class in the replayed images"}
            continue
        cls_preds = sorted((p for p in preds if p[2] == cls), key=lambda p: -p[3])
        used: dict[tuple, set] = defaultdict(set)
        tp = 0
        for ref, seq, _c, _conf, *box in cls_preds:
            cands = gt.get((ref, cls), [])
            best, best_i = 0.0, None
            for i, g in enumerate(cands):
                if i in used[(ref, seq)]:
                    continue
                iou = _iou(box, g)
                if iou > best:
                    best, best_i = iou, i
            if best_i is not None and best >= IOU_MIN:
                used[(ref, seq)].add(best_i)
                tp += 1
        n_gt = sum(len(v) for (ref, c), v in gt.items() if c == cls)
        fp, fn = len(cls_preds) - tp, n_gt - tp
        per_class[cls] = _prf(tp, fp, fn) | {"gt": n_gt, "pred": len(cls_preds)}
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    overall = _prf(tp_all, fp_all, fn_all)
    return {"run_id": run_id, "detectors": detectors, "frames_processed": n_ok,
            "frames_offered": config.get("frames_offered"),
            "frames_dropped": config.get("frames_dropped"), "target_fps": fps,
            "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
            "throughput_fps": round(n_ok / (t1 - t0).total_seconds(), 2)
            if n_ok and t1 > t0 else None,
            "bytes_stored": int(bytes_stored), "raw_bytes": config.get("raw_bytes_offered"),
            "started_at": t0, "ended_at": t1, "overall": overall, "per_class": per_class,
            "unscored": unscored}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4)}


def record(conn: psycopg.Connection, report: dict) -> str:
    bench_id = str(uuid.uuid4())
    o = report["overall"]
    grid = {"iou_min": IOU_MIN, "detectors": report["detectors"],
            "target_fps": report["target_fps"], "per_class": report["per_class"],
            "unscored": report["unscored"],
            "throughput_fps": report["throughput_fps"], "raw_bytes": report["raw_bytes"]}
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bench_runs (bench_run_id, label, grid_point, frames_offered, "
            "frames_processed, frames_dropped, p50_ms, p95_ms, p99_ms, precision, recall, f1, "
            "bytes_stored, started_at, ended_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (bench_id, f"e2e {report['run_id'][:8]}", json.dumps(grid),
             report["frames_offered"], report["frames_processed"], report["frames_dropped"],
             report["p50_ms"], report["p95_ms"], report["p99_ms"], o["precision"],
             o["recall"], o["f1"], report["bytes_stored"], report["started_at"],
             report["ended_at"] or datetime.now(timezone.utc)))
        cur.execute("UPDATE survey_runs SET bench_run_id = %s WHERE run_id = %s",
                    (bench_id, report["run_id"]))
    return bench_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a run against ground truth")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--load-gt", type=Path, default=None)
    args = ap.parse_args()
    with psycopg.connect(Settings.from_env().pg_dsn, autocommit=True) as conn:
        if args.load_gt:
            print(f"ground truth rows offered: {load_ground_truth(conn, args.load_gt)}")
        run_id = args.run_id or latest_run_id(conn)
        report = evaluate(conn, run_id)
        bench_id = record(conn, report)
        o = report["overall"]
        print(f"run {run_id} · bench {bench_id}\n"
              f"  detector   {report['detectors']}\n"
              f"  frames     {report['frames_processed']} processed / "
              f"{report['frames_offered']} offered, {report['frames_dropped']} dropped\n"
              f"  latency    p50 {report['p50_ms']:.0f} ms · p95 {report['p95_ms']:.0f} ms · "
              f"p99 {report['p99_ms']:.0f} ms · {report['throughput_fps']} fps overall\n"
              f"  box-level  P {o['precision']:.3f} · R {o['recall']:.3f} · F1 {o['f1']:.3f} "
              f"(IoU≥{IOU_MIN})")
        for cls, m in report["per_class"].items():
            print(f"    {cls}  P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
                  f"(gt {m['gt']}, pred {m['pred']})")
        for cls, m in report["unscored"].items():
            print(f"    {cls}  unscored — {m['reason']} ({m['pred']} predictions)")


if __name__ == "__main__":
    main()
