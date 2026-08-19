"""Export the frozen and provisional contracts as JSON Schema for non-Python consumers.

Contracts 1 and 2 (FRAME_ENVELOPE, INFERENCE_RESULT) are exported FROM the
dataclasses' own behaviour: every field, required/optional split and
dependency below was checked against `src/edgecv/contracts/frame.py` and
`src/edgecv/contracts/detection.py` (and their tests), not copied from an
older draft. See docs/CONTRACTS.md for the mismatches that check turned up
against an earlier version of this file.

Contracts 3-6 are PROVISIONAL: their shape is derived from the real schema
(src/edgecv/db/migrations/001_initial.sql) and the one read query that exists
today (src/edgecv/dashboard/app.py), but the named owner has not signed off
on the semantics yet. Each carries "x-status" and "x-owner" so the status
travels with the file, not just with docs/CONTRACTS.md.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("src/edgecv/contracts/schemas")

# ---------------------------------------------------------------------------
# CONTRACT 1 — FrameEnvelope (src/edgecv/contracts/frame.py)
# ---------------------------------------------------------------------------
FRAME_ENVELOPE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FrameEnvelope",
    "description": "CONTRACT 1 — published by feed-sim onto the frames stream. "
                    "FROZEN 2026-08-19.",
    "type": "object",
    "required": ["run_id", "seq", "captured_at", "width", "height",
                 "source_ref", "sha256", "transport"],
    "properties": {
        "run_id": {"type": "string", "format": "uuid"},
        "seq": {"type": "integer", "minimum": 0},
        "captured_at": {"type": "string", "format": "date-time",
                         "description": "Absolute time, timezone-aware. "
                                        "GPS-derived UTC preferred when a fix exists."},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "source_ref": {"type": "string"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "transport": {"enum": ["inline", "reference"]},
        "payload": {"type": "string", "contentEncoding": "base64",
                    "description": "Required when transport=inline, absent otherwise."},
        "path": {"type": "string",
                 "description": "Required when transport=reference, absent otherwise."},
        # Position/clock fields: optional everywhere. Omitted from the wire
        # entirely when None (see FrameEnvelope.to_fields) — a Redis stream
        # field that is absent, not a field present with an empty value.
        "lat": {"type": ["number", "null"]},
        "lon": {"type": ["number", "null"]},
        "heading_deg": {"type": ["number", "null"]},
        "speed_mps": {"type": ["number", "null"]},
        "gps_accuracy_m": {"type": ["number", "null"]},
        "capture_mono_ns": {
            "type": ["integer", "null"],
            "description": "Boot-relative. Comparable only within one "
                            "(device_boot_id, run_id). Never an absolute time.",
        },
        "device_boot_id": {
            "type": ["string", "null"], "format": "uuid",
            "description": "Required when capture_mono_ns is present.",
        },
    },
    # FrameEnvelope.__post_init__ enforces both of these; the schema mirrors
    # them so a non-Python producer/consumer can validate without the class.
    "dependentRequired": {
        "capture_mono_ns": ["device_boot_id"],
        "lat": ["lon"],
        "lon": ["lat"],
    },
    "if": {"properties": {"transport": {"const": "inline"}}},
    "then": {"required": ["payload"]},
    "else": {"required": ["path"]},
}

BBOX = {
    "type": "object",
    "required": ["x", "y", "w", "h"],
    "properties": {
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "w": {"type": "integer", "minimum": 1},
        "h": {"type": "integer", "minimum": 1},
    },
}

# ---------------------------------------------------------------------------
# CONTRACT 2 — InferenceResult (src/edgecv/contracts/detection.py)
# ---------------------------------------------------------------------------
# NOTE on DetectorInfo.params_hash: it is a computed @property, not a
# dataclass field, and DetectorInfo.as_dict() — the only thing that reaches
# the wire, via InferenceResult.to_json() — does not include it. It is
# derived (sha256 of the canonical, sort_keys, no-space JSON of `params`,
# truncated to 16 hex chars) and recomputed independently by the metadata
# writer (src/edgecv/db/repository.py::_upsert_detector) after it deserialises
# `params`. Decision: params_hash is deliberately OMITTED from this schema.
# It is not part of the wire contract; documenting it as a property here
# would imply a producer must send it, which none do and none should — any
# consumer that needs it recomputes it from `params` with the formula above.
INFERENCE_RESULT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "InferenceResult",
    "description": "CONTRACT 2 — emitted by workers, persisted by metadata-writer. "
                    "FROZEN 2026-08-19.",
    "type": "object",
    "required": ["run_id", "seq", "captured_at", "width", "height", "source_ref",
                 "frame_sha256", "detector", "worker_id", "started_at",
                 "latency_ms", "status", "error", "detections",
                 "snippet_sha256s", "thumbnail_sha256"],
    "properties": {
        "run_id": {"type": "string", "format": "uuid"},
        "seq": {"type": "integer", "minimum": 0},
        "captured_at": {"type": "string", "format": "date-time"},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "source_ref": {"type": "string"},
        "frame_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "detector": {
            "type": "object",
            "required": ["name", "version", "params"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "params": {"type": "object"},
            },
            "description": "params_hash is NOT part of the wire contract — see "
                            "module docstring and docs/CONTRACTS.md.",
        },
        "worker_id": {"type": "string"},
        "started_at": {"type": "string", "format": "date-time"},
        "latency_ms": {"type": "number", "minimum": 0},
        "status": {"enum": ["ok", "failed", "skipped"]},
        "error": {"type": ["string", "null"]},
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["defect_class", "confidence", "bbox", "severity"],
                "properties": {
                    "defect_class": {"type": "string"},
                    "confidence": {"type": "number"},
                    "bbox": BBOX,
                    "severity": {"enum": ["minor", "major", "critical"]},
                },
            },
        },
        "snippet_sha256s": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "description": "Parallel to detections — same length, same order.",
        },
        "thumbnail_sha256": {"type": ["string", "null"]},
        # Copied off FrameEnvelope by the worker so the writer can persist
        # them (see repository.py::write_results, INSERT INTO frames). Not
        # required: InferenceResult.from_json() reads these with dict.get(),
        # so a producer may omit them entirely.
        "lat": {"type": ["number", "null"]},
        "lon": {"type": ["number", "null"]},
        "heading_deg": {"type": ["number", "null"]},
        "speed_mps": {"type": ["number", "null"]},
        "gps_accuracy_m": {"type": ["number", "null"]},
        "capture_mono_ns": {"type": ["integer", "null"]},
        "device_boot_id": {"type": ["string", "null"], "format": "uuid"},
    },
}

# ---------------------------------------------------------------------------
# CONTRACT 3 — Read API (Joseph). Shape derived from the one read query that
# exists today (src/edgecv/dashboard/app.py::run_options/coverage_series),
# which is exactly the query the read API is meant to replace behind an HTTP
# boundary (see the dashboard module docstring).
# ---------------------------------------------------------------------------
READ_API = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ReadAPI",
    "description": "CONTRACT 3 — DB -> Joseph's read API. PROVISIONAL.",
    "x-status": "PROVISIONAL",
    "x-owner": "Joseph",
    "type": "object",
    "properties": {
        "run_summary": {
            "type": "object",
            "description": "One row of survey_runs, shaped as run_options() "
                            "in dashboard/app.py already reads it.",
            "required": ["run_id", "authority_id", "started_at", "target_fps"],
            "properties": {
                "run_id": {"type": "string", "format": "uuid"},
                "authority_id": {"type": "string"},
                "started_at": {"type": "string", "format": "date-time"},
                "target_fps": {"type": "number"},
            },
        },
        "coverage_point": {
            "type": "object",
            "description": "One row of the run_coverage_1min view, shaped as "
                            "coverage_series() in dashboard/app.py already reads it.",
            "required": ["bucket", "frames_ingested", "frames_processed",
                         "frames_flagged", "frames_without_fix"],
            "properties": {
                "bucket": {"type": "string", "format": "date-time"},
                "frames_ingested": {"type": "integer", "minimum": 0},
                "frames_processed": {"type": "integer", "minimum": 0},
                "frames_flagged": {"type": "integer", "minimum": 0},
                "frames_without_fix": {"type": "integer", "minimum": 0},
            },
        },
    },
    "x-notes": "Only the two shapes the dashboard already queries are shaped "
               "here. The map, worst-N segments and review-queue endpoints "
               "(dashboard/app.py's stated Week 9 replacement scope) still "
               "need Joseph's design before they can be frozen; segments and "
               "instance_reviews table shapes exist in 001_initial.sql for "
               "when that happens.",
}

# ---------------------------------------------------------------------------
# CONTRACT 4 — bench_runs + grid config (Ilana). Shape derived directly from
# the bench_runs table (001_initial.sql). Grid axes are explicitly open.
# ---------------------------------------------------------------------------
BENCH_RUN = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BenchRun",
    "description": "CONTRACT 4 — one row of bench_runs. PROVISIONAL.",
    "x-status": "PROVISIONAL",
    "x-owner": "Ilana",
    "type": "object",
    "required": ["bench_run_id"],
    "properties": {
        "bench_run_id": {"type": "string", "format": "uuid"},
        "label": {"type": ["string", "null"]},
        "grid_point": {
            "type": "object",
            "description": "Free-form today (jsonb DEFAULT '{}'). Grid axes "
                            "are not yet decided — do not assume any key here "
                            "survives.",
        },
        "frames_offered": {"type": ["integer", "null"], "minimum": 0},
        "frames_processed": {"type": ["integer", "null"], "minimum": 0},
        "frames_dropped": {"type": ["integer", "null"], "minimum": 0},
        "p50_ms": {"type": ["number", "null"]},
        "p95_ms": {"type": ["number", "null"]},
        "p99_ms": {"type": ["number", "null"]},
        "precision": {"type": ["number", "null"]},
        "recall": {"type": ["number", "null"]},
        "f1": {"type": ["number", "null"]},
        "bytes_stored": {"type": ["integer", "null"], "minimum": 0},
        "peak_rss_mb": {"type": ["number", "null"]},
        "cpu_pct": {"type": ["number", "null"]},
        "started_at": {"type": ["string", "null"], "format": "date-time"},
        "ended_at": {"type": ["string", "null"], "format": "date-time"},
    },
    "x-notes": "survey_runs.bench_run_id references this table, so a run can "
               "be tied to a benchmark grid point, but the grid axes "
               "(what varies between bench_run rows) are Ilana's to define.",
}

# ---------------------------------------------------------------------------
# CONTRACT 5 — Capture manifest (Dexter -> Ryan). No table exists yet: this
# is Dexter's device-side manifest before frames reach the pipeline. Shape
# derived from FrameEnvelope's own fields (the manifest is what lets Ryan's
# ingest path build a FrameEnvelope later) plus the concrete requirements in
# the "Inbound changes" section of docs/CONTRACTS.md. Deliberately a JSON
# example, not a dataclass — the shape will move once Dexter engages.
# ---------------------------------------------------------------------------
CAPTURE_MANIFEST = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CaptureManifest",
    "description": "CONTRACT 5 — Dexter's device -> Ryan's ingest path. PROVISIONAL.",
    "x-status": "PROVISIONAL",
    "x-owner": "Dexter",
    "type": "object",
    "required": ["run_id", "seq", "captured_at", "capture_mono_ns",
                 "device_boot_id", "sha256"],
    "properties": {
        "run_id": {"type": "string", "format": "uuid"},
        "seq": {"type": "integer", "minimum": 0},
        "captured_at": {
            "type": "string", "format": "date-time",
            "description": "GPS-derived UTC where a fix exists at capture "
                            "time; best-effort device clock otherwise.",
        },
        "capture_mono_ns": {
            "type": "integer",
            "description": "Required (not optional, unlike the frame "
                            "envelope) — the manifest is device-authored, so "
                            "the boot-relative clock always exists at capture time.",
        },
        "device_boot_id": {"type": "string", "format": "uuid"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "lat": {"type": ["number", "null"]},
        "lon": {"type": ["number", "null"]},
        "heading_deg": {"type": ["number", "null"]},
        "speed_mps": {"type": ["number", "null"]},
        "gps_accuracy_m": {"type": ["number", "null"]},
    },
    "dependentRequired": {"lat": ["lon"], "lon": ["lat"]},
    "x-notes": "spool_seq is DEFERRED, not forgotten — it depends on an "
               "unanswered question about whether inference is deferred to a "
               "laptop, so it is deliberately absent from this schema rather "
               "than guessed at. The device spool itself is SQLite/WAL + tus "
               "resumable upload, not Redis, so this manifest is not a "
               "FrameEnvelope and is not published on the frames stream "
               "directly by the device.",
}

# ---------------------------------------------------------------------------
# CONTRACT 6 — Segments GeoJSON + condition index (Ryan shape / Ilana index).
# Shape derived from segments + segment_condition (001_initial.sql).
# ---------------------------------------------------------------------------
SEGMENT_FEATURE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SegmentFeature",
    "description": "CONTRACT 6 — one GeoJSON Feature: a road segment plus its "
                    "latest condition score. PROVISIONAL.",
    "x-status": "PROVISIONAL",
    "x-owner": "Ryan (shape) / Ilana (index)",
    "type": "object",
    "required": ["type", "geometry", "properties"],
    "properties": {
        "type": {"const": "Feature"},
        "geometry": {
            "type": "object",
            "required": ["type", "coordinates"],
            "properties": {
                "type": {"const": "LineString"},
                "coordinates": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[longitude, latitude] — RFC 7946 "
                                       "order, X before Y. Same order as "
                                       "PostGIS ST_MakePoint(lon, lat), and "
                                       "the same trap: swap them and every "
                                       "point lands in the wrong hemisphere "
                                       "without erroring.",
                    },
                },
            },
        },
        "properties": {
            "type": "object",
            "required": ["segment_id", "authority_id", "road_name", "length_m"],
            "properties": {
                "segment_id": {"type": "integer"},
                "authority_id": {"type": "string"},
                "road_name": {"type": "string"},
                "road_ref": {"type": ["string", "null"]},
                "length_m": {"type": "number"},
                "surface_type": {"type": "string"},
                "run_id": {"type": ["string", "null"], "format": "uuid"},
                "assessed_at": {"type": ["string", "null"], "format": "date-time"},
                "condition_index": {
                    "type": ["number", "null"],
                    "description": "Definition is Ilana's to make — this is "
                                    "the slot it lands in, not a formula.",
                },
                "condition_band": {"type": ["string", "null"]},
                "counts": {
                    "type": "object",
                    "description": "Per-class instance counts. jsonb, DEFAULT '{}'.",
                },
                "frames_assessed": {"type": ["integer", "null"]},
                "coverage_m": {"type": ["number", "null"]},
            },
        },
    },
    "x-notes": "One score per (segment_id, run_id), so re-surveying ADDS a "
               "row rather than overwriting one — a consumer that wants "
               "'current condition' must pick the latest run_id per segment "
               "itself; this schema does not do that filtering for you.",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schemas = (
        ("frame_envelope", FRAME_ENVELOPE),
        ("inference_result", INFERENCE_RESULT),
        ("read_api", READ_API),
        ("bench_run", BENCH_RUN),
        ("capture_manifest", CAPTURE_MANIFEST),
        ("segment_feature", SEGMENT_FEATURE),
    )
    for name, schema in schemas:
        path = OUT_DIR / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
