# The six contracts

Frozen interfaces. Everyone codes against these from day one; skip that and Week 6 becomes an
integration weekend.

| # | Contract | Direction | Owner | Status |
|---|---|---|---|---|
| 1 | Frame envelope | Ryan -> Shervin | Ryan | FROZEN 2026-08-19 |
| 2 | Detector plugin interface | Ryan <-> Shervin | Ryan | FROZEN 2026-08-19 |
| 3 | Read API | DB + bus -> dashboard | Joseph | PROVISIONAL — implemented (`api/main.py`), awaiting Joseph's sign-off |
| 4 | bench_runs + grid config | Ilana <-> all | Ilana | PROVISIONAL — table exists, grid axes open |
| 5 | Capture manifest | Dexter -> Ryan | Dexter | PROVISIONAL — requirements below |
| 6 | Segments GeoJSON + condition index | Ryan -> Dexter, Joseph | Ryan (shape) / Ilana (index) | PROVISIONAL |

**PROVISIONAL means:** the shape is written down and you can code against it today, but the named
owner has not signed off on the semantics. If you are the owner, review yours and either sign off or
send a change — before Week 5, when code starts depending on it.

Exported JSON Schema for every contract lives under
[`src/edgecv/contracts/schemas/`](../src/edgecv/contracts/schemas/), built by
[`scripts/export_schemas.py`](../scripts/export_schemas.py). For contracts 1 and 2 **the dataclasses
are the source of truth** — `src/edgecv/contracts/frame.py` and `detection.py`. Where this document or
the exported schema disagrees with the dataclass, the dataclass wins; every place that happened during
this pass is called out below rather than silently fixed.

---

## 1. Frame envelope

`FrameEnvelope` in [`frame.py`](../src/edgecv/contracts/frame.py). Published by feed-sim onto the
Redis `frames` stream as flattened byte fields (`to_fields()` / `from_fields()`), one message per
frame.

| Field | Type | Required | Notes |
|---|---|---|---|
| `run_id` | uuid string | yes | |
| `seq` | integer ≥ 0 | yes | |
| `captured_at` | timezone-aware datetime | yes | Absolute time. GPS-derived UTC preferred when a fix exists — immune to handset clock steps. |
| `width`, `height` | integer ≥ 1 | yes | |
| `source_ref` | string | yes | |
| `sha256` | 64-char hex | yes | |
| `transport` | `inline` \| `reference` | yes | Picks which of the next two fields carries the frame. |
| `payload` | bytes | if `transport=inline` | |
| `path` | string | if `transport=reference` | |
| `lat`, `lon` | float | together or not at all | Half a fix is rejected — a lat with no lon is a capture bug, not a partial result. |
| `heading_deg`, `speed_mps`, `gps_accuracy_m` | float | no | |
| `capture_mono_ns` | integer | requires `device_boot_id` | Boot-relative. Comparable only within one `(device_boot_id, run_id)` pair. **Never** store it as an absolute time. |
| `device_boot_id` | uuid string | required alongside `capture_mono_ns` | |

Position and clock fields are optional because the skeleton runs on generated fixtures with no phone
attached. **Optional means absent from the wire, not present-and-empty**: `to_fields()` omits a `None`
field entirely rather than serialising it as an empty string.

Example (inline transport, no position — what the skeleton actually sends today):

```json
{
  "run_id": "11111111-1111-1111-1111-111111111111",
  "seq": 42,
  "captured_at": "2026-08-19T09:15:03.120000+00:00",
  "width": 1280,
  "height": 720,
  "source_ref": "fixtures/defect_003.png",
  "sha256": "a1b2c3...64 hex chars",
  "transport": "inline",
  "payload": "<base64 bytes>"
}
```

Example (reference transport, full position + clock — what a phone will send):

```json
{
  "run_id": "11111111-1111-1111-1111-111111111111",
  "seq": 43,
  "captured_at": "2026-08-19T09:15:03.220000+00:00",
  "width": 1280,
  "height": 720,
  "source_ref": "device/frame_00043.jpg",
  "sha256": "b2c3d4...64 hex chars",
  "transport": "reference",
  "path": "/spool/frame_00043.jpg",
  "lat": -33.8688,
  "lon": 151.2093,
  "heading_deg": 275.5,
  "speed_mps": 16.7,
  "gps_accuracy_m": 4.2,
  "capture_mono_ns": 1234567890123,
  "device_boot_id": "22222222-2222-2222-2222-222222222222"
}
```

**What changing it would break:** feed-sim ([`feedsim/main.py`](../src/edgecv/feedsim/main.py)) is the
only producer today; a new required field breaks it immediately, and a removed or renamed field breaks
every worker's `FrameEnvelope.from_fields()` call
([`worker/main.py`](../src/edgecv/worker/main.py)). Because `worker/main.py` copies the position/clock fields
straight onto `InferenceResult` (contract 2), a change here that isn't mirrored there breaks the
writer's `frames` table insert in
[`db/repository.py`](../src/edgecv/db/repository.py)`::write_results` the moment it runs.

**Mismatch found against the exported schema:** the JSON Schema template this document was built from
predated the frame envelope's position/clock fields entirely — it had none of `lat`, `lon`,
`heading_deg`, `speed_mps`, `gps_accuracy_m`, `capture_mono_ns` or `device_boot_id`. The exported
[`frame_envelope.schema.json`](../src/edgecv/contracts/schemas/frame_envelope.schema.json) adds all
seven, plus two rules the dataclass's `__post_init__` enforces that the template didn't encode at all:
`lat` and `lon` must appear together (`dependentRequired`), and `payload`/`path` is required
conditionally on `transport` (`if`/`then`/`else`). Both were verified against `FrameEnvelope` directly
with the `jsonschema` validator, not just read off the source.

---

## 2. Detector plugin interface

`DetectorInfo`, `Detection`, `BBox`, `InferenceResult` and the `Detector` protocol in
[`detection.py`](../src/edgecv/contracts/detection.py). Shervin's worker code implements `Detector`
(`info` property + `detect(image)`); Ryan's worker loop wraps that output into an `InferenceResult` and
puts it on the Redis `results` stream as JSON (`to_json()` / `from_json()`) for the writer to persist.

| Field | Type | Required | Notes |
|---|---|---|---|
| `run_id`, `seq`, `captured_at`, `width`, `height`, `source_ref` | — | yes | Copied from the frame envelope. |
| `frame_sha256` | 64-char hex | yes | |
| `detector` | `{name, version, params}` | yes | See below — `params_hash` is deliberately **not** here. |
| `worker_id` | string | yes | Must be distinct per replica — see the note under contract 4 / inbound changes. |
| `started_at` | datetime | yes | |
| `latency_ms` | number ≥ 0 | yes | |
| `status` | `ok` \| `failed` \| `skipped` | yes | |
| `error` | string or null | **yes** (nullable) | Always present on the wire, even when null. |
| `detections` | array of `{defect_class, confidence, bbox, severity}` | yes | |
| `snippet_sha256s` | array of 64-char hex | yes | Parallel to `detections` — same length, same order. |
| `thumbnail_sha256` | string or null | **yes** (nullable) | Always present on the wire, even when null. |
| `lat`, `lon`, `heading_deg`, `speed_mps`, `gps_accuracy_m`, `capture_mono_ns`, `device_boot_id` | — | no | Copied off the frame envelope by the worker; absent when the frame envelope didn't carry them. |

Severity is derived from a detection's pixel area, not sent as a free choice: `≤100px² → minor`,
`≤1000px² → major`, else `critical` (`severity_for()`).

Example:

```json
{
  "run_id": "11111111-1111-1111-1111-111111111111",
  "seq": 42,
  "captured_at": "2026-08-19T09:15:03.120000+00:00",
  "width": 1280, "height": 720,
  "source_ref": "fixtures/defect_003.png",
  "frame_sha256": "a1b2c3...64 hex chars",
  "detector": {"name": "threshold", "version": "1.0.0",
               "params": {"k": 3.0, "min_area": 60, "blur_ksize": 5}},
  "worker_id": "worker-283f72b1cddc",
  "started_at": "2026-08-19T09:15:03.400000+00:00",
  "latency_ms": 42.7,
  "status": "ok",
  "error": null,
  "detections": [
    {"defect_class": "pothole", "confidence": 0.87,
     "bbox": {"x": 100, "y": 200, "w": 40, "h": 30}, "severity": "minor"}
  ],
  "snippet_sha256s": ["c3d4e5...64 hex chars"],
  "thumbnail_sha256": null
}
```

**On `DetectorInfo.params_hash`:** it is a computed `@property` (`sha256` of the canonical,
`sort_keys`, no-space JSON encoding of `params`, truncated to 16 hex characters), not a dataclass
field. `DetectorInfo.as_dict()` — the only thing `InferenceResult.to_json()` calls — does not include
it, so it never touches the wire. The writer
([`repository.py`](../src/edgecv/db/repository.py)`::_upsert_detector`) recomputes it independently
after deserialising `params`, because that's the value that has to match the `detectors` table's
`UNIQUE (name, version, params_hash)` constraint. **Decision: `params_hash` is excluded from the
exported schema.** Putting a property that no producer sends and no consumer should expect into a
schema meant to describe the wire format would be actively misleading. Anyone who needs it — e.g. a
non-Python consumer deduplicating detector configs — recomputes it from `params` with the formula
above; that formula is now written down in the schema's `description` and here rather than only living
in one property getter.

**What changing it would break:** the worker loop (constructs `InferenceResult`), any `Detector`
implementation such as [`detectors/yolo12s.py`](../src/edgecv/detectors/yolo12s.py) (must return
`Detection`/`BBox` in this exact shape), and
[`db/repository.py`](../src/edgecv/db/repository.py)`::write_results`, which reads `detector.name`,
`.version`, `.params`, `.params_hash`, every `result.<position field>`, and every
`detection.bbox.x/y/w/h`, `.area`, `.severity` directly — a field rename there is a silent `AttributeError`
at write time, not a schema error caught earlier. Changing how `params_hash` is computed doesn't break
code (nothing on the wire depends on it) but **does** retroactively fork every existing detector row in
`detectors`, because a retuned hash is a different `(name, version, params_hash)` identity by design —
that's what makes the benchmark grid meaningful, and also what makes changing the formula a
data-migration problem, not just a code change.

**Mismatch found against the exported schema:** the template `InferenceResult` schema this was built
from carried none of the position/clock fields the worker actually copies onto every result (see the
table above) — a real gap, not just a stale comment, confirmed by reading
`repository.py::write_results`, which inserts `result.lat`, `result.lon`, etc. straight into the
`frames` table. Separately, `error` and `thumbnail_sha256` were missing from the schema's `required`
list even though `InferenceResult.from_json()` reads both by direct key access (`d["error"]`,
`d["thumbnail_sha256"]`) rather than `.get()` — so a message missing either key raises `KeyError` in
the real deserialiser, meaning the wire format truly requires them present (with `null` allowed), it
just doesn't require them non-null. The position/clock fields, by contrast, genuinely are read with
`.get()` in `from_json()`, so they were correctly left out of `required`. All of this was checked by
round-tripping a real `InferenceResult` through `to_json()`/`from_json()` and validating the JSON
against the schema with `jsonschema`, not by inspection alone.

---

## 3. Read API

**PROVISIONAL — owner Joseph. Implemented** in [`api/main.py`](../src/edgecv/api/main.py)
(FastAPI, `uvicorn edgecv.api.main:app`, port 8000). Its one consumer is the dashboard it serves
at `/` ([`api/static/app.js`](../src/edgecv/api/static/app.js)); the browser talks to nothing else,
so frames and crops are served from here too. Exported schema:
[`read_api.schema.json`](../src/edgecv/contracts/schemas/read_api.schema.json) — the run row, the
run summary and bus health.

| Endpoint | Returns | Read by |
|---|---|---|
| `GET /api/health` | `{"ok": true}` once Postgres answers | compose / demos |
| `GET /api/runs` | run rows, newest first (`ended_at` null = live) | run menu, live follow |
| `GET /api/runs/{id}/summary` | `run_summary` — the stat chips | stat chips, LIVE badge |
| `GET /api/runs/{id}/segments.geojson` | contract 6 FeatureCollection, unsurveyed segments included | map |
| `GET /api/runs/{id}/worklist?limit=25` | worst-first segments + change against the authority's previous run | work list |
| `GET /api/runs/{id}/instances?state=&segment_id=` | defect instances with their effective review state | map dots, review queue |
| `GET /api/instances/{run}/{cluster_key}` | one instance, its best frame and every box on that frame | evidence panel |
| `GET /api/frames/{run}/{seq}/image`, `GET /api/blobs/{crop\|thumbnail}/{sha256}` | pixels (frames only from under `FRAME_ROOT`) | evidence panel |
| `POST /api/instances/{run}/{cluster_key}/review` | writes `instance_reviews`, re-scores the run | Confirm / Reject |
| `GET /api/runs/{id}/bench` | contract 4 row or null | benchmark box |
| `GET /api/runs/{id}/position`, `/route`, `/track`, `/log`; `GET /api/network` | vehicle, planned route, driven legs, frame log, state roads | map, frame log |
| `POST /api/runs/start`, `GET /api/runner`, `POST /api/runs/{id}/control` | queue a run, runner status, pause / resume / cancel | header controls |
| `GET /api/bus?stuck_ms=30000` | `bus_health` | Bus health card |

```json
{
  "run_summary": {
    "run": {"run_id": "11111111-1111-1111-1111-111111111111", "authority_id": "demo-council",
            "started_at": "2026-09-27T07:00:00+00:00", "ended_at": null,
            "source_kind": "dataset-replay", "target_fps": 8.0},
    "assessed_km": 14.2, "gap_m": 31.5, "defects": 412, "pending_review": 37,
    "frames_offered": 5758, "frames_ingested": 5700, "frames_processed": 5700,
    "frames_dropped": 15, "frames_in_flight": 43, "frames_accounted_pct": 99.3,
    "bytes_stored": 48211002, "raw_bytes": 1043338211,
    "segments_scored": 118, "segment_coverage_km": 11.8
  },
  "bus_health": {
    "readable": true, "stream": "frames", "group": "workers",
    "stream_length": 61, "consumers": 2, "pending": 4, "lag": 57,
    "consumer_rows": [{"name": "worker-a1", "pending": 2, "idle_ms": 140},
                      {"name": "worker-b7", "pending": 2, "idle_ms": 95}],
    "stuck_ms": 30000, "stuck": []
  }
}
```

Semantics that are easy to get wrong:

- **`frames_accounted_pct`** is `(ingested + dropped) / offered`. A dropped frame is *accounted for*
  (counted, and reported as `gap_m` of road not assessed). Only frames still in flight are not.
  While a run is live, `offered` and `dropped` come from feed-sim's counters on the bus, because
  `survey_runs.config` is written only when the feed ends.
- **`lag: null`** means Redis cannot compute the backlog (entries ahead of the group were
  deleted). Show "unknown", never 0 — see [`bus/observe.py`](../src/edgecv/bus/observe.py).
- **`/api/bus` is not per run.** The bus is shared, so the card does not follow the run menu. It
  always returns 200; `readable: false` with `error` means Redis is down or the `workers` group
  does not exist yet, and the rest of the page (Postgres) still renders.
- **Review state is derived, never stored on `defect_instances`.** The segmenter replaces those rows
  every pass, so the effective state is `instance_reviews.review_state`, else `auto-accepted` at or
  above `AUTO_ACCEPT_CONF`, else `pending`.

**What changing it would break:** `api/static/app.js` (every panel), and the e2e assertions in
`tests/e2e/test_pipeline.py` (summary, segments, worklist, bench, instances, evidence, review →
re-score) and `tests/e2e/test_worker_failure.py` (`/api/bus` reports a dead worker's stuck
entries). Still PROVISIONAL rather than FROZEN because the owner has not signed off — freezing it is
Joseph's call.

---

## 4. `bench_runs` + grid config

**PROVISIONAL — owner Ilana.** The table exists (`bench_runs` in
[`001_initial.sql`](../src/edgecv/db/migrations/001_initial.sql)); the benchmark grid's axes — what
actually varies from one `bench_run` row to the next — are not decided.

```json
{
  "bench_run_id": "33333333-3333-3333-3333-333333333333",
  "label": "threshold-vs-min-area-grid",
  "grid_point": {},
  "frames_offered": 3000,
  "frames_processed": 2985,
  "frames_dropped": 15,
  "p50_ms": 12.4, "p95_ms": 38.1, "p99_ms": 61.0,
  "precision": 0.81, "recall": 0.74, "f1": 0.77,
  "bytes_stored": 104857600,
  "peak_rss_mb": 512.0, "cpu_pct": 63.5,
  "started_at": "2026-08-19T10:00:00+00:00",
  "ended_at": "2026-08-19T10:20:00+00:00"
}
```

`grid_point` is `jsonb DEFAULT '{}'` — free-form on purpose. Do not assume any key inside it survives
until Ilana signs off; it's the one part of this contract that is genuinely undecided, not just
unreviewed.

**What changing it would break:** `survey_runs.bench_run_id` is a foreign key into `bench_runs`, and
the migration file has this table created *before* `survey_runs` specifically because of that FK — a
schema change here that isn't reflected in a new migration breaks `make migrate` outright for anyone
starting fresh. Once benchmark tooling exists that inserts specific columns (`p50_ms`, `precision`,
etc.), renaming one silently stops that column being written rather than raising, because every insert
here is expected to go through `ON CONFLICT ... DO NOTHING`-style idempotent code, not a strict
all-or-nothing write.

---

## 5. Capture manifest

**PROVISIONAL — owner Dexter.** No table or code path exists for this yet — it's Dexter's device-side
manifest, produced before a frame ever reaches Redis. The shape below is deliberately a JSON example,
not a dataclass, because it depends on the frame envelope's own optional fields and on requirements
that are still open (see "Inbound changes" below); freezing it as code now would freeze the wrong
thing.

```json
{
  "run_id": "11111111-1111-1111-1111-111111111111",
  "seq": 43,
  "captured_at": "2026-08-19T09:15:03.220000+00:00",
  "capture_mono_ns": 1234567890123,
  "device_boot_id": "22222222-2222-2222-2222-222222222222",
  "sha256": "b2c3d4...64 hex chars",
  "lat": -33.8688,
  "lon": 151.2093,
  "heading_deg": 275.5,
  "speed_mps": 16.7,
  "gps_accuracy_m": 4.2
}
```

Unlike the frame envelope, `capture_mono_ns` and `device_boot_id` are **required, not optional** here:
the manifest is device-authored, so the boot-relative clock always exists at capture time even though
it's optional on the wire for the fixture-generated skeleton. `captured_at` should be GPS-derived UTC
whenever a fix exists at capture time, and a best-effort device clock otherwise.

`spool_seq` is **deferred, not forgotten**, and is deliberately absent from this shape rather than
guessed at — see the inbound-changes note for why. The device spool itself is SQLite/WAL plus tus
resumable upload, **not** Redis, so this manifest is not published directly onto the `frames` stream by
the device; something on Ryan's side turns a validated manifest entry into a `FrameEnvelope`.

**What changing it would break:** nothing runs against this yet, so the immediate blast radius is
zero — but the frame envelope's own validation (`lat`/`lon` paired, `capture_mono_ns` requiring
`device_boot_id`) already assumes the manifest supplies both together. If Dexter's device path ever
produces one without the other, `FrameEnvelope.__post_init__` raises `ValueError` and that frame never
reaches the pipeline at all — better to catch that on the device than to discover it at ingest.

---

## 6. Segments GeoJSON + condition index

**PROVISIONAL — shape owner Ryan, condition-index owner Ilana.** One GeoJSON `Feature` per road
segment, combining `segments` and its latest `segment_condition` row
([`001_initial.sql`](../src/edgecv/db/migrations/001_initial.sql)).

```json
{
  "type": "Feature",
  "geometry": {
    "type": "LineString",
    "coordinates": [[151.2090, -33.8685], [151.2098, -33.8691]]
  },
  "properties": {
    "segment_id": 17,
    "authority_id": "council-042",
    "road_name": "Parramatta Road",
    "road_ref": "A44",
    "length_m": 412.6,
    "surface_type": "sealed",
    "run_id": "11111111-1111-1111-1111-111111111111",
    "assessed_at": "2026-08-19T11:00:00+00:00",
    "condition_index": 0.62,
    "condition_band": "fair",
    "counts": {"pothole": 3, "crack": 11},
    "frames_assessed": 412,
    "coverage_m": 398.1
  }
}
```

**Coordinate order — read this before writing any code against contract 6.** GeoJSON coordinates are
`[longitude, latitude]` per RFC 7946: **X before Y**, the opposite of the "lat, lon" order most people
say out loud. This is the *same* order PostGIS's `ST_MakePoint(lon, lat)` uses to build the `geog`
column on `defect_instances` — so it's one trap, not two, but it catches people twice if they don't
know it's the same trap. Swap the order and every point lands in the wrong hemisphere without any
error at all; nothing in the type system catches a valid-looking pair of floats in the wrong order.

`segment_condition` is one row per `(segment_id, run_id)` — re-surveying a segment **adds** a row
rather than overwriting the previous one, so "current condition" means picking the latest `run_id` per
segment yourself; this schema doesn't do that filtering for you, and neither does the table.

**What changing it would break:** `segments_geom_gix` and `defect_instances_geog_gix` are GIST indexes
built over the exact geometry columns this contract exposes — renaming or retyping either column
breaks those indexes' `CREATE INDEX` statements on a fresh migration. `defect_instances.geog` is a
`GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) STORED` column: change which
of `lat`/`lon` feeds which argument and every existing point silently relocates rather than erroring.
`instance_reviews` has no foreign key to `defect_instances` on purpose — it joins on `cluster_key`
instead, specifically so a review outlives a segmenter re-run — so a change to how `cluster_key` is
derived breaks that join for every review already recorded, not just new ones.

---

## Inbound changes for each person

These are real changes each teammate needs, and they currently exist only in a research note nobody
else has read.

- **Shervin:** the frame envelope's position and clock fields are all optional. Your worker needs no
  change to keep working.
- **Dexter:** the capture manifest must carry `capture_mono_ns` and `device_boot_id`, and GPS-derived
  UTC for `captured_at` where a fix exists. `spool_seq` is **deferred, not forgotten** — it depends on
  an unanswered question about whether inference is deferred to a laptop. Your device spool is
  SQLite/WAL + tus resumable upload, **not** Redis.
- **Ilana:** free hand on clustering — idempotency is handled above your layer by
  recompute-and-replace, so DBSCAN, tracking-by-detection or ego-motion IoU are all open. Two things
  are yours: **DBSCAN vs DBSCAN\*** (border-point determinism decides whether the idempotency claim
  holds strictly), and the **condition-index definition**. Your segmenter must recompute-and-replace
  per `(run_id, segment_id)` in one transaction.
- **Joseph:** Compose and CI need **`imresamu/postgis:16-3.5-alpine`**, not `postgis/postgis` — the
  official PostGIS images have no arm64 manifest. Also `pg_partman` and `pg_cron` are **not** in that
  image, so Week 9 partition-retention work needs a custom Dockerfile. And the read API writes
  `instance_reviews`, **never** `defect_instances` — instances are derived and replaced wholesale.
- **Everyone:** the run table is `survey_runs` with `authority_id` (it was `production_runs` /
  `line_id` before the domain pivot). Compose and CI sit in the observability/infrastructure lane;
  `docker-compose.yml` as it stands today is a placeholder pending its owner.
