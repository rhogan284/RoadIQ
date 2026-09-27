# E2E demo — working dashboard, YOLOv12s in the worker, RDD2022 test split end to end

**Date:** 2026-09-27 · **Branch:** `e2e-demo` (from `t1-coverage-gaps-and-lag` @ 72569ff)
**Deadline:** owner meeting 5, Wed 30 Sep 2026, 2pm
**Goal (16 Sep owner meeting):** combine all components and run the dataset through the
whole pipeline end to end.

## 1. What success looks like

`make e2e` on a clean clone (after `make weights dataset`) does all of this with no manual step:

1. Starts the full Compose stack.
2. Replays **every image of the RDD2022 test split (5,758)** once, along real Sydney
   streets, through feed-sim → Redis → YOLOv12s workers → writer → Postgres + blob store.
3. Runs the segmenter: detections → `defect_instances` → `segment_condition`.
4. Runs the bench evaluation against RDD2022 ground truth and writes a `bench_runs` row.
5. Leaves the dashboard at `http://localhost:8000` showing the mock-up's four panels with
   live data: stat chips, condition map, work list, evidence panel with working
   Confirm / Reject.

Swapping to Shervin's own weights is one env var: `YOLO_WEIGHTS=weights/yolo12s_rdd2022_continued.pt`.

## 2. Decisions taken (with Ryan, 2026-09-27)

| # | Decision | Choice |
|---|---|---|
| D1 | Model weights | Public `rezzzq/yolo12s-road-damage-rdd2022` (`yolo12s_RDD2022_best.pt`, MIT) now — the checkpoint Shervin's v4 continued from. His `yolo12s_rdd2022_continued.pt` drops in when he sends it |
| D2 | Dataset | RDD2022 **test split**, 5,758 images, from Hugging Face `dronefreak/RDD2022` (CC-BY-SA-4.0, ungated, same 26,869/5,758/5,758 split sizes as the Kaggle set Shervin used). No Kaggle token needed; fits in the free disk |
| D3 | Dashboard stack | FastAPI read API (component 7) serving JSON + one HTML/JS page (component 8) that follows the mock-up layout. Matches the 16 Sep decision "dashboard connects via FastAPI" |
| D4 | Road network | Committed OpenStreetMap extract of Sydney streets near UTS, cut into 100 m segments; feed-sim drives a route along them |
| D5 | Condition index | PCI-style deduct subset (ASTM D6433 spirit): 100 − Σ deducts per 100 m. Bands Good ≥70, Fair 40–69, Poor <40. Weights in one config module Ilana can replace |
| D6 | Review split | Peak confidence ≥0.50 → auto-accepted; 0.25–0.50 → review queue; <0.25 never emitted by the model |
| D7 | Partitioning | **Not touched.** Frames still land in `frames_default`. Ryan does the partition work himself as ILC T2 |

### Decisions I made inside those choices (flag if wrong)

- **Worker decodes colour, not grayscale.** YOLO needs BGR. `ThresholdDetector` converts
  to gray itself, so it behaves as before. Crops and thumbnails become colour, so
  bytes-per-km rises against the M2 grayscale baseline — the M2 figure must be quoted as
  grayscale.
- **Class codes are the RDD codes:** `D00` longitudinal, `D10` transverse, `D20` alligator,
  `D40` pothole. The model's 5th class (`Repair` on the base checkpoint, `other corruption`
  on Shervin's) becomes `other`: stored and shown, excluded from scoring and from the index
  because the 4-class ground truth has no equivalent.
- **Contract 2 stays frozen.** Its area-based severity (≤100 px minor, ≤1000 px major)
  marks nearly every RDD2022 box `critical`. The condition index therefore uses box area as
  a fraction of the frame, not the contract's severity label.
- **feed-sim gets an `--order all` mode:** every manifest image exactly once, seeded
  shuffle. Prevalence sampling (with replacement) stays the default for benchmark runs.
- **Clustering tolerance is 0 m for `dataset-replay` runs.** Consecutive RDD2022 frames are
  unrelated photos from different countries, so merging neighbours would invent physical
  defects. Each detection is its own instance there; the 3 m along-road merge applies to
  `drive` runs. This is honest, not a shortcut, and the report should say so.
- **Pending instances do not move the index** until confirmed. Rejected ones never do.
- **Leaflet and the base map:** Leaflet is vendored into the static folder (no CDN at
  demo time). OSM raster tiles need internet; if they fail, the segments still draw on a
  plain background.
- **The Streamlit coverage/bus panel stays** on :8501 as the observability view
  (component 12). It is not the product dashboard any more.
- **Shervin's plugin is vendored, not merged.** His `YOLO12sDetector` (class map,
  base-vs-continued fallback) is copied with attribution; a thin adapter makes it satisfy
  Contract 2. His branch is not merged as-is (Week 07 decision).
- **Held-out caveat.** Shervin's continued weights never saw the test split. The public base
  checkpoint's author split is unknown, so base-checkpoint metrics on this split may be
  optimistic. The bench row records which weights ran.

## 3. Architecture (components unchanged)

```
feed-sim(1) --frames--> Redis(2) --> worker×2(3) + yolo12s(4) --results--> writer(5) --> Postgres
                                          |                                             ^   |
                                          +--crops/thumbs--> blob-store(6)              |   v
                                                                     segmenter(11) -----+   |
                                                                     bench(9) --------------+
                                     api(7) <-- Postgres + blob-store + dataset frames (ro)
                                     dashboard(8) = static page served by api(7)
                                     observability(12) = Streamlit coverage/bus panel
```

No component is renamed, merged or removed. New code sits inside the owner's package.

## 4. Component changes

### 3/4 — detector + worker (`src/edgecv/detectors/`, `src/edgecv/worker/main.py`)
- `detectors/yolo12s.py`: Shervin's plugin logic + `Yolo12sDetector` implementing
  `Detector`. `info.params` records weights file name, weights sha256 (first 16),
  `imgsz`, `conf`, class map — so a weights swap is a new `detectors` row (params_hash).
- `detectors/registry.py`: `build_detector(os.environ["DETECTOR"])`; `threshold` (default,
  keeps every existing test green) or `yolo12s`.
- Worker builds the detector **once** at start-up and passes it to `process_one`
  (the parameter already exists). Decode switches to `IMREAD_COLOR`.
- `ultralytics` goes in an optional dependency group `yolo`; the Docker image installs it.
  Weights are not committed; `make weights` downloads them to `weights/` (gitignored),
  mounted read-only.

### 1 — feed-sim (`src/edgecv/feedsim/`)
- `route.py`: `RouteTrack` — same `fix_for(seq)` / `distance_m(n)` interface as
  `SyntheticTrack`, following a polyline at constant speed; heading from the local bearing.
- `main.py`: `--order all|sample`, `--route <geojson>`, `--source-kind`. Records
  `raw_bytes_offered` (sum of replayed file sizes) in `survey_runs.config` for the
  "uploaded of captured" chip.

### Data scripts (`scripts/`)
- `fetch_rdd2022.py`: downloads the test split images + labels (huggingface_hub,
  `allow_patterns`) to `data/rdd2022/test/`, writes `data/rdd2022/manifest.json`
  (clean = empty label file) and converts YOLO labels to pixel `ground_truth` rows keyed by
  the same relative `source_ref` feed-sim sends.
- `fetch_osm_roads.py`: one-time Overpass query → `src/edgecv/roads/sydney_demo.geojson`
  (committed, ODbL attribution) plus a precomputed drive route covering every way.
- `load_ground_truth` and `load_segments` run inside `make e2e`, both idempotent.

### 11 — segmenter (`src/edgecv/segmenter/`)
- Snap each detection's frame fix to the nearest segment within 15 m (PostGIS
  `ST_DWithin` / `<->`); unsnapped → `segment_id NULL`, flagged, never guessed.
- Cluster (tolerance per source kind, above), deterministic `cluster_key` =
  md5(run_id, class, min(seq), ordinal) as the schema comment specifies.
- Recompute-and-replace per `(run_id, segment_id)` in one transaction (R-R2).
- `segment_condition`: index per §5 below, `counts` per class, `frames_assessed`,
  `coverage_m`.
- Runs as an always-on Compose service (poll every 10 s) and as `--once` for `make e2e`.

### Condition index (`src/edgecv/segmenter/index.py`)
- Deduct per instance = `CLASS_WEIGHT[class] × extent_factor(area_fraction)`, with
  `CLASS_WEIGHT = {D40: 25, D20: 15, D10: 8, D00: 6}` and `extent_factor` 1.0 / 1.5 / 2.0 for
  box area < 5 % / 5–15 % / >15 % of the frame.
- Normalised to 100 m: `index = max(0, 100 − Σ deducts × 100 / coverage_m)`.
- Counts auto-accepted + confirmed instances; excludes pending, rejected, `other`.
- Segments with `coverage_m < 20` get no score (band `insufficient`) rather than a guess.
- Every constant sits in this one module with a docstring naming it as v0 pending R-I3.

### 7 — read API (`src/edgecv/api/`)
FastAPI + uvicorn, one service on :8000. Contract 3 moves from PROVISIONAL to a written
shape (`read_api.schema.json` regenerated).

| Endpoint | Feeds |
|---|---|
| `GET /api/runs` | run picker |
| `GET /api/runs/{run}/summary` | stat chips: km assessed, instances, % frames accounted, MB uploaded of raw captured |
| `GET /api/runs/{run}/segments.geojson` | map (Contract 6 shape; Dexter's `latest_segments_geojson` query reused) |
| `GET /api/runs/{run}/worklist?limit=` | worst-first table, change vs the authority's previous run |
| `GET /api/runs/{run}/instances?state=&segment=` | review queue, segment drill-down |
| `GET /api/instances/{run}/{cluster_key}` | evidence panel metadata + every box on that frame |
| `GET /api/frames/{run}/{seq}/image` | full frame (reference transport, read-only mount) |
| `GET /api/blobs/{kind}/{sha256}` | crop / thumbnail |
| `POST /api/instances/{run}/{cluster_key}/review` | Confirm / Reject / Reclassify → `instance_reviews`, then segment rescore |
| `GET /api/runs/{run}/bench` | accuracy + latency row |

### 8 — dashboard (`src/edgecv/api/static/`)
`index.html`, `app.js`, `styles.css`, vendored Leaflet. Layout and palette from
`RoadIQ-Product-Mockup.gen.py`: dark chrome bar, four stat cards, "Network condition" map
(segments coloured Good/Fair/Poor/unsurveyed, defect dots, selected-segment highlight,
legend), work list (click a row → map + evidence), evidence card (full frame with solid
boxes for accepted, dashed amber for review, HUD row with frame/seq/position/speed,
metadata grid, Confirm / Reject / Add to work order). "Export as work order" downloads CSV.
No framework, no build step.

### 9 — bench (`src/edgecv/bench/evaluate.py`)
Per-class precision / recall / F1 at IoU 0.5 against `ground_truth` by `source_ref`,
greedy matching by confidence; p50/p95/p99 latency from `inferences`; writes `bench_runs`.
`other` excluded.

## 5. Error handling

- Missing weights → worker exits non-zero with the `make weights` hint (no silent fallback
  to the threshold detector; a demo that quietly runs the wrong model is worse than one
  that fails).
- Missing dataset → feed-sim exits with the `make dataset` hint.
- Frame file missing at API time → 404 with a placeholder in the panel; metadata still shows.
- Tiles unreachable → map still draws segments.
- Throughput below offered rate → the bounded stream drops and the coverage chip shows it;
  `make e2e` picks an fps below the measured worker throughput so the headline run has no
  drops, and the drop behaviour stays demonstrable with a higher `FPS=`.

## 6. Testing

- Unit: adapter with a fake model (class map, bbox int conversion, `other`), registry,
  `RouteTrack`, segment cutting, clustering determinism + idempotent re-run, index maths,
  API endpoints with FastAPI `TestClient` against the test database, GT conversion.
- Integration: 50 real RDD2022 images through the live stack → assert frame/inference row
  counts, instances created, segment rows scored, review POST changes the score.
- Full run: `make e2e` on the 5,758-image split; record throughput, drops, bench metrics.
- Existing suite stays green (`make test`, `make itest`).

## 7. Out of scope

Partitioning / compaction (Ryan, T2) · batched COPY writer · real phone capture (Dexter) ·
auth · PDF work orders · CI changes · merging Dexter's `dexter-contract6-map` branch (its
query is reused; the branch stays his to merge).

## 8. T2-relevant notes (for Ryan's PDLJ)

These arise from this build and touch ILC Goal T2 (partitioning, compaction, batched
idempotent writes). They are decisions and observations, not T2 work done:

1. Partitioning deliberately left alone (D7). The e2e run puts ~5.8k real rows into
   `frames_default` — a realistic dataset for re-running finding 1 (DEFAULT blocks
   `CREATE TABLE … PARTITION OF`) at volume and for testing the migration out of DEFAULT.
2. The segmenter's recompute-and-replace churns `defect_instances` ids per run. Derived
   tables keyed by `(run_id, …)` are the natural next candidates for retention by
   partition — relevant to finding 2 (only `frames` is partitioned).
3. The writer is still per-row `INSERT … ON CONFLICT DO NOTHING`. The e2e run gives the
   baseline to measure a batched `COPY` against.
4. `survey_runs.config.raw_bytes_offered` gives the raw-vs-stored ratio per run, the input
   to the retention argument.

## 9. Changes made during the build (2026-09-27) — confirm or reverse

1. **D5 formula changed, band thresholds kept.** The count-per-100 m deduct in §4 saturated
   at 0/100 on every segment of the full replay (RDD2022 is selected for damage: 66 % of
   test images are defective, ~57 frames land per 100 m, so ~150 instances per segment).
   `segmenter/index.py` now uses **distress density** (mean share of frame area per class)
   through a log-shaped deduct per class, combined sub-additively. That is closer to ASTM
   D6433, whose deducts are functions of density. First full run: 14 Good, 55 Fair, 0 Poor,
   25 insufficient coverage. Scores cluster because a shuffled replay has no spatial
   structure — a real drive would.
2. **Ground-truth class 3 is `other`, not D40.** The dronefreak mirror's "pothole" is RDD
   "other corruption" (manholes, grates, faded lines — checked by eye), and the real
   potholes were dropped by the mirror. D40 is unscored on this split; the evaluator only
   scores classes that have ground truth.
3. **Stored bytes come from the blob store**, not `snippets.bytes` (a writer placeholder, I2).
4. **Compose host ports are overridable** (`PG_PORT`, `REDIS_PORT`) so the demo stack can
   run beside a long-lived dev stack.
5. **`tests/__init__.py` added** — ultralytics 8.4.x installs a top-level `tests` package
   that shadowed the namespace-package test directory.
6. **Live mode, vehicle marker, frame log, main-road loop** (added on request, same day).
   Ryan chose fixed speed + fixed fps (images evenly spaced) and a main-road loop over an
   every-street route. The council network is 20 disconnected pieces without the state
   arterials, so arterials are in the network file as `council: false`: driven, drawn grey,
   never segmented or scored.
7. **Heading-aware snapping** (bug found on the loop run). A frame snaps to a council
   segment only if the vehicle's heading runs along it (±30°, either direction). Before,
   frames on Harris St / Cleveland St within 15 m of a side-street mouth credited their
   defects to the side street — a wrong-road attribution, which §6 ranks worse than
   unlocated. About 1 % of council frames (mid-turn at corners) are now unsnapped instead.
   The read API's `/track` uses the same SQL (`segmenter.SNAP_LATERAL`), so the dashed
   "driven, not surveyed" legs on the map are exactly the frames the score ignored.
8. **Run control.** Pause / resume / cancel from the dashboard via a Redis key feed-sim
   checks before each frame. Resume restarts pacing (no catch-up burst). Cancel keeps what
   was sent; `make e2e` still drains, segments and scores the partial run.
9. **Random route per run** (replaces the fixed loop as default; Ryan, same day). The loop
   was 4.7 km against the ~10 km that 5,758 frames at 50 km/h / 8 fps need, so a run drove
   it 2.1 times. `roads.build_random_route` draws a fresh seeded drive each run over the
   council streets, side streets included: an undriven street at each junction (weighted to
   carry straight on and towards more undriven road), a cheapest path to the nearest
   undriven street when stuck (driven road ×2, state roads ×5), dead-end stubs ≤150 m left
   out. Sized to the frame count, so it ends as the images do. Re-driven share 18–25 %
   over five seeds (a street grid can't be covered without repeating some block; the loop
   re-drove ~53 %). The planned route and seed are written to `survey_runs.config` before
   the first frame; the map shows it dotted ahead of the car. `finish_run` now MERGES
   config instead of replacing it, so the start-of-run write survives.
