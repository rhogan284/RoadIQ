# RoadIQ

Edge-First Road Condition Assessment Tool - Studios 2026

UTS 41087 Applications Studio B, Spring 2026. Product owner: A/Prof Wenjing Jia.

## Quick start

    make install          # uv sync
    cp .env.example .env
    make up               # redis + postgres
    make migrate          # apply schema
    make test             # unit tests
    make itest            # integration tests (needs `make up`; stops app services first, see below)
    make fixtures         # generate synthetic fixture images (needed once for make demo)
    make demo             # full pipeline under compose
    make bench-bytes      # bytes-per-kilometre figure for the latest run

## End-to-end demo — RDD2022 through every component

The working demo for owner meeting 5 (30 Sep 2026): the RDD2022 test split replayed along
real Sydney streets through every component, into a live dashboard.
Design, decisions and every change made on the way:
`docs/superpowers/specs/2026-09-27-e2e-demo-design.md`.

### Setup (once)

    uv sync --group yolo    # + ultralytics / CPU torch for the YOLOv12s detector
    make weights            # YOLOv12s road-damage weights → weights/ (19 MB, gitignored)
    make dataset            # RDD2022 test split → data/rdd2022/ (5,758 images, ~1.6 GB, gitignored)

Already running `make up` on 5432/6379? Give the demo its own project and ports with a
local `.env` (gitignored; Compose reads it for every command, so nothing can land on your
dev stack by accident):

    COMPOSE_PROJECT_NAME=roadiq-e2e
    PG_PORT=55432
    REDIS_PORT=56379

and prefix host-side commands with
`PG_DSN=postgresql://edgecv:edgecv@localhost:55432/edgecv REDIS_URL=redis://localhost:56379/0`.

### Run it

    make stack                     # redis, postgres, writer, 2 workers, segmenter, runner, api, dashboard
    open http://localhost:8000     # product dashboard (components 7 + 8)
    open http://localhost:8501     # pipeline observability: coverage + bus health (component 12)

Then either press **＋ New run** on the dashboard, or from a terminal:

    make e2e                       # FPS=12 make e2e to push past what the workers can keep up with

A run replays the images through feed-sim → Redis → YOLOv12s workers → writer → Postgres +
blob store, waits for the pipeline to drain, runs the segmenter (detections →
`defect_instances` → `segment_condition`) and scores the run against RDD2022 ground truth
into `bench_runs`. The whole split takes ~12 min at 8 fps on an M-series Mac.

### The dashboard

| Feature | What it shows |
|---|---|
| Stat chips | km assessed (+ metres not assessed), defects on the register / to review, **frames accounted for** (landed + counted drops ÷ offered, with frames in flight and dropped), this run's crops uploaded vs raw bytes captured |
| Network condition map | 100 m council segments coloured Good ≥70 / Fair 40–69 / Poor <40 (index v0), unsurveyed segments dashed, state roads grey ("not ours", never scored), defect dots (amber = awaiting review) |
| Vehicle | arrow at the newest frame on the bus, rotated to its heading; trail of landed frames behind it — the gap is the pipeline's lag |
| Planned route | the run's route, dotted ahead of the car, faint behind it |
| Driven, not surveyed | grey dashed legs where the car drove but nothing was scored (state roads, turns) |
| Work list | worst-first segments, change against the authority's previous run, CSV export as a work order |
| Evidence panel | the frame the vehicle saw with every box on it (solid = accepted, dashed amber = review), HUD, metadata, **Confirm / Reject / Add to work order**; review queue ‹ › |
| Benchmark | per-class precision / recall / F1 and latency for the run |
| **● LIVE** | click for the frame log: one line per frame in landing order (`#log` deep link) |
| **❚❚ Pause / ▶ Resume / ■ Cancel** | control a live run; cancel keeps what was sent and still scores it |
| **＋ New run** | whole split, 1,000 or 300 images · 4 / 8 / 12 fps · random route or the fixed loop · optional seed (`#new` deep link) |

Live runs refresh every 5 s (vehicle every 1 s) and the page follows the newest run
unless you pick an older one from the run menu.

### Routes

Each run drives a **new random route** over the council streets, side streets included,
sized so the images run out as the route does (no second lap). At each junction it takes
an undriven street, weighted to carry straight on; when stuck it takes the cheapest path
to the nearest undriven street (state roads only as connectors); dead-end stubs ≤150 m are
left out. Re-driving is 18–25 % over five seeds — a street grid cannot be covered without
repeating some blocks. The planned route and its seed are written to `survey_runs.config`
before the first frame.

- `--route-seed N` repeats a route; `--route-mode loop` is a fixed 4.7 km main-road loop;
  `--route-mode cover` drives every street (with jumps between road pieces).
- The network (`src/edgecv/roads/sydney_demo.geojson`, © OpenStreetMap contributors, ODbL)
  is Ultimo / Chippendale / Glebe council roads plus the state arterials that join them
  (`council: false`).
- A frame snaps to a segment only within 15 m **and** if the vehicle's heading runs along
  it (±30°), so a frame on Harris St never credits a defect to a side street.

### Services added for the demo

| Service | Component | Role |
|---|---|---|
| `worker` | 3 + 4 | now runs Shervin's YOLOv12s by default (`DETECTOR=yolo12s`); `DETECTOR=threshold` restores the baseline |
| `segmenter` | 11 | always-on: snap, cluster into `defect_instances`, score `segment_condition` (recompute-and-replace per run) |
| `api` | 7 + 8 | FastAPI read layer; serves the dashboard at `/` |
| `runner` | harness | runs **＋ New run** requests from a Redis queue (feed-sim → drain → segmenter → bench), in its own container so an API redeploy can't kill a run |
| `dashboard` | 12 | the Streamlit coverage + bus-health panel, unchanged |

### Read API

| Endpoint | |
|---|---|
| `GET /api/runs`, `/api/runs/{id}/summary`, `/segments.geojson`, `/worklist`, `/instances`, `/bench` | the dashboard's panels |
| `GET /api/instances/{run}/{cluster_key}`, `/api/frames/{run}/{seq}/image`, `/api/blobs/{kind}/{sha}` | evidence panel |
| `POST /api/instances/{run}/{cluster_key}/review` | `confirmed` / `rejected` / `reclassified` / `pending` → `instance_reviews`, then re-score |
| `GET /api/runs/{id}/position`, `/route`, `/track`, `/log`; `GET /api/network` | vehicle, planned route, driven legs, frame log, state roads |
| `POST /api/runs/start`, `GET /api/runner` | start a run (409 while another is feeding), runner status |
| `POST /api/runs/{id}/control` | `pause` / `resume` / `cancel` |

### Knobs

`FPS`, `WORKERS` (default 2), `WORKER_THREADS`, `DETECTOR`, `YOLO_WEIGHTS`,
`FRAMES_MAXLEN` (bus depth, default 1,000). To swap in Shervin's own weights: put
`yolo12s_rdd2022_continued.pt` in `weights/` and set
`YOLO_WEIGHTS=weights/yolo12s_rdd2022_continued.pt`. The weights hash is part of the
detector's `params`, so a swap is a new `detectors` row and benchmark rows never mix models.

### Results so far (2026-09-27, public `rezzzq` base weights)

- Whole split at 8 fps: 5,758 / 5,758 frames, 0 dropped, 8.0 fps, p50 166 ms / p95 336 ms
  per frame; box-level P 0.888 · R 0.835 · F1 0.861 at IoU ≥ 0.5 on D00/D10/D20.
- Whole split at 12 fps: the workers land ~8–9 fps, the 1,000-deep bus fills after ~4 min,
  and 600 frames are refused and counted — the bounded-buffer design working as intended.
- Storage: ~225 MB of crops for 1.63 GB captured (~7×). Crops are colour now (the worker
  decodes colour for YOLO), so this is not comparable with the grayscale M2 figure below.

**Caveats that must travel with the accuracy numbers:**

1. **Pothole (D40) is unscored.** The `dronefreak/RDD2022` mirror's "pothole" label is really
   RDD "other corruption" (manholes, grates, faded lines — checked by eye) and it dropped the
   real potholes; see `scripts/fetch_rdd2022.py`.
2. **The public checkpoint may have seen these images** — its training split is unknown, and
   F1 0.86 is well above Shervin's mAP50 of 0.54. Rerun with his continued weights, which
   never saw this split, before quoting accuracy.
3. **The condition index is v0**, a PCI-style density deduct standing in for Ilana's R-I3
   (`segmenter/index.py`). A shuffled replay has no spatial structure, so segment scores
   cluster; a real drive would not.

### Known limits

- `snippets.bytes` is still a writer placeholder (I2); per-run storage is measured on disk
  from the run's crop hashes. Thumbnails are not linked to a run.
- A run started from the dashboard stops if the `runner` container is restarted; it is
  left open and can be cancelled.
- Partitioning is untouched: frames still land in `frames_default` (ILC T2 work).

## Architecture

Redis Streams is the edge/backend seam. Left of it (feed-sim, worker) is what would run on
a device; right of it (writer, Postgres, dashboard) is backend infrastructure.

feed-sim is the exception to that split: it also registers each run in Postgres'
`survey_runs` (`upsert_run`/`finish_run`), because it's the only component that knows
`run_id`, `target_fps`, `prevalence`, `transport` and the source. That's a Postgres write
on the edge side of the seam -- an accepted trade for this milestone, since feed-sim is a
test harness standing in for both the device and the run-registration step, not a
statement that the real device will write to Postgres directly.

Services: `feedsim` (1) · `redis` (2, the bus) · `worker` ×2 (3 + 4) · `writer` (5) ·
blob store volume (6) · `api` (7 + 8) · `bench` (9) · `segmenter` (11) · `dashboard` (12),
plus `postgres` and the demo's `runner`. Component 10 (capture) is Dexter's phone rig and
is replaced here by feed-sim replaying the dataset along a route.

See `docs/` and the design spec for detail.

## Measurement — bytes per kilometre

Success criterion 1 is "store and upload at least 100x less data per kilometre than
keeping every frame". `make bench-bytes` reports it for the most recent run:

    RoadIQ -- bytes per kilometre (success criterion 1)
      frames          600  (600 with a GPS fix, 0 without)
      distance        0.555 km
      stored           106.8 KiB    192.5 KiB /km
      every frame       24.7 MiB     44.5 MiB /km
      reduction       236.7x   (target 100x)   PASS

It exits non-zero when a run misses the target, so it can gate a build later.

The per-kilometre figures divide by the **assessed** distance, not the whole attempted
track. Stored bytes and the every-frame baseline both come from the frames actually
processed, so the road those frames covered is the matching denominator. The ratio is
unaffected either way — both sides divide by the same number — but the absolutes are
not, which is only visible now that coverage is reported next to them.

**feed-sim now emits a synthetic GPS fix on every frame** (`--no-gps` opts out).
Before this every frame reached Postgres with `lat IS NULL`, so there was no
distance to divide by and the figure could not be computed at all. The track is a
rhumb line at constant speed -- deliberately not realistic, because a measurement
rig wants a denominator that is analytically known. `survey_runs.config.gps_track`
records the parameters so the distance can be audited.

Three things to know before quoting the number:

- **The fixtures are synthetic 256px images averaging ~42 KB**, not real RDD2022
  road photos at ~500 KB. The every-frame baseline here is therefore about 12x
  smaller than a real drive's, so this figure is a working measurement of the
  pipeline, not a claim about real-world storage.
- **Storage scales with defect prevalence.** This run used `--prevalence 0.03`
  and stored 15 detections' crops. A rougher road stores more.
- **`bytes_stored` is measured off the blob store, not `snippets.bytes`** -- that
  column is a 0 placeholder on every row, because the writer only ever sees
  content hashes on the wire. The store totals the whole tree, and snippets carry
  no run linkage, so a per-run figure needs a store holding one run:
  `BLOB_ROOT=/blobs/m2-run02 make bench-bytes`. Correct attribution on a shared
  store needs a `run_snippets` join table written by the worker (Week 8).

## Coverage — drops as metres of road not assessed

Success criterion 2 says "any frame we drop is counted and shown as a gap in the
survey, measured in metres of road not assessed". `make bench-bytes` now reports it
alongside the storage figures:

    -- coverage (success criterion 2) ------------------------
      assessed             472.3 m
      not assessed          82.4 m   (88 frame(s) lost in 1 gap(s))
      largest gap           82.4 m
      coverage              85.1%

Measured on a deliberately degraded run: `FRAMES_MAXLEN=5`, both workers paused for
six seconds twelve seconds into a 40-second drive. 88 of 600 frames were refused, and
they show up as one 82.4 m hole in the survey rather than as a frame count.

A dropped frame has no row and no position of its own, but the frames either side of
it do, so the hole is measured as the distance between its neighbours. That needs no
knowledge of the track, so it works the same way on a real drive.

Two absences look alike in SQL and mean opposite things, and `bench/coverage.py`
keeps them apart:

- **no row for a seq** — the frame was dropped. That road was never assessed. A gap.
- **row with `lat` NULL** — the frame was processed, so coverage is proved; we just
  cannot place it. The distance across it still counts as assessed, interpolated from
  its positioned neighbours. Calling this a gap would understate coverage we can
  actually evidence.

`FRAMES_MAXLEN` is overridable so a sweep can vary the buffer depth and force drops
on purpose: `FRAMES_MAXLEN=5 docker compose up -d worker writer`.

## Bus health — lag, pending, and stuck work

`edgecv.bus.observe` reads what Redis exposes about the consumer group, and the
dashboard shows it. Three numbers that get confused constantly:

| | meaning |
|---|---|
| **lag** | entries the group has not been *delivered* yet — backlog |
| **pending** | entries delivered but not yet `XACK`ed — work in progress |
| **idle** | how long an entry has sat in a consumer's PEL undone — trouble |

Rising `lag` means the workers cannot keep up. Rising `pending` with a high `idle`
means a worker took work and died, which is what `XAUTOCLAIM` exists to fix.

**`lag` can come back nil, and the panel shows "unknown" rather than 0.** That happens
when entries *ahead* of the group's last-delivered-id are deleted, which makes the
count uncomputable for that group from then on. Our writer only `XDEL`s after `XACK`
— at or below last-delivered-id — so our lag stays computable.

This is also why the producer refuses the **newest** frame when the buffer is full
rather than using Redis's `XADD MAXLEN`/`XTRIM`, which evicts the **oldest**.
Trimming does not respect the pending-entries list: it will delete entries a worker
has claimed and not acknowledged, leaving dangling PEL references that `XPENDING`
still reports as in flight. And under backpressure the oldest entries are exactly the
ones a lagging consumer has not reached — so trimming would blind the lag panel
precisely when the pipeline is under the load you most need to watch. One design
choice protects both the data and the ability to see what is happening to it.

Measured, not assumed — see `PDLJ/_evidence/2026-09-05-t1-maxlen-vs-pel.txt` and
`2026-09-06-t1-lag-counter-invalidation.txt` in the vault.

## Contracts

`src/edgecv/contracts/` is frozen. Changing `frame.py` or `detection.py` breaks other
people's work — raise it with the team before editing.

## Development / Troubleshooting

On some machines (observed on macOS) the editable-install `.pth` file that `uv sync`
generates isn't picked up by `site.py`, so a bare `python -c "import edgecv"` fails
with `ModuleNotFoundError` even right after `make install`. `uv run pytest` still
works because pytest's own `pythonpath = ["src"]` setting doesn't depend on the
`.pth` file.

All `make` targets set `PYTHONPATH=src` so they aren't affected. If you run a
bare `python`/`uv run python` command outside of `make` and hit
`ModuleNotFoundError: No module named 'edgecv'`, either use the equivalent `make`
target or export it yourself:

    PYTHONPATH=src uv run python -m edgecv.db.migrate

The integration suite (`make itest`, or `tests/integration/`) is mutually exclusive
with a running app stack, not just by convention: its `rds` fixture flushes the
whole Redis DB on setup and teardown, and `tests/integration/test_chaos_worker_kill.py`
reads from the same `frames`/`workers` stream and consumer group the containerised
`worker` service consumes from, so a live stack races the test's own consumers and
breaks its exact frame counts. `make itest` stops `worker`, `writer`, `feedsim` and
`dashboard` before running pytest (redis and postgres are left up); it does not
restart them afterwards, so run `make demo` or `docker compose up -d` again when you
want the full pipeline back. If you invoke `pytest` directly instead of through
`make itest`, stop those four services yourself first.
