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

## Architecture

Redis Streams is the edge/backend seam. Left of it (feed-sim, worker) is what would run on
a device; right of it (writer, Postgres, dashboard) is backend infrastructure.

feed-sim is the exception to that split: it also registers each run in Postgres'
`survey_runs` (`upsert_run`/`finish_run`), because it's the only component that knows
`run_id`, `target_fps`, `prevalence`, `transport` and the source. That's a Postgres write
on the edge side of the seam -- an accepted trade for this milestone, since feed-sim is a
test harness standing in for both the device and the run-registration step, not a
statement that the real device will write to Postgres directly.

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
