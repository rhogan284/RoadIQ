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
