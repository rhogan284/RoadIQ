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
