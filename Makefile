.PHONY: install up down migrate test itest fixtures demo bench-bytes weights dataset stack e2e evaluate

# Editable-install .pth files can end up ignored by site.py (e.g. macOS marks
# them UF_HIDDEN, or the working directory just isn't on sys.path). Setting
# PYTHONPATH=src here makes every target below resolve `edgecv` regardless.
export PYTHONPATH := src

install:
	uv sync

up:
	docker compose up -d redis postgres

down:
	docker compose down -v

migrate:
	uv run python -m edgecv.db.migrate

test:
	uv run pytest -m "not integration" -v

itest:
	# The integration suite (in particular tests/integration/test_chaos_worker_kill.py)
	# owns the `frames`/`workers` stream and consumer group outright: its `rds`
	# fixture flushes the whole Redis DB on setup AND teardown, and its chaos
	# test reads from the same stream/group the containerised workers consume
	# from. A live app stack racing the test's own consumers breaks its frame
	# counts, and the flush destroys the live stack's stream regardless. So
	# stop the app services first -- redis and postgres stay up, `make up`
	# brings the app services back when you want them.
	docker compose stop worker writer feedsim dashboard
	uv run pytest -m integration -v

fixtures:
	uv run python -m scripts.make_fixtures

demo: fixtures
	docker compose up --build

# Week 6 milestone: first bytes-per-kilometre measured and recorded.
# Runs in a container because the blob store is the `blobs` named volume, which
# the host cannot see. Exits non-zero when the run misses success criterion 1's
# 100x target, so it can gate a build later.
bench-bytes:
	docker compose run --rm bench

# --- E2E demo (docs/superpowers/specs/2026-09-27-e2e-demo-design.md) ---------------
# One-time downloads, both gitignored: the YOLOv12s weights (~19 MB) and the RDD2022
# test split (5,758 images, ~1 GB).
weights:
	uv run --with huggingface_hub python -m scripts.fetch_weights

dataset:
	uv run python -m scripts.fetch_rdd2022

# The long-lived services. feed-sim is left out: `docker compose up` would otherwise
# replay the synthetic fixtures into the same database on every start.
stack:
	docker compose up -d --build redis postgres writer worker segmenter runner api dashboard

# Replay the whole test split along the Sydney route, wait for the writer to land every
# frame, segment, score against ground truth. Dashboard: http://localhost:8000
FPS ?= 8
e2e: stack
	uv run python -m scripts.e2e --fps $(FPS)

# Re-score the latest run without replaying it.
evaluate:
	docker compose run --rm -T bench python -m edgecv.bench.evaluate --load-gt data/rdd2022/ground_truth.json
