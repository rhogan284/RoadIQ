.PHONY: install up down migrate test itest fixtures demo bench-bytes

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
