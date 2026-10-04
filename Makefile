.PHONY: install up down migrate test weights dataset stack e2e evaluate video

# Editable-install .pth files can end up ignored by site.py (e.g. macOS marks
# them UF_HIDDEN, or the working directory just isn't on sys.path). Setting
# PYTHONPATH=src here makes every target below resolve `edgecv` regardless.
export PYTHONPATH := src

install:
	uv sync --group yolo

up:
	docker compose up -d redis postgres

down:
	docker compose down -v

migrate:
	uv run python -m edgecv.db.migrate

# The e2e suite (tests/e2e/) owns the `frames`/`workers` stream and consumer group
# outright: its `rds` fixture flushes the whole Redis DB on setup AND teardown, and
# the worker-failure tests read from the same stream/group the containerised workers
# consume from. It also TRUNCATES Postgres. So the app services stop first -- redis
# and postgres stay up; `make stack` brings the rest back. Needs `make weights dataset`.
test:
	docker compose stop writer worker segmenter runner api dashboard
	uv run pytest -v

# --- E2E demo (docs/superpowers/specs/2026-09-27-e2e-demo-design.md) ---------------
# One-time downloads, both gitignored: the YOLOv12s weights (~19 MB) and the RDD2022
# test split (5,758 images, ~1 GB).
weights:
	uv run --with huggingface_hub python -m scripts.fetch_weights

dataset:
	uv run python -m scripts.fetch_rdd2022

# The long-lived services. feed-sim and bench are one-shot tools that `make e2e` runs.
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

# Replay a dashcam video instead of the dataset, in capture order, along the same
# simulated route. No ground truth, so no bench step.
#   make video VIDEO=~/Downloads/road_footage.mp4
VIDEO ?=
VIDEO_DIR = data/video/$(basename $(notdir $(VIDEO)))
video: stack
	uv run python -m scripts.extract_video "$(VIDEO)" --fps $(FPS) --out $(VIDEO_DIR)
	uv run python -m scripts.e2e --fps $(FPS) --manifest $(VIDEO_DIR)/manifest.json --no-bench
