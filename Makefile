.PHONY: install up down migrate test itest demo

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
	uv run pytest -m integration -v

demo:
	docker compose up --build
