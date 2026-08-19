.PHONY: install up down migrate test itest demo

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
