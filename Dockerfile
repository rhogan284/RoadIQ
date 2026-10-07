FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

# The cache mount below is its own filesystem, so uv cannot hard-link from it.
ENV UV_LINK_MODE=copy

WORKDIR /app
# Dependencies first, from the lock file alone: this layer (torch and all, ~1 GB) is
# rebuilt only when pyproject.toml or uv.lock changes, not on every code or README edit.
# The cache mount keeps downloaded wheels across builds, so a lock change fetches only
# what changed.
# --group yolo: the YOLOv12s detector's runtime (CPU-only torch on Linux).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group yolo --no-install-project

COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group yolo

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
