FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
# --group yolo: the YOLOv12s detector's runtime (CPU-only torch on Linux).
RUN uv sync --frozen --no-dev --group yolo

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
