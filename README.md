# RoadIQ

Edge-First Road Condition Assessment Tool - Studios 2026

UTS 41087 Applications Studio B, Spring 2026. Product owner: A/Prof Wenjing Jia.

## Quick start

    make install          # uv sync
    cp .env.example .env
    make up               # redis + postgres
    make migrate          # apply schema
    make test             # unit tests
    make itest            # integration tests (needs `make up`)
    make demo             # full pipeline under compose

## Architecture

Redis Streams is the edge/plant seam. Left of it (feed-sim, worker) is what would run on
a device; right of it (writer, Postgres, dashboard) is plant infrastructure.

See `docs/` and the design spec for detail.

## Contracts

`src/edgecv/contracts/` is frozen. Changing `frame.py` or `detection.py` breaks other
people's work — raise it with the team before editing.
