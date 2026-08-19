"""metadata-writer: the sole Postgres writer.

One writer means one place that owns pooling, batching and idempotency. Workers stay
stateless. Batch size is a benchmark axis — swapping these per-row INSERTs for COPY is
a measurable optimisation, deliberately left for the benchmark phase.

Coverage is read from the plain `run_coverage_1min` VIEW (see db/migrations); it is
computed on read, so there is no periodic refresh step here.
"""
from __future__ import annotations

import signal

import psycopg
import redis

from edgecv.config import Settings
from edgecv.contracts.detection import InferenceResult
from edgecv.db.migrate import apply_migrations
from edgecv.db.repository import Repository, WriteStats

BLOCK_MS = 2000


def _ensure_group(client: redis.Redis, stream: str, group: str) -> None:
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def drain_once(client: redis.Redis, repo: Repository, *, stream: str, group: str,
               consumer: str, batch: int, block_ms: int = 0) -> WriteStats:
    _ensure_group(client, stream, group)
    # Own PEL first: entries delivered to this consumer but never acked (a crash
    # between XREADGROUP and XACK, or an exception raised anywhere inside
    # write_results). ">" never returns them, so without this read they are
    # unreachable forever -- a poison message will now be retried forever
    # instead, which is the accepted trade-off at this scale (no dead-letter
    # handling is in scope).
    response = client.xreadgroup(group, consumer, {stream: "0"}, count=batch)
    if not response or not response[0][1]:
        response = client.xreadgroup(group, consumer, {stream: ">"},
                                     count=batch, block=block_ms or None)
    if not response:
        return WriteStats(frames=0, inferences=0, detections=0, skipped_duplicates=0)

    _stream_name, entries = response[0]
    results, entry_ids = [], []
    for entry_id, fields in entries:
        raw = fields.get(b"json") or fields.get("json")
        results.append(InferenceResult.from_json(raw))
        entry_ids.append(entry_id)

    stats = repo.write_results(results)
    # Ack only after the transaction committed. A crash before this point means
    # redelivery, which the unique constraints make harmless.
    client.xack(stream, group, *entry_ids)
    client.xdel(stream, *entry_ids)
    return stats


def main() -> None:
    settings = Settings.from_env()
    client = redis.from_url(settings.redis_url, decode_responses=False)
    running = True

    def stop(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        apply_migrations(conn)
        repo = Repository(conn)
        print("writer started", flush=True)
        while running:
            stats = drain_once(client, repo, stream=settings.results_stream,
                               group="writers", consumer="writer-1", batch=100,
                               block_ms=BLOCK_MS)
            if stats.frames:
                print(f"wrote frames={stats.frames} detections={stats.detections} "
                      f"dupes={stats.skipped_duplicates}", flush=True)


if __name__ == "__main__":
    main()
