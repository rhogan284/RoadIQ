"""Read frames from a Redis Stream consumer group.

Delivery is at-least-once: a frame stays pending until acked, and XAUTOCLAIM lets a
surviving worker take over a dead worker's in-flight frames. Duplicate processing is
therefore expected — the database's UNIQUE(run_id, seq, detector_id) makes it harmless.

ack() does XACK *and* XDEL so XLEN reflects true backlog, which is what the producer's
drop policy tests against.
"""
from __future__ import annotations

import redis

from edgecv.contracts.frame import FrameEnvelope

Delivery = tuple[str, FrameEnvelope]


class FrameConsumer:
    def __init__(self, client: redis.Redis, *, stream: str, group: str,
                 consumer: str, block_ms: int = 2000) -> None:
        self.client = client
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.block_ms = block_ms

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    @staticmethod
    def _decode(entry_id) -> str:
        return entry_id.decode() if isinstance(entry_id, bytes) else entry_id

    def read(self, *, count: int = 10) -> list[Delivery]:
        response = self.client.xreadgroup(
            self.group, self.consumer, {self.stream: ">"},
            count=count, block=self.block_ms,
        )
        if not response:
            return []
        _stream_name, entries = response[0]
        return [(self._decode(eid), FrameEnvelope.from_fields(fields))
                for eid, fields in entries]

    def reclaim(self, *, min_idle_ms: int, count: int = 10) -> list[Delivery]:
        """Take over frames another consumer read but never acked."""
        _cursor, entries, _deleted = self.client.xautoclaim(
            self.stream, self.group, self.consumer,
            min_idle_time=min_idle_ms, count=count,
        )
        return [(self._decode(eid), FrameEnvelope.from_fields(fields))
                for eid, fields in entries if fields]

    def ack(self, entry_id: str) -> None:
        self.client.xack(self.stream, self.group, entry_id)
        self.client.xdel(self.stream, entry_id)
