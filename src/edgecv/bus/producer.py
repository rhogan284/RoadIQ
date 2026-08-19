"""Publish frame envelopes onto a bounded Redis Stream.

Backpressure is explicit: when the stream is at capacity the frame is DROPPED and
counted, mirroring an edge device with a finite buffer. Consumers XDEL after XACK, so
XLEN reflects true backlog rather than total history.
"""
from __future__ import annotations

import redis

from edgecv.contracts.frame import FrameEnvelope


class FrameProducer:
    def __init__(self, client: redis.Redis, *, stream: str, maxlen: int) -> None:
        self.client = client
        self.stream = stream
        self.maxlen = maxlen
        self.offered = 0
        self.dropped = 0

    def publish(self, envelope: FrameEnvelope) -> str | None:
        """Return the stream entry id, or None if the frame was dropped."""
        self.offered += 1
        if self.client.xlen(self.stream) >= self.maxlen:
            self.dropped += 1
            self.client.incr(f"stats:{envelope.run_id}:dropped")
            return None
        entry_id = self.client.xadd(self.stream, envelope.to_fields())
        self.client.incr(f"stats:{envelope.run_id}:published")
        return entry_id.decode() if isinstance(entry_id, bytes) else entry_id
