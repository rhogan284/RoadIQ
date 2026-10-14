"""Read frames from a Redis Stream consumer group.

Delivery is at-least-once: a frame stays pending until acked, and XAUTOCLAIM lets a
surviving worker take over a dead worker's in-flight frames. Duplicate processing is
therefore expected — the database's UNIQUE(run_id, seq, detector_id) makes it harmless.

ack() does XACK *and* XDEL so XLEN reflects true backlog, which is what the producer's
drop policy tests against.
"""
from __future__ import annotations

from dataclasses import dataclass

import redis

from edgecv.contracts.frame import FrameEnvelope

Delivery = tuple[str, FrameEnvelope]


@dataclass(frozen=True, slots=True)
class ReclaimOutcome:
    """What one XAUTOCLAIM pass took over, and what it found already destroyed.

    XAUTOCLAIM answers with three things, and this module used to keep only the
    middle one:

      [0] cursor    how far the scan of the pending-entries list got
      [1] entries   pending entries this consumer now owns — recoverable work
      [2] deleted   pending entries it PURGED from the PEL, because the stream
                    entry they point at no longer exists — work that is gone

    Element [2] is the only channel Redis offers for the third case. Measured on
    Redis 7.4.10: a destroyed entry does not come back in `entries` at all, not
    even with empty fields, so discarding [2] made a frame lost in flight
    indistinguishable from an idle pass that found nothing to do.

    The payload dies with the entry, so `lost_entry_ids` are stream ids and
    nothing else — no run_id, no seq, no position. We can say a frame was
    destroyed after a worker had accepted it. We cannot say which metre of road
    it covered. That is the whole difference between the drop policy's *known*
    coverage gap and this *unknown* one.
    """

    deliveries: list[Delivery]
    lost_entry_ids: list[str]

    @property
    def progressed(self) -> bool:
        """True if this pass did anything, so a drain loop must call again.

        Purging tombstones is progress even though it recovers no work. A loop
        that stops as soon as `deliveries` is empty gives up on a page of
        deletions and strands the live orphans queued behind them in the PEL.
        """
        return bool(self.deliveries or self.lost_entry_ids)


class FrameConsumer:
    def __init__(self, client: redis.Redis, *, stream: str, group: str,
                 consumer: str, block_ms: int = 2000) -> None:
        self.client = client
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.block_ms = block_ms
        # Mirrors FrameProducer.dropped. That counter is frames refused at the
        # door; this one is frames accepted and then destroyed underneath us.
        self.lost_in_flight = 0

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

    def reclaim(self, *, min_idle_ms: int, count: int = 10) -> ReclaimOutcome:
        """Take over frames another consumer read but never acked.

        One XAUTOCLAIM call. The cursor is deliberately not looped here — a
        worker's main loop must not block on an unbounded drain — so the caller
        decides how hard to drain. Loop on `progressed`, never on `deliveries`:
        see ReclaimOutcome.
        """
        reply = self.client.xautoclaim(
            self.stream, self.group, self.consumer,
            min_idle_time=min_idle_ms, count=count,
        )
        _cursor, entries = reply[0], reply[1]
        # Redis >= 7.0 reports purged ids as a third element. Redis 6.2 has no
        # third element and instead hands back a purged entry inside `entries`
        # with no fields. Fold both shapes into one list so the caller does not
        # have to care which server it is talking to.
        deleted = reply[2] if len(reply) > 2 else []

        deliveries = [(self._decode(eid), FrameEnvelope.from_fields(fields))
                      for eid, fields in entries if fields]
        lost = [self._decode(eid) for eid, fields in entries if not fields]
        lost += [self._decode(eid) for eid in deleted]

        self.lost_in_flight += len(lost)
        return ReclaimOutcome(deliveries=deliveries, lost_entry_ids=lost)

    def ack(self, entry_id: str) -> None:
        self.client.xack(self.stream, self.group, entry_id)
        self.client.xdel(self.stream, entry_id)
