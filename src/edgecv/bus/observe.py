"""Consumer-group observability — is the bus keeping up, and is anything stuck?

The chaos test proves `XAUTOCLAIM` recovery works. Until now nothing let us
*watch* it: the property was asserted in a test and then the pipeline ran blind.
This reads the three things Redis exposes about a consumer group and gives the
dashboard something to show.

The three numbers mean different things and get confused constantly:

  lag       entries the group has not been DELIVERED yet. Backlog.
  pending   entries delivered but not yet XACKed. Work in progress.
  idle      how long an entry has sat in a consumer's PEL undone. Trouble.

A rising `lag` means the workers cannot keep up. A rising `pending` with a high
`idle` means a worker took work and died — which is what `XAUTOCLAIM` exists to
fix, and what the chaos test simulates.

`lag` can come back nil, and a panel must show that as "unknown" rather than
zero. It happens when entries AHEAD of the group's last-delivered-id are
deleted, which makes the count uncomputable for that group from then on. Our
writer only ever XDELs after XACK — at or below last-delivered-id — so our lag
stays computable. Redis's own `XADD MAXLEN` trimming would not have been safe
here: under backpressure the oldest entries are exactly the ones a lagging
consumer has not reached, so trimming would blind this panel precisely when the
pipeline is under the load you most need to watch. Measured, see
`PDLJ/_evidence/2026-09-06-t1-lag-counter-invalidation.txt`.
"""
from __future__ import annotations

from dataclasses import dataclass

import redis


def _field(mapping: dict, key: str):
    """XINFO replies come back with bytes keys when decode_responses=False,
    which the app sets so frame payloads stay binary."""
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode())


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True, slots=True)
class GroupHealth:
    stream: str
    group: str
    stream_length: int
    consumers: int
    pending: int
    entries_read: int | None
    lag: int | None
    last_delivered_id: str

    @property
    def lag_known(self) -> bool:
        """False means Redis cannot compute the backlog — show "unknown", never 0."""
        return self.lag is not None


@dataclass(frozen=True, slots=True)
class ConsumerHealth:
    name: str
    pending: int
    idle_ms: int
    inactive_ms: int | None


@dataclass(frozen=True, slots=True)
class StuckEntry:
    entry_id: str
    consumer: str
    idle_ms: int
    delivery_count: int


def group_health(client: redis.Redis, *, stream_name: str,
                 group: str) -> GroupHealth:
    for raw in client.xinfo_groups(stream_name):
        if _text(_field(raw, "name")) != group:
            continue
        lag = _field(raw, "lag")
        entries_read = _field(raw, "entries-read")
        return GroupHealth(
            stream=stream_name,
            group=group,
            stream_length=client.xlen(stream_name),
            consumers=int(_field(raw, "consumers")),
            pending=int(_field(raw, "pending")),
            entries_read=None if entries_read is None else int(entries_read),
            lag=None if lag is None else int(lag),
            last_delivered_id=_text(_field(raw, "last-delivered-id")),
        )
    raise LookupError(
        f"no consumer group {group!r} on stream {stream_name!r} — "
        f"has the group been created?"
    )


def consumer_health(client: redis.Redis, *, stream_name: str,
                    group: str) -> list[ConsumerHealth]:
    out = []
    for raw in client.xinfo_consumers(stream_name, group):
        inactive = _field(raw, "inactive")   # Redis 7.2+
        out.append(ConsumerHealth(
            name=_text(_field(raw, "name")),
            pending=int(_field(raw, "pending")),
            idle_ms=int(_field(raw, "idle")),
            inactive_ms=None if inactive is None else int(inactive),
        ))
    return out


def stuck_entries(client: redis.Redis, *, stream_name: str, group: str,
                  min_idle_ms: int, count: int = 10) -> list[StuckEntry]:
    """Entries a consumer claimed and has not finished within `min_idle_ms`.

    Set the threshold above a healthy frame's processing time — anything above it
    is a worker that died holding work, not a worker that is merely busy.
    """
    raw_entries = client.xpending_range(
        stream_name, group, min="-", max="+", count=count, idle=min_idle_ms,
    )
    return [
        StuckEntry(
            entry_id=_text(entry["message_id"]),
            consumer=_text(entry["consumer"]),
            idle_ms=int(entry["time_since_delivered"]),
            delivery_count=int(entry["times_delivered"]),
        )
        for entry in raw_entries
    ]
