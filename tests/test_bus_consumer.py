from datetime import datetime, timezone
import pytest
from edgecv.bus.consumer import FrameConsumer
from edgecv.bus.producer import FrameProducer
from edgecv.contracts.frame import FrameEnvelope

pytestmark = pytest.mark.integration

RUN_ID = "11111111-1111-1111-1111-111111111111"


def env(seq: int) -> FrameEnvelope:
    return FrameEnvelope(
        run_id=RUN_ID, seq=seq,
        captured_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        width=256, height=256, source_ref=f"img_{seq}.png",
        sha256=f"{seq}".ljust(64, "0"), transport="reference",
        path=f"/data/img_{seq}.png",
    )


@pytest.fixture
def producer(rds):
    return FrameProducer(rds, stream="frames", maxlen=100)


def consumer(rds, name="c1"):
    c = FrameConsumer(rds, stream="frames", group="workers", consumer=name,
                      block_ms=100)
    c.ensure_group()
    return c


def test_ensure_group_creates_stream_and_group(rds):
    consumer(rds)
    groups = rds.xinfo_groups("frames")
    assert [g["name"] for g in groups] == [b"workers"]


def test_ensure_group_is_idempotent(rds):
    consumer(rds)
    consumer(rds)          # must not raise BUSYGROUP


def test_read_returns_envelopes(rds, producer):
    producer.publish(env(1))
    producer.publish(env(2))
    batch = consumer(rds).read(count=10)
    assert [e.seq for _entry_id, e in batch] == [1, 2]


def test_read_returns_empty_when_stream_empty(rds):
    assert consumer(rds).read(count=10) == []


def test_ack_removes_entry_from_stream(rds, producer):
    producer.publish(env(1))
    c = consumer(rds)
    entry_id, _e = c.read(count=1)[0]
    c.ack(entry_id)
    assert rds.xlen("frames") == 0


def test_unacked_entry_stays_pending(rds, producer):
    producer.publish(env(1))
    c = consumer(rds)
    c.read(count=1)
    assert rds.xpending("frames", "workers")["pending"] == 1


def test_two_consumers_split_the_work(rds, producer):
    for seq in range(4):
        producer.publish(env(seq))
    a = consumer(rds, "a").read(count=2)
    b = consumer(rds, "b").read(count=2)
    assert {e.seq for _i, e in a}.isdisjoint({e.seq for _i, e in b})
    assert len(a) + len(b) == 4


def test_reclaim_recovers_a_dead_consumers_frames(rds, producer):
    producer.publish(env(1))
    dead = consumer(rds, "dead")
    dead.read(count=1)                       # read but never ack — consumer "dies"
    survivor = consumer(rds, "survivor")
    outcome = survivor.reclaim(min_idle_ms=0, count=10)
    assert [e.seq for _i, e in outcome.deliveries] == [1]
    assert outcome.lost_entry_ids == []      # nothing was destroyed, only orphaned
    assert outcome.progressed is True


def test_reclaim_ignores_recently_delivered_frames(rds, producer):
    producer.publish(env(1))
    consumer(rds, "busy").read(count=1)
    survivor = consumer(rds, "survivor")
    outcome = survivor.reclaim(min_idle_ms=60_000, count=10)
    assert outcome.deliveries == []
    assert outcome.lost_entry_ids == []
    assert outcome.progressed is False       # a genuinely idle pass


def test_reclaim_reports_entries_destroyed_under_the_pel(rds, producer):
    """A pending entry whose stream entry is gone is a frame lost in flight.

    This is what Redis's own XADD MAXLEN / XTRIM would do to in-flight work, and
    what the drop policy exists to avoid. XAUTOCLAIM cannot recover the data —
    only the bookkeeping — and it reports the ids it purged as its third return
    value. Simulated here with XDEL, which leaves the PEL in the identical state.
    """
    producer.publish(env(1))
    dead = consumer(rds, "dead")
    entry_id, _e = dead.read(count=1)[0]
    rds.xdel("frames", entry_id)             # entry destroyed while still pending
    assert rds.xpending("frames", "workers")["pending"] == 1   # PEL still claims it

    survivor = consumer(rds, "survivor")
    outcome = survivor.reclaim(min_idle_ms=0, count=10)

    assert outcome.deliveries == []          # the payload is unrecoverable
    assert outcome.lost_entry_ids == [entry_id]
    assert survivor.lost_in_flight == 1
    assert rds.xpending("frames", "workers")["pending"] == 0   # PEL now honest


def test_reclaim_progressed_when_only_destroyed_entries_were_purged(rds, producer):
    """Purging tombstones is progress, so a drain loop must not stop on it.

    Recovering nothing and finding nothing are different outcomes. Treating them
    the same is how live orphans queued behind a page of deletions get stranded.
    """
    for seq in range(2):
        producer.publish(env(seq))
    dead = consumer(rds, "dead")
    ids = [entry_id for entry_id, _e in dead.read(count=2)]
    rds.xdel("frames", *ids)

    outcome = consumer(rds, "survivor").reclaim(min_idle_ms=0, count=10)
    assert outcome.deliveries == []
    assert sorted(outcome.lost_entry_ids) == sorted(ids)
    assert outcome.progressed is True


def test_reclaim_separates_recoverable_from_destroyed_in_one_pass(rds, producer):
    """One XAUTOCLAIM page can carry both, and the two must not be conflated."""
    producer.publish(env(1))
    producer.publish(env(2))
    dead = consumer(rds, "dead")
    deliveries = dead.read(count=2)
    doomed_id = deliveries[0][0]
    rds.xdel("frames", doomed_id)

    outcome = consumer(rds, "survivor").reclaim(min_idle_ms=0, count=10)
    assert [e.seq for _i, e in outcome.deliveries] == [2]
    assert outcome.lost_entry_ids == [doomed_id]
