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
    reclaimed = survivor.reclaim(min_idle_ms=0, count=10)
    assert [e.seq for _i, e in reclaimed] == [1]


def test_reclaim_ignores_recently_delivered_frames(rds, producer):
    producer.publish(env(1))
    consumer(rds, "busy").read(count=1)
    survivor = consumer(rds, "survivor")
    assert survivor.reclaim(min_idle_ms=60_000, count=10) == []
