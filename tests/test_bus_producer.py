from datetime import datetime, timezone
import pytest
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


def test_publish_adds_to_stream(rds):
    producer = FrameProducer(rds, stream="frames", maxlen=10)
    assert producer.publish(env(1)) is not None
    assert rds.xlen("frames") == 1


def test_published_envelope_roundtrips(rds):
    producer = FrameProducer(rds, stream="frames", maxlen=10)
    producer.publish(env(42))
    _entry_id, fields = rds.xrange("frames")[0]
    assert FrameEnvelope.from_fields(fields) == env(42)


def test_drops_when_stream_is_full(rds):
    producer = FrameProducer(rds, stream="frames", maxlen=3)
    for seq in range(5):
        producer.publish(env(seq))
    assert rds.xlen("frames") == 3
    assert producer.offered == 5
    assert producer.dropped == 2


def test_drop_returns_none(rds):
    producer = FrameProducer(rds, stream="frames", maxlen=1)
    assert producer.publish(env(1)) is not None
    assert producer.publish(env(2)) is None


def test_drop_counter_persisted_to_redis(rds):
    producer = FrameProducer(rds, stream="frames", maxlen=1)
    producer.publish(env(1))
    producer.publish(env(2))
    assert int(rds.get(f"stats:{RUN_ID}:dropped")) == 1


def test_inline_transport_carries_bytes(rds):
    producer = FrameProducer(rds, stream="frames", maxlen=10)
    inline = FrameEnvelope(
        run_id=RUN_ID, seq=1,
        captured_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        width=2, height=2, source_ref="x.png", sha256="a" * 64,
        transport="inline", payload=b"\x00\x01\x02\xff",
    )
    producer.publish(inline)
    _entry_id, fields = rds.xrange("frames")[0]
    assert FrameEnvelope.from_fields(fields).payload == b"\x00\x01\x02\xff"
