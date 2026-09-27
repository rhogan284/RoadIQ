"""Consumer-group observability — XINFO, XPENDING, and the lag counter.

The chaos test proves recovery works. Nothing until now let us *watch* it on a
running system: we asserted the property in a test and then ran blind.
"""
import time

import pytest

from edgecv.bus.observe import consumer_health, group_health, stuck_entries

pytestmark = pytest.mark.integration

STREAM = "obs_frames"
GROUP = "workers"


@pytest.fixture
def stream(rds):
    for i in range(6):
        rds.xadd(STREAM, {"n": str(i).encode()})
    rds.xgroup_create(STREAM, GROUP, id="0")
    return rds


def test_a_fresh_group_is_behind_by_the_whole_stream(stream):
    h = group_health(stream, stream_name=STREAM, group=GROUP)
    assert h.stream_length == 6
    assert h.lag == 6
    assert h.lag_known is True
    assert h.pending == 0
    assert h.consumers == 0


def test_delivering_entries_moves_lag_and_pending(stream):
    stream.xreadgroup(GROUP, "w1", {STREAM: ">"}, count=2)
    h = group_health(stream, stream_name=STREAM, group=GROUP)
    assert h.entries_read == 2
    assert h.lag == 4
    assert h.pending == 2
    assert h.consumers == 1
    assert h.last_delivered_id != "0-0"


def test_acking_clears_pending_without_moving_lag(stream):
    entries = stream.xreadgroup(GROUP, "w1", {STREAM: ">"}, count=2)
    ids = [entry_id for _s, entry_list in entries for entry_id, _f in entry_list]
    stream.xack(STREAM, GROUP, *ids)
    h = group_health(stream, stream_name=STREAM, group=GROUP)
    assert h.pending == 0
    assert h.lag == 4          # lag is about delivery, not completion


def test_deleting_an_undelivered_entry_makes_lag_unknown(stream):
    """The failure mode that matters: the panel must say "unknown", not "0".
    Proven separately in _evidence/2026-09-06-t1-lag-counter-invalidation.txt."""
    stream.xreadgroup(GROUP, "w1", {STREAM: ">"}, count=2)
    newest = stream.xrevrange(STREAM, count=1)[0][0]
    stream.xdel(STREAM, newest)
    h = group_health(stream, stream_name=STREAM, group=GROUP)
    assert h.lag is None
    assert h.lag_known is False


def test_consumer_health_reports_each_worker_and_its_backlog(stream):
    stream.xreadgroup(GROUP, "w1", {STREAM: ">"}, count=2)
    stream.xreadgroup(GROUP, "w2", {STREAM: ">"}, count=1)
    by_name = {c.name: c for c in consumer_health(stream, stream_name=STREAM,
                                                  group=GROUP)}
    assert set(by_name) == {"w1", "w2"}
    assert by_name["w1"].pending == 2
    assert by_name["w2"].pending == 1
    assert by_name["w1"].idle_ms >= 0


def test_stuck_entries_finds_work_nobody_is_finishing(stream):
    stream.xreadgroup(GROUP, "w1", {STREAM: ">"}, count=2)
    time.sleep(0.05)
    stuck = stuck_entries(stream, stream_name=STREAM, group=GROUP,
                          min_idle_ms=10)
    assert len(stuck) == 2
    assert stuck[0].consumer == "w1"
    assert stuck[0].idle_ms >= 10
    assert stuck[0].delivery_count == 1


def test_stuck_entries_ignores_work_that_is_merely_in_progress(stream):
    stream.xreadgroup(GROUP, "w1", {STREAM: ">"}, count=2)
    assert stuck_entries(stream, stream_name=STREAM, group=GROUP,
                         min_idle_ms=60_000) == []


def test_an_unknown_group_is_a_clear_error_not_an_empty_reading(stream):
    with pytest.raises(LookupError, match="nosuchgroup"):
        group_health(stream, stream_name=STREAM, group="nosuchgroup")
