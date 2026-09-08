"""Frame-envelope construction, including the GPS fix.

Split out from test_feedsim_main.py because that module is integration-marked
(it needs Redis) and this needs nothing -- the position wiring is exactly the
part worth being able to test without docker running.
"""
import hashlib

import pytest

from edgecv.feedsim.gpstrack import SyntheticTrack
from edgecv.feedsim.main import build_envelope

SYD = (-33.8688, 151.2093)
DATA = b"not-really-a-png"
SHA = hashlib.sha256(DATA).hexdigest()


def _envelope(fix=None):
    return build_envelope(run_id="11111111-1111-1111-1111-111111111111", seq=3,
                          path="fixtures/defect_000.png", data=DATA,
                          width=64, height=48, transport="reference", fix=fix)


def test_envelope_without_a_fix_carries_no_position():
    """The pre-GPS behaviour has to keep working: contract 1 says absent means
    absent, and the coverage view counts frames_without_fix off exactly this."""
    env = _envelope()
    assert env.lat is None and env.lon is None
    assert env.speed_mps is None
    assert env.sha256 == SHA


def test_envelope_with_a_fix_carries_the_position_fields():
    track = SyntheticTrack(start=SYD, bearing_deg=90.0, speed_mps=13.89, fps=10.0)
    env = _envelope(fix=track.fix_for(3))
    assert env.lat == pytest.approx(SYD[0])
    assert env.lon > SYD[1]
    assert env.speed_mps == pytest.approx(13.89)
    assert env.heading_deg == pytest.approx(90.0)
    assert env.gps_accuracy_m > 0


def test_position_survives_a_round_trip_through_the_redis_fields():
    """The fix is only useful if it reaches the worker, and to_fields/from_fields
    is the wire. A fix that doesn't survive this is a null in the frames table."""
    from edgecv.contracts.frame import FrameEnvelope

    track = SyntheticTrack(start=SYD, bearing_deg=42.0, speed_mps=8.5, fps=10.0)
    env = _envelope(fix=track.fix_for(17))
    back = FrameEnvelope.from_fields(env.to_fields())
    assert back.lat == pytest.approx(env.lat)
    assert back.lon == pytest.approx(env.lon)
    assert back.speed_mps == pytest.approx(env.speed_mps)
    assert back.heading_deg == pytest.approx(env.heading_deg)
    assert back.gps_accuracy_m == pytest.approx(env.gps_accuracy_m)
