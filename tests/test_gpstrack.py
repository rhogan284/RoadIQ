"""The synthetic GPS track that unblocks the Week 6 milestone.

The fixtures carry no GPS fix, so there was no distance to divide by and
bytes-per-kilometre could not be computed at all. This stands in for the
phone's receiver until Dexter's capture rig lands.

Constant bearing and constant speed on purpose: the distance is then
analytically known (speed x duration), so any error in the bytes-per-km figure
is attributable to the bytes, not to the denominator. A realistically wandering
track would be worse here, not better -- a measurement harness wants a ruler it
can trust, not a plausible one.
"""
import pytest

from edgecv.feedsim.gpstrack import Fix, SyntheticTrack
from edgecv.geo import haversine_m

SYD = (-33.8688, 151.2093)


@pytest.fixture
def track():
    # 13.89 m/s is 50 km/h -- an urban survey speed. 10 fps is the rate
    # success criterion 2 commits to.
    return SyntheticTrack(start=SYD, bearing_deg=90.0, speed_mps=13.89, fps=10.0)


def test_first_fix_is_at_the_start_point(track):
    fix = track.fix_for(0)
    assert (fix.lat, fix.lon) == pytest.approx(SYD)


def test_distance_to_frame_n_is_speed_times_elapsed_time(track):
    """The property the whole measurement rests on."""
    n = 600  # one minute at 10 fps
    expected_m = 13.89 * (n / 10.0)
    actual_m = haversine_m(SYD, (track.fix_for(n).lat, track.fix_for(n).lon))
    assert actual_m == pytest.approx(expected_m, rel=1e-4)


def test_consecutive_fixes_are_one_frame_interval_apart(track):
    a = track.fix_for(41)
    b = track.fix_for(42)
    step_m = haversine_m((a.lat, a.lon), (b.lat, b.lon))
    assert step_m == pytest.approx(13.89 / 10.0, rel=1e-4)


def test_due_east_bearing_holds_latitude_and_increases_longitude(track):
    fix = track.fix_for(100)
    assert fix.lat == pytest.approx(SYD[0], abs=1e-9)
    assert fix.lon > SYD[1]


def test_due_north_bearing_holds_longitude_and_increases_latitude():
    north = SyntheticTrack(start=SYD, bearing_deg=0.0, speed_mps=10.0, fps=10.0)
    fix = north.fix_for(100)
    assert fix.lon == pytest.approx(SYD[1], abs=1e-9)
    assert fix.lat > SYD[0]


def test_every_fix_reports_speed_heading_and_accuracy(track):
    fix = track.fix_for(7)
    assert isinstance(fix, Fix)
    assert fix.speed_mps == pytest.approx(13.89)
    assert fix.heading_deg == pytest.approx(90.0)
    assert fix.gps_accuracy_m > 0


def test_path_length_over_n_frames_matches_analytic_distance(track):
    assert track.distance_m(600) == pytest.approx(13.89 * 60.0, rel=1e-4)


def test_distance_over_zero_frames_is_zero(track):
    assert track.distance_m(0) == 0.0


def test_a_stationary_track_covers_no_distance():
    parked = SyntheticTrack(start=SYD, bearing_deg=90.0, speed_mps=0.0, fps=10.0)
    assert parked.distance_m(1000) == pytest.approx(0.0)


def test_rejects_a_non_positive_frame_rate():
    with pytest.raises(ValueError, match="fps"):
        SyntheticTrack(start=SYD, bearing_deg=0.0, speed_mps=10.0, fps=0.0)
