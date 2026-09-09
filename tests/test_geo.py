"""Great-circle helpers. The bytes-per-kilometre denominator depends on these,
so a wrong constant here silently flatters or ruins the headline figure."""
import math

import pytest

from edgecv.geo import haversine_m, path_length_m

# Sydney CBD, near the corner of George and Market St.
SYD = (-33.8688, 151.2093)


def test_distance_between_identical_points_is_zero():
    assert haversine_m(SYD, SYD) == 0.0


def test_one_degree_of_latitude_is_about_111_km():
    # A degree of latitude is ~111.19 km anywhere on a spherical earth.
    north = (SYD[0] + 1.0, SYD[1])
    assert haversine_m(SYD, north) == pytest.approx(111195, rel=1e-3)


def test_distance_is_symmetric():
    north = (SYD[0] + 0.01, SYD[1])
    assert haversine_m(SYD, north) == pytest.approx(haversine_m(north, SYD))


def test_a_degree_of_longitude_shrinks_with_latitude():
    """Guards against swapping lat and lon in the formula: at Sydney's latitude a
    degree of longitude is cos(33.87 deg) ~ 0.83 of a degree of latitude."""
    east = (SYD[0], SYD[1] + 1.0)
    north = (SYD[0] + 1.0, SYD[1])
    ratio = haversine_m(SYD, east) / haversine_m(SYD, north)
    assert ratio == pytest.approx(math.cos(math.radians(SYD[0])), rel=1e-3)


def test_path_length_sums_consecutive_legs():
    a = SYD
    b = (SYD[0] + 0.001, SYD[1])
    c = (SYD[0] + 0.002, SYD[1])
    assert path_length_m([a, b, c]) == pytest.approx(
        haversine_m(a, b) + haversine_m(b, c)
    )


def test_path_length_of_a_single_point_is_zero():
    assert path_length_m([SYD]) == 0.0


def test_path_length_of_no_points_is_zero():
    assert path_length_m([]) == 0.0
