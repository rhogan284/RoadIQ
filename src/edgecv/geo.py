"""Great-circle distance on a spherical earth.

Deliberately not PostGIS. The frames table stores position as plain floats
(migration 001, R-R1 hybrid storage), so the survey distance that the
bytes-per-kilometre figure divides by is computed in Python from those floats.
PostGIS stays on `segments` and `defect_instances`, where geometry is the point.

A sphere is accurate to about 0.5% against WGS84, which is far inside the
tolerance that matters here: the denominator is a synthetic track whose length
we choose, and a half-percent error on it moves a 100x reduction factor to
100.5x. Using an ellipsoid would be false precision.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Iterable, Sequence

#: Mean earth radius, metres (IUGG).
EARTH_RADIUS_M = 6_371_008.8

LatLon = tuple[float, float]


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance between two (lat, lon) pairs, in metres."""
    lat1, lon1 = radians(a[0]), radians(a[1])
    lat2, lon2 = radians(b[0]), radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(h))


def path_length_m(points: Iterable[LatLon]) -> float:
    """Total length of the polyline through `points`, in metres.

    Fewer than two points is zero distance, not an error: a run that captured a
    single frame really did travel no measurable distance.
    """
    pts: Sequence[LatLon] = list(points)
    return sum(haversine_m(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
