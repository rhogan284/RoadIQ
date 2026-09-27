"""Great-circle distance on a spherical earth.

Deliberately not PostGIS. The frames table stores position as plain floats
(migration 001, R-R1 hybrid storage), so frame-to-frame distances (coverage gaps,
route lengths) are computed in Python from those floats. PostGIS stays on
`segments` and `defect_instances`, where geometry is the point.

A sphere is accurate to about 0.5% against WGS84, far inside any tolerance that
matters here. Using an ellipsoid would be false precision.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

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

