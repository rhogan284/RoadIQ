"""A synthetic GPS track, standing in for the phone's receiver.

Why this exists: the generated fixtures carry no position, so every frame
reached Postgres with `lat IS NULL`. Bytes-per-kilometre has no denominator
without a distance, which is what blocked the Week 6 milestone. Week 5 put two
options on the table -- generate a synthetic track, or wait for Dexter's real
capture manifest -- and this is the synthetic one, chosen so the measurement
harness stops depending on another person's component.

Constant bearing, constant speed, no positional noise. That is not an attempt
at realism and it should not be "improved" into a wandering track: the point of
a measurement rig is a denominator that is analytically known, so that error in
the headline figure is attributable to the numerator. `gps_accuracy_m` is
reported because contract 1 carries it and the coverage view reads it, but it
is a declared constant, not scatter applied to the coordinates.

The path is a rhumb line (constant compass bearing), not a great circle,
because that is what holding a heading actually traces -- and it makes a due-east
drive hold its latitude exactly instead of drifting a few nanodegrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, log, pi, radians, sin, tan

from edgecv.geo import EARTH_RADIUS_M, LatLon, path_length_m

#: A phone's horizontal accuracy with a clear sky view, metres.
DEFAULT_ACCURACY_M = 5.0

#: Below this projected-latitude delta the rhumb formula's q term is
#: ill-conditioned, so fall back to the limit value cos(lat).
_MERIDIONAL_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Fix:
    """One position report, shaped to contract 1's optional position fields."""
    lat: float
    lon: float
    heading_deg: float
    speed_mps: float
    gps_accuracy_m: float


class SyntheticTrack:
    def __init__(self, *, start: LatLon, bearing_deg: float, speed_mps: float,
                 fps: float, accuracy_m: float = DEFAULT_ACCURACY_M) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self.start = start
        self.bearing_deg = bearing_deg
        self.speed_mps = speed_mps
        self.fps = fps
        self.accuracy_m = accuracy_m

    def _advance(self, distance_m: float) -> LatLon:
        """Rhumb-line destination `distance_m` along the bearing from the start.

        Solved from the start point every time rather than stepped from the
        previous fix: stepping accumulates floating-point drift over a long run,
        and the drift lands in the denominator of the headline figure.
        """
        lat1, lon1 = radians(self.start[0]), radians(self.start[1])
        theta = radians(self.bearing_deg)
        delta = distance_m / EARTH_RADIUS_M

        dlat = delta * cos(theta)
        lat2 = lat1 + dlat

        # Inverse Gudermannian (Mercator) latitudes, whose difference is what a
        # constant bearing is constant *against*.
        dpsi = log(tan(lat2 / 2 + pi / 4) / tan(lat1 / 2 + pi / 4))
        q = dlat / dpsi if abs(dpsi) > _MERIDIONAL_EPS else cos(lat1)

        dlon = delta * sin(theta) / q
        lon2 = lon1 + dlon
        # Normalise into [-180, 180) so a track crossing the antimeridian
        # doesn't report a longitude of 190.
        return degrees(lat2), (degrees(lon2) + 540) % 360 - 180

    def fix_for(self, seq: int) -> Fix:
        """The fix for frame `seq`, counting from 0 at the start point."""
        travelled_m = self.speed_mps * (seq / self.fps)
        lat, lon = self._advance(travelled_m)
        return Fix(lat=lat, lon=lon, heading_deg=self.bearing_deg,
                   speed_mps=self.speed_mps, gps_accuracy_m=self.accuracy_m)

    def fixes(self, n_frames: int) -> list[Fix]:
        """Fixes for frames 0..n_frames inclusive."""
        return [self.fix_for(seq) for seq in range(n_frames + 1)]

    def distance_m(self, n_frames: int) -> float:
        """Distance covered by the time frame `n_frames` is captured.

        Measured through the generated fixes with `path_length_m`, not returned
        from `speed x time`. Returning the analytic value would make this agree
        with itself by construction; measuring it the same way the harness
        measures a real run is what makes the number worth reporting.
        """
        return path_length_m([(f.lat, f.lon) for f in self.fixes(n_frames)])
