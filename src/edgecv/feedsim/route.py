"""A GPS track that follows real streets, for dataset replay.

Same interface as `SyntheticTrack` (`fix_for`, `fixes`, `distance_m`), so `run_feed` does
not care which it gets. The straight rhumb line stays the measurement rig for bytes-per-km;
this one exists so frames land on real road segments and the condition map has something
to colour. Constant speed, no positional noise, for the same reason as the rhumb track:
the error in anything downstream should be attributable to the pipeline, not the fixture.

If the run is longer than the route, the vehicle goes round again — a second lap is a
re-survey of the same segments, which is honest about what happened.
"""
from __future__ import annotations

from bisect import bisect_right
from math import atan2, cos, degrees, radians, sin
from pathlib import Path

from edgecv.feedsim.gpstrack import DEFAULT_ACCURACY_M, Fix
from edgecv.geo import LatLon, haversine_m
from edgecv.roads import DEFAULT_NETWORK, build_loop, build_random_route, build_route, load_ways


def _bearing(a: LatLon, b: LatLon) -> float:
    la, lb = radians(a[0]), radians(b[0])
    dl = radians(b[1] - a[1])
    y = sin(dl) * cos(lb)
    x = cos(la) * sin(lb) - sin(la) * cos(lb) * cos(dl)
    return (degrees(atan2(y, x)) + 360) % 360


class RouteTrack:
    def __init__(self, route: list[LatLon], *, speed_mps: float, fps: float,
                 accuracy_m: float = DEFAULT_ACCURACY_M) -> None:
        if fps <= 0 or len(route) < 2:
            raise ValueError("need fps > 0 and a route of at least two points")
        self.route = route
        self.speed_mps = speed_mps
        self.fps = fps
        self.accuracy_m = accuracy_m
        self._cum = [0.0]
        for a, b in zip(route, route[1:]):
            self._cum.append(self._cum[-1] + haversine_m(a, b))
        self.length_m = self._cum[-1]

    @classmethod
    def from_network(cls, path: Path = DEFAULT_NETWORK, *, mode: str = "random",
                     n_frames: int | None = None, seed: int = 0, **kw) -> RouteTrack:
        """mode="random": a fresh randomised drive over the council streets, sized so
        `n_frames` at this speed and fps end as the route does (no second lap).
        mode="loop": the fixed main-road survey loop (closed, so laps join up).
        mode="cover": the greedy every-street route (has jumps between road pieces)."""
        ways = load_ways(path)
        if mode == "random":
            if n_frames is None:
                raise ValueError("random routes are sized from n_frames")
            target = n_frames * kw["speed_mps"] / kw["fps"] + 20.0
            return cls(build_random_route(ways, target_m=target, seed=seed), **kw)
        return cls(build_loop(ways) if mode == "loop" else build_route(ways), **kw)

    def fix_for(self, seq: int) -> Fix:
        d = (self.speed_mps * seq / self.fps) % self.length_m
        i = min(bisect_right(self._cum, d) - 1, len(self.route) - 2)
        a, b = self.route[i], self.route[i + 1]
        span = self._cum[i + 1] - self._cum[i]
        t = (d - self._cum[i]) / span if span else 0.0
        return Fix(lat=a[0] + (b[0] - a[0]) * t, lon=a[1] + (b[1] - a[1]) * t,
                   heading_deg=round(_bearing(a, b), 1), speed_mps=self.speed_mps,
                   gps_accuracy_m=self.accuracy_m)

    def fixes(self, n_frames: int) -> list[Fix]:
        return [self.fix_for(seq) for seq in range(n_frames + 1)]

    def distance_m(self, n_frames: int) -> float:
        # Measured through the fixes, like SyntheticTrack. A lap wrap jumps back to the
        # start, so sum per step and skip the wrap step rather than measure the jump.
        fx = self.fixes(n_frames)
        step = self.speed_mps / self.fps
        return sum(d for d in (haversine_m((a.lat, a.lon), (b.lat, b.lon))
                               for a, b in zip(fx, fx[1:])) if d <= step * 1.5)

