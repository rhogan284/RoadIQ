"""The demo road network: OSM ways → 100 m segments, and a drive route over them.

Segments are the unit of analysis (spec §5). They are cut here in Python rather than with
`ST_LineSubstring`, so the same cut can be used by feed-sim (no database on the edge side)
and loaded into `segments` by the office side — one cut, two consumers, no drift.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from edgecv.geo import LatLon, haversine_m

DEFAULT_NETWORK = Path(__file__).resolve().parent / "sydney_demo.geojson"
SEGMENT_M = 100.0
#: A tail shorter than this is merged into the previous segment instead of standing
#: alone: a 12 m segment gets a wild condition index from one crack.
MIN_TAIL_M = 30.0


@dataclass(frozen=True, slots=True)
class Way:
    osm_id: int
    name: str
    highway: str
    points: tuple[LatLon, ...]      # (lat, lon)


@dataclass(frozen=True, slots=True)
class Segment:
    road_ref: str                   # osm:<way>:<index> — stable across reloads
    road_name: str
    points: tuple[LatLon, ...]
    length_m: float

    def wkt(self) -> str:
        return "LINESTRING(" + ",".join(f"{lon} {lat}" for lat, lon in self.points) + ")"


def load_ways(path: Path = DEFAULT_NETWORK) -> list[Way]:
    fc = json.loads(Path(path).read_text())
    return [Way(osm_id=f["properties"]["osm_id"], name=f["properties"]["name"],
                highway=f["properties"]["highway"],
                points=tuple((lat, lon) for lon, lat in f["geometry"]["coordinates"]))
            for f in fc["features"]]


def _interp(a: LatLon, b: LatLon, t: float) -> LatLon:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def cut_way(way: Way, seg_m: float = SEGMENT_M) -> list[Segment]:
    pieces: list[list[LatLon]] = [[way.points[0]]]
    lengths = [0.0]
    for a, b in zip(way.points, way.points[1:]):
        d = haversine_m(a, b)
        pos = 0.0
        while d - pos > 1e-9:
            room = seg_m - lengths[-1]
            step = min(room, d - pos)
            pos += step
            pt = _interp(a, b, pos / d)
            pieces[-1].append(pt)
            lengths[-1] += step
            if lengths[-1] >= seg_m - 1e-9 and d - pos > 1e-9:
                pieces.append([pt])
                lengths.append(0.0)
    if len(pieces) > 1 and lengths[-1] < MIN_TAIL_M:
        pieces[-2].extend(pieces[-1][1:])
        lengths[-2] += lengths[-1]
        pieces.pop()
        lengths.pop()
    return [Segment(road_ref=f"osm:{way.osm_id}:{i}", road_name=way.name,
                    points=tuple(p), length_m=round(L, 2))
            for i, (p, L) in enumerate(zip(pieces, lengths)) if len(p) >= 2 and L > 0]


def segments(ways: list[Way]) -> list[Segment]:
    return [s for w in ways for s in cut_way(w)]


def _key(p: LatLon) -> tuple[float, float]:
    return round(p[0], 6), round(p[1], 6)


def build_route(ways: list[Way]) -> list[LatLon]:
    """One continuous drive that covers every way at least once.

    Greedy edge cover: keep taking an untravelled edge from where the vehicle is; when
    there is none, drive (over travelled road) to the nearest node that has one. Not an
    optimal Chinese-postman tour, and it doesn't need to be — the output is a plausible
    survey route, and being deterministic is what matters (same route every replay).
    One-way restrictions are ignored: this is a positional track, not navigation.
    """
    adj: dict[tuple, list[tuple]] = defaultdict(list)
    coords: dict[tuple, LatLon] = {}
    for w in ways:
        for a, b in zip(w.points, w.points[1:]):
            ka, kb = _key(a), _key(b)
            if ka == kb:
                continue
            coords[ka], coords[kb] = a, b
            adj[ka].append(kb)
            adj[kb].append(ka)
    untravelled = {frozenset((a, b)) for a in adj for b in adj[a]}
    start = min(adj)                                       # deterministic
    route, here = [coords[start]], start
    while untravelled:
        nxt = next((n for n in adj[here] if frozenset((here, n)) in untravelled), None)
        if nxt is not None:
            untravelled.discard(frozenset((here, nxt)))
            route.append(coords[nxt])
            here = nxt
            continue
        # BFS over the road graph to the nearest node with untravelled road.
        prev, q, found = {here: None}, deque([here]), None
        while q:
            n = q.popleft()
            if any(frozenset((n, m)) in untravelled for m in adj[n]):
                found = n
                break
            for m in adj[n]:
                if m not in prev:
                    prev[m] = n
                    q.append(m)
        if found is None:                                  # disconnected component
            found = next(iter(next(iter(untravelled))))
            route.append(coords[found])
        else:
            path = []
            while found is not None and found != here:
                path.append(found)
                found = prev[found]
            route.extend(coords[n] for n in reversed(path))
        here = _key(route[-1])
    return route
