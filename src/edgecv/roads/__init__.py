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

DEFAULT_NETWORK = Path(__file__).resolve().parent / "lackey_road.geojson"
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
    #: False for state arterials: driven, never segmented or scored ("not ours").
    council: bool = True


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
                points=tuple((lat, lon) for lon, lat in f["geometry"]["coordinates"]),
                council=f["properties"].get("council", True))
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
    return [s for w in ways if w.council for s in cut_way(w)]


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


# ---------------------------------------------------------------------------- loop route
#: Cost multiplier per road class: the loop prefers the bigger council roads and uses
#: residential streets only as connectors. (Broadway, Harris St, City Rd and Parramatta Rd
#: are state arterials, excluded from the extract — not the council's to survey.)
CLASS_COST = {"secondary": 1.0, "tertiary": 1.0, "unclassified": 1.4, "residential": 3.0}
#: Arterials connect the council's pieces but are not surveyed, so they cost more than a
#: council main road: the loop uses one only when there is no council way round.
ARTERIAL_COST = 5.0

#: The demo survey loop, clockwise from Glebe Point Road: (lat, lon) waypoints, each
#: snapped to the nearest road node. Chosen on the bigger roads around the extract's edge
#: so the car drives a ring an officer would recognise, not a space-filling scribble.
DEMO_LOOP = (
    (-33.8830, 151.1900),   # Glebe Point Road
    (-33.8795, 151.1905),   # St Johns Road
    (-33.8775, 151.1935),   # Wentworth Park Road
    (-33.8800, 151.2003),   # Systrum Street
    (-33.8829, 151.2000),   # Thomas Street
    (-33.8866, 151.2012),   # Wellington Street
    (-33.8905, 151.1950),   # Shepherd Street (south)
    (-33.8845, 151.1950),   # Bay Street
)


def _graph(ways: list[Way]):
    adj: dict[tuple, list[tuple[tuple, float]]] = defaultdict(list)
    coords: dict[tuple, LatLon] = {}
    _graph.council_nodes = set()
    for w in ways:
        if w.council:
            _graph.council_nodes.update(_key(p) for p in w.points)
        f = CLASS_COST.get(w.highway, 3.0) if w.council else ARTERIAL_COST
        for a, b in zip(w.points, w.points[1:]):
            ka, kb = _key(a), _key(b)
            if ka == kb:
                continue
            coords[ka], coords[kb] = a, b
            cost = haversine_m(a, b) * f
            adj[ka].append((kb, cost))
            adj[kb].append((ka, cost))
    return adj, coords


def _shortest(adj, src, dst) -> list[tuple]:
    import heapq
    dist, prev, heap = {src: 0.0}, {src: None}, [(0.0, src)]
    while heap:
        d, n = heapq.heappop(heap)
        if n == dst:
            break
        if d > dist[n]:
            continue
        for m, c in adj[n]:
            nd = d + c
            if nd < dist.get(m, float("inf")):
                dist[m], prev[m] = nd, n
                heapq.heappush(heap, (nd, m))
    if dst not in prev:
        raise ValueError(f"no road path between {src} and {dst}")
    path, n = [], dst
    while n is not None:
        path.append(n)
        n = prev[n]
    return path[::-1]


def _largest_component(adj) -> list[tuple]:
    seen: set = set()
    best: list[tuple] = []
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for m, _ in adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        if len(comp) > len(best):
            best = comp
    return best


def build_loop(ways: list[Way], waypoints=DEMO_LOOP) -> list[LatLon]:
    """A closed survey loop through `waypoints` along real roads, preferring main roads.

    Closed on purpose: when feed-sim runs past the end it starts the next lap from where
    it already is, so the car never jumps. Back-and-forth spikes (A→B→A, from a waypoint
    that snapped onto a side stub) are removed, so the car never U-turns mid-street.
    """
    adj, coords = _graph(ways)
    # Waypoints go on council roads (they are what is being surveyed) inside the largest
    # connected piece (a waypoint on an island would be unreachable).
    nodes = [n for n in _largest_component(adj) if n in _graph.council_nodes]
    snap = [min(nodes, key=lambda n: haversine_m(coords[n], wp)) for wp in waypoints]
    keys: list[tuple] = [snap[0]]
    for a, b in zip(snap, snap[1:] + snap[:1]):
        keys.extend(_shortest(adj, a, b)[1:])
    out: list[tuple] = []
    for k in keys:
        if len(out) >= 2 and out[-2] == k:
            out.pop()                       # A→B→A: drop the spike
        elif not out or out[-1] != k:
            out.append(k)
    return [coords[k] for k in out]


# ---------------------------------------------------------------------------- random route
#: Dead-end chains up to this long are left out of the drive: a survey vehicle doesn't
#: U-turn out of every short cul-de-sac, and each one would re-drive its own length. Most
#: are stubs where the extract dropped the lane or unnamed road they really connect to.
STUB_MAX_M = 150.0


def _prune_stubs(adj, coords, max_m: float = STUB_MAX_M) -> None:
    """Remove, in place, every dead-end chain (degree-1 node back to the first junction)
    no longer than max_m."""
    changed = True
    while changed:
        changed = False
        for end in [n for n in list(adj) if len(adj[n]) == 1]:
            chain, n, prev, length = [end], end, None, 0.0
            while len(adj[n]) <= 2:
                nxt = [m for m, _ in adj[n] if m != prev]
                if not nxt:
                    break
                length += haversine_m(coords[n], coords[nxt[0]])
                prev, n = n, nxt[0]
                chain.append(n)
                if length > max_m:
                    break
            if length > max_m or len(adj[n]) <= 2:
                continue                        # long street, or an isolated segment
            for a, b in zip(chain, chain[1:]):
                adj[a] = [(m, c) for m, c in adj[a] if m != b]
                adj[b] = [(m, c) for m, c in adj[b] if m != a]
            for x in chain[:-1]:
                if not adj[x]:
                    del adj[x]
            changed = True


def build_random_route(ways: list[Way], *, target_m: float, seed: int) -> list[LatLon]:
    """A randomised survey drive over the council network, side streets included, that is
    `target_m` long and re-drives as little road as it can.

    At each junction the vehicle takes an undriven council street, picked at random but
    weighted towards going straight (a survey driver works along a street, they don't zig-
    zag every block). A dead end is the only place it U-turns. When no undriven street
    leaves the junction, it takes the cheapest path (state roads cost ARTERIAL_COST, so
    they are connectors only) to the nearest junction that has one. Seeded, so a run's
    route can be regenerated from the seed recorded in survey_runs.config.
    """
    import heapq
    import random
    from math import atan2, cos, radians, sin

    rng = random.Random(seed)
    adj, coords = _graph(ways)
    _prune_stubs(adj, coords)
    council_edges: set[frozenset] = set()
    for w in ways:
        if w.council:
            for a, b in zip(w.points, w.points[1:]):
                if _key(a) != _key(b):
                    council_edges.add(frozenset((_key(a), _key(b))))
    comp = set(_largest_component(adj))
    untravelled = {e for e in council_edges if e <= comp}

    def bearing(a, b) -> float:
        (la1, lo1), (la2, lo2) = coords[a], coords[b]
        y = sin(radians(lo2 - lo1)) * cos(radians(la2))
        x = cos(radians(la1)) * sin(radians(la2)) - sin(radians(la1)) * cos(radians(la2)) * cos(radians(lo2 - lo1))
        return atan2(y, x)

    def turn(prev, here, nxt) -> float:
        d = abs(bearing(prev, here) - bearing(here, nxt))
        return min(d, 2 * 3.141592653589793 - d)          # 0 = straight on

    starts = sorted(n for n in comp if any(frozenset((n, m)) in untravelled for m, _ in adj[n]))
    here = rng.choice(starts)
    route, prev, length = [here], None, 0.0

    def step(n) -> None:
        nonlocal here, prev, length
        untravelled.discard(frozenset((here, n)))
        length += haversine_m(coords[here], coords[n])
        route.append(n)
        prev, here = here, n

    while length < target_m:
        options = [m for m, _ in adj[here] if frozenset((here, m)) in untravelled and m != prev]
        if not options and prev is not None and frozenset((here, prev)) in untravelled:
            options = [prev]
        if options:
            def open_from(n, via) -> int:
                """Undriven council edges reachable in one more step: a street that leads
                into undriven road beats one that leads into driven road."""
                return sum(1 for m, _ in adj[n]
                           if m != via and frozenset((n, m)) in untravelled)
            weights = []
            for m in options:
                w = 1.0 + 2.0 * open_from(m, here)
                if prev is not None and turn(prev, here, m) < 0.5:
                    w *= 3.0                              # keep working along the street
                weights.append(w)
            step(rng.choices(options, weights)[0])
            continue
        # Nothing undriven here: cheapest path to the nearest junction that has some.
        dist, back, heap, found = {here: 0.0}, {here: None}, [(0.0, here)], None
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist[n]:
                continue
            if n != here and any(frozenset((n, m)) in untravelled for m, _ in adj[n]):
                found = n
                break
            for m, c in adj[n]:
                # Driven council road costs double on the way to new road, so the car
                # takes an undriven street over a driven one when both get it there.
                e = frozenset((n, m))
                c2 = c * (2.0 if e in council_edges and e not in untravelled else 1.0)
                if d + c2 < dist.get(m, float("inf")):
                    dist[m], back[m] = d + c2, n
                    heapq.heappush(heap, (d + c2, m))
        if found is None:                   # network exhausted: start a fresh pass
            untravelled = {e for e in council_edges if e <= comp}
            continue
        path, n = [], found
        while n != here:
            path.append(n)
            n = back[n]
        for n in reversed(path):
            step(n)
    return [coords[k] for k in route]
