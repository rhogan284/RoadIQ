"""One-time: pull the demo road network from OpenStreetMap and commit it.

    uv run python -m scripts.fetch_osm_roads

Named public roads (residential → secondary) in Ultimo / Chippendale / Glebe, around UTS.
The output is committed (`src/edgecv/roads/sydney_demo.geojson`), so a demo never depends
on the Overpass API being up — it was down twice while this was written.

State arterials (Broadway, Harris St, City Rd, Parramatta Rd …) are included with
`council: false`: the survey vehicle drives them to get between suburbs, but they are not
the council's asset, so they are never cut into segments or scored — the mock-up's
"Not ours" grey. Without them the council network is 20 disconnected pieces.

Lanes, places and courts are dropped: they are real roads, but they are 30–80 m stubs that
turn the map into noise and add little a council would prioritise.

Data © OpenStreetMap contributors, ODbL 1.0 — https://www.openstreetmap.org/copyright
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Moss Vale, NSW bounding box (south, west, north, east)
BBOX = (-34.5600, 150.3600, -34.5300, 150.4000)
HIGHWAYS = "residential|tertiary|secondary|unclassified"
ARTERIALS = "primary|trunk|primary_link|secondary_link|tertiary_link"
DROP_SUFFIXES = (" Lane", " Place", " Court")
MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]
OUT = Path(__file__).resolve().parents[1] / "src/edgecv/roads/sydney_demo.geojson"


def fetch(highways: str, bbox=BBOX) -> list[dict]:
    q = (f'[out:json][timeout:90];way["highway"~"^({highways})$"]["name"]'
         f'({",".join(map(str, bbox))});out geom;')
    body = urllib.parse.urlencode({"data": q}).encode()
    for url in MIRRORS:
        try:
            req = urllib.request.Request(url, body, headers={"User-Agent": "RoadIQ-student/0.1"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)["elements"]
        except (OSError, ValueError) as exc:
            print(f"{url}: {exc}")
            time.sleep(2)
    raise SystemExit("every Overpass mirror failed")


def main() -> None:
    feats = []
    # Arterials from a slightly wider box, so the ring roads around the extract are whole.
    wide = (BBOX[0] - 0.0035, BBOX[1] - 0.004, BBOX[2] + 0.004, BBOX[3] + 0.003)
    ways = [(w, True) for w in fetch(HIGHWAYS)] + [(w, False) for w in fetch(ARTERIALS, wide)]
    for w, council in ways:
        name = w["tags"]["name"]
        if name.endswith(DROP_SUFFIXES) or len(w["geometry"]) < 2:
            continue
        feats.append({"type": "Feature",
                      "properties": {"osm_id": w["id"], "name": name,
                                     "highway": w["tags"]["highway"], "council": council},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(p["lon"], 7), round(p["lat"], 7)]
                                                   for p in w["geometry"]]}})
    feats.sort(key=lambda f: f["properties"]["osm_id"])
    OUT.write_text(json.dumps({"type": "FeatureCollection",
                               "attribution": "© OpenStreetMap contributors, ODbL 1.0",
                               "features": feats}, separators=(",", ":")))
    print(f"wrote {len(feats)} ways to {OUT}")


if __name__ == "__main__":
    main()
