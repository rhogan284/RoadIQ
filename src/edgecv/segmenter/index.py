"""Condition index v0 — a PCI-style deduct subset. PLACEHOLDER for Ilana's R-I3.

In the spirit of ASTM D6433 (PCI): start at 100 and subtract deduct values, where each
distress type's deduct is a function of its DENSITY on the sample unit (here: a 100 m
segment), with diminishing returns as density rises, and multiple distresses combine
sub-additively. What we observe from one camera is surface distress only, so this is a
*subset* of PCI — no roughness, rutting or structural terms — and it does not reproduce
PCI's published deduct curves or its corrected-deduct iteration. Every constant lives here
so the real definition replaces one module.

    density[c]  = Σ (box area / frame area) of accepted class-c instances on the segment
                  ÷ frames assessed on the segment              (mean % of view distressed)
    deduct[c]   = MAX_DEDUCT[c] × log10(1 + 100·density[c]) / 2   (0 at 0 %, 1× at 99 %)
    total       = largest deduct + ½ × the rest                   (sub-additive combination)
    index       = max(0, 100 − total)

Revised 2026-09-27 from a count-per-100 m version, which saturated at 0 on every segment
of the RDD2022 replay: the dataset is selected for damage (66 % of test images are
defective) and ~57 frames land on each 100 m, so every segment carried ~150 instances.
Density is bounded by construction and is also what PCI deducts are a function of.
"""
from __future__ import annotations

from math import log10

#: Deduct at ~100 % density, by RDD class. Potholes are a safety defect, alligator
#: cracking is structural fatigue, single cracks are the least urgent. `other` is never
#: scored.
MAX_DEDUCT = {"D40": 90.0, "D20": 80.0, "D10": 50.0, "D00": 45.0}
GOOD_MIN, FAIR_MIN = 70.0, 40.0
#: Below this much assessed road a score would be one crack's opinion. Report no score.
MIN_COVERAGE_M = 20.0
#: Review split (spec D6): at or above → auto-accepted; below → review queue.
AUTO_ACCEPT_CONF = 0.50


def deduct(defect_class: str, density: float) -> float:
    d = min(max(density, 0.0), 0.99)
    return MAX_DEDUCT.get(defect_class, 0.0) * log10(1 + 100 * d) / 2


def score(area_by_class: dict[str, float], frames_assessed: int,
          coverage_m: float) -> tuple[float | None, str]:
    if coverage_m < MIN_COVERAGE_M or frames_assessed <= 0:
        return None, "insufficient"
    deducts = sorted((deduct(c, a / frames_assessed) for c, a in area_by_class.items()),
                     reverse=True)
    total = (deducts[0] + 0.5 * sum(deducts[1:])) if deducts else 0.0
    idx = max(0.0, 100.0 - total)
    return round(idx, 1), band(idx)


def band(idx: float) -> str:
    return "good" if idx >= GOOD_MIN else "fair" if idx >= FAIR_MIN else "poor"
