"""Success criterion 1 -- storage and bandwidth per kilometre.

    "We store and upload at least 100x less data per kilometre than keeping
     every frame would."   -- C3 proposal, success criteria table, row 1

Three numbers, and the whole Week 6 milestone was blocked on the third:

  1. what we actually store   -- measured off the blob store, which is the thing
                                 holding the bytes (`BlobStore.total_bytes`)
  2. what keeping every frame -- frame count x mean frame size, both observed
     would have cost             from the run rather than assumed
  3. how far we drove         -- great-circle length of the run's GPS fixes

Kept pure on purpose. Gathering the inputs touches Postgres, the filesystem and
a synthetic track; the arithmetic that produces the headline figure does not,
so it can be checked on paper and argued with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: The factor the proposal committed to for success criterion 1.
TARGET_FACTOR = 100.0


@dataclass(frozen=True, slots=True)
class BytesPerKm:
    distance_km: float
    n_frames: int
    stored_bytes: int
    baseline_bytes: int
    stored_bytes_per_km: float
    baseline_bytes_per_km: float
    reduction_factor: float
    target_factor: float
    meets_target: bool


def measure(*, stored_bytes: int, n_frames: int, mean_frame_bytes: float,
            distance_m: float, target_factor: float = TARGET_FACTOR) -> BytesPerKm:
    """Compute the bytes-per-kilometre figures for one survey run.

    `mean_frame_bytes` is the average encoded size of the frames this run
    replayed, so the baseline is what *these* frames would have cost to keep --
    not the proposal's illustrative 500 KB.
    """
    if distance_m <= 0:
        raise ValueError(
            f"distance must be positive to report a per-kilometre figure, "
            f"got {distance_m!r} m -- do the run's frames carry a GPS fix?"
        )
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames!r}")

    distance_km = distance_m / 1_000.0
    baseline_bytes = round(n_frames * mean_frame_bytes)

    # A run that flagged nothing stored nothing. That is a real outcome, not an
    # error, but an infinite factor is a degenerate measurement rather than a
    # good result -- so it is reported as inf and left conspicuous.
    reduction = math.inf if stored_bytes == 0 else baseline_bytes / stored_bytes

    return BytesPerKm(
        distance_km=distance_km,
        n_frames=n_frames,
        stored_bytes=stored_bytes,
        baseline_bytes=baseline_bytes,
        stored_bytes_per_km=stored_bytes / distance_km,
        baseline_bytes_per_km=baseline_bytes / distance_km,
        reduction_factor=reduction,
        target_factor=target_factor,
        meets_target=reduction >= target_factor,
    )
