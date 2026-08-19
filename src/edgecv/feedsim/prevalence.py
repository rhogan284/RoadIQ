"""Control the defect rate of the simulated line, and pace frames without drift."""
from __future__ import annotations

import random
from typing import Iterator, Sequence


class PrevalenceSampler:
    """Draw image paths with a target defect prevalence.

    Public defect datasets are near-balanced; production lines are not. Sampling with
    replacement from separate pools is what lets the dashboard show a realistic base
    rate, which the p-chart's control limits depend on.
    """

    def __init__(self, clean: Sequence[str], defective: Sequence[str], *,
                 prevalence: float, rng: random.Random) -> None:
        if not 0.0 <= prevalence <= 1.0:
            raise ValueError("prevalence must be between 0 and 1")
        if prevalence < 1.0 and not clean:
            raise ValueError("clean pool is empty but prevalence < 1")
        if prevalence > 0.0 and not defective:
            raise ValueError("defect pool is empty but prevalence > 0")
        self.clean = list(clean)
        self.defective = list(defective)
        self.prevalence = prevalence
        self.rng = rng

    def next(self) -> tuple[str, bool]:
        is_defect = self.rng.random() < self.prevalence
        pool = self.defective if is_defect else self.clean
        return self.rng.choice(pool), is_defect


def pace_deadlines(*, start: float, fps: float, n: int) -> Iterator[float]:
    """Absolute deadlines for n frames.

    Accumulating on the deadline rather than sleeping a fixed interval prevents the
    feed rate drifting slow as per-frame work varies.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    interval = 1.0 / fps
    for i in range(n):
        yield start + i * interval
