"""Success criterion 2, second half — drops as metres of road not assessed.

    "Any frame we drop is counted and shown as a gap in the survey, measured in
     metres of road not assessed."   -- C3 proposal, success criteria table, row 2

A dropped frame has no database row and no position of its own. But the frames
either side of it do, so the hole is measurable as the distance between its
neighbours — which means this needs no knowledge of the track that produced the
run and works identically on a real drive.

Two absences that look alike in SQL and mean opposite things:

  no row for a seq      the frame was dropped or lost. That road was never
                        assessed. This is a GAP.
  row with lat NULL     the frame was processed, so coverage is proved; we just
                        cannot place it. The distance across it is still
                        assessed, interpolated from its positioned neighbours.
                        Calling this a gap would understate coverage we can
                        actually evidence.

Why the drop policy this measures is the right one: `FrameProducer` refuses the
*newest* frame when the buffer is full, rather than using Redis's own
`XADD MAXLEN`/`XTRIM`, which evicts the *oldest*. Trimming does not respect the
pending-entries list — it will delete entries a worker has claimed and not yet
acknowledged, leaving dangling PEL references that `XPENDING` still reports as
in flight. A refused frame is a known gap. A trimmed in-flight frame is an
unknown one, and no component-level test can see it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from edgecv.geo import haversine_m

#: (seq, lat|None, lon|None)
Row = tuple[int, float | None, float | None]


@dataclass(frozen=True, slots=True)
class Gap:
    after_seq: int
    before_seq: int
    missing_frames: int
    metres: float


@dataclass(frozen=True, slots=True)
class CoverageReport:
    frames: int
    frames_without_fix: int
    missing_frames: int
    assessed_m: float
    gap_m: float
    gaps: list[Gap] = field(default_factory=list)

    @property
    def attempted_m(self) -> float:
        """Road we set out to assess: what we covered plus what we lost."""
        return self.assessed_m + self.gap_m

    @property
    def gap_count(self) -> int:
        return len(self.gaps)

    @property
    def largest_gap_m(self) -> float:
        """A single 50 m hole matters more than fifty 1 m holes."""
        return max((g.metres for g in self.gaps), default=0.0)

    @property
    def coverage_pct(self) -> float:
        """A run that covered no distance has nothing missing, so it is 100%.
        Reporting 0% for an empty run would read as a total failure."""
        if self.attempted_m == 0:
            return 100.0
        return 100.0 * self.assessed_m / self.attempted_m


def coverage_from_rows(rows: list[Row]) -> CoverageReport:
    rows = sorted(rows, key=lambda r: r[0])
    present = {seq for seq, _lat, _lon in rows}
    positioned = [(seq, lat, lon) for seq, lat, lon in rows
                  if lat is not None and lon is not None]

    assessed_m = 0.0
    gap_m = 0.0
    gaps: list[Gap] = []

    for (seq_a, lat_a, lon_a), (seq_b, lat_b, lon_b) in zip(positioned,
                                                            positioned[1:]):
        span_m = haversine_m((lat_a, lon_a), (lat_b, lon_b))
        absent = [s for s in range(seq_a + 1, seq_b) if s not in present]
        if absent:
            gap_m += span_m
            gaps.append(Gap(after_seq=seq_a, before_seq=seq_b,
                            missing_frames=len(absent), metres=span_m))
        else:
            # Contiguous, or bridged only by present-but-unpositioned frames.
            assessed_m += span_m

    return CoverageReport(
        frames=len(rows),
        frames_without_fix=len(rows) - len(positioned),
        missing_frames=sum(g.missing_frames for g in gaps),
        assessed_m=assessed_m,
        gap_m=gap_m,
        gaps=gaps,
    )
