"""Success criterion 2's second half: "Any frame we drop is counted and shown as a
gap in the survey, measured in metres of road not assessed."

Only computable since frames started carrying a GPS fix. A dropped frame has no
database row and no position of its own, but the frames either side of it do, so
the distance across the hole is measurable without knowing anything about the
track that generated it -- which means this works on a real drive too.
"""
import pytest

from edgecv.bench.coverage import coverage_from_rows

# ~1.389 m apart: 13.89 m/s at 10 fps. Built from the track so the numbers are
# checkable by hand.
from edgecv.feedsim.gpstrack import SyntheticTrack

SYD = (-33.8688, 151.2093)
TRACK = SyntheticTrack(start=SYD, bearing_deg=90.0, speed_mps=13.89, fps=10.0)
STEP_M = 1.389


def rows(*specs):
    """specs are (seq, positioned) pairs, in seq order."""
    out = []
    for seq, positioned in specs:
        fix = TRACK.fix_for(seq)
        out.append((seq, fix.lat, fix.lon) if positioned else (seq, None, None))
    return out


def contiguous(n):
    return rows(*[(s, True) for s in range(n)])


def test_a_contiguous_run_has_no_gaps():
    r = coverage_from_rows(contiguous(11))
    assert r.frames == 11
    assert r.missing_frames == 0
    assert r.gap_count == 0
    assert r.gap_m == 0.0
    assert r.assessed_m == pytest.approx(10 * STEP_M, rel=1e-3)


def test_attempted_is_assessed_plus_gap():
    r = coverage_from_rows(contiguous(11))
    assert r.attempted_m == pytest.approx(r.assessed_m + r.gap_m)


def test_one_dropped_frame_becomes_a_gap_in_metres():
    """seq 5 never reached the database. The hole spans seq 4 to seq 6, which is
    two frame intervals of road nobody assessed."""
    r = coverage_from_rows(*[rows(*[(s, True) for s in range(11) if s != 5])])
    assert r.missing_frames == 1
    assert r.gap_count == 1
    assert r.gap_m == pytest.approx(2 * STEP_M, rel=1e-3)
    assert r.assessed_m == pytest.approx(8 * STEP_M, rel=1e-3)


def test_two_separate_gaps_are_counted_separately():
    r = coverage_from_rows(rows(*[(s, True) for s in range(21)
                                  if s not in (5, 6, 15)]))
    assert r.gap_count == 2
    assert r.missing_frames == 3


def test_the_largest_gap_is_reported():
    """A single 50 m hole matters more than fifty 1 m holes, so the worst one is
    reported and not just the total."""
    r = coverage_from_rows(rows(*[(s, True) for s in range(31)
                                  if s not in (5, 20, 21, 22, 23)]))
    assert r.gap_count == 2
    assert r.largest_gap_m == pytest.approx(5 * STEP_M, rel=1e-3)


def test_coverage_pct_is_assessed_over_attempted():
    r = coverage_from_rows(rows(*[(s, True) for s in range(11) if s != 5]))
    assert r.coverage_pct == pytest.approx(100 * 8 / 10, rel=1e-3)


def test_a_frame_present_but_without_a_fix_is_not_a_gap():
    """The row exists, so coverage is proved -- we just don't know where it was.
    The distance across it is still assessed, interpolated from its neighbours.
    Treating it as a gap would understate coverage we can actually evidence."""
    r = coverage_from_rows(rows(*[(s, s != 5) for s in range(11)]))
    assert r.frames == 11
    assert r.frames_without_fix == 1
    assert r.gap_count == 0
    assert r.missing_frames == 0
    assert r.assessed_m == pytest.approx(10 * STEP_M, rel=1e-3)


def test_a_gap_and_an_unpositioned_frame_are_distinguished():
    r = coverage_from_rows(rows(*[(s, s != 3) for s in range(11) if s != 7]))
    assert r.frames_without_fix == 1
    assert r.missing_frames == 1
    assert r.gap_count == 1


def test_a_single_frame_covers_no_distance():
    r = coverage_from_rows(contiguous(1))
    assert r.frames == 1
    assert r.assessed_m == 0.0
    assert r.attempted_m == 0.0
    assert r.coverage_pct == 100.0


def test_no_frames_at_all():
    r = coverage_from_rows([])
    assert r.frames == 0
    assert r.assessed_m == 0.0
    assert r.coverage_pct == 100.0


def test_gaps_carry_the_seqs_either_side_so_they_can_be_investigated():
    r = coverage_from_rows(rows(*[(s, True) for s in range(11) if s != 5]))
    gap = r.gaps[0]
    assert gap.after_seq == 4
    assert gap.before_seq == 6
    assert gap.missing_frames == 1
    assert gap.metres == pytest.approx(2 * STEP_M, rel=1e-3)


def test_rows_are_sorted_by_seq_before_analysis():
    """Defensive: the SQL orders by seq, but a caller passing rows any other way
    would otherwise get silently wrong distances rather than an error."""
    shuffled = rows(*[(s, True) for s in [3, 0, 4, 1, 2]])
    shuffled = [shuffled[1], shuffled[3], shuffled[4], shuffled[0], shuffled[2]]
    r = coverage_from_rows(shuffled)
    assert r.gap_count == 0
    assert r.assessed_m == pytest.approx(4 * STEP_M, rel=1e-3)
