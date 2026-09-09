"""Success criterion 1: store and upload at least 100x less data per kilometre
than keeping every frame would.

These tests pin the arithmetic. The inputs are gathered elsewhere -- distance
from the frames table's fixes, stored bytes from the blob store -- so that this
part is pure and can be argued about on paper.
"""
import math

import pytest

from edgecv.bench.bytes_per_km import TARGET_FACTOR, measure


def test_reduction_factor_is_baseline_over_stored():
    r = measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=5_000,
                distance_m=1_000)
    assert r.baseline_bytes == 500_000
    assert r.reduction_factor == pytest.approx(500.0)


def test_per_kilometre_figures_divide_by_the_distance():
    r = measure(stored_bytes=3_000, n_frames=10, mean_frame_bytes=1_000,
                distance_m=2_000)
    assert r.distance_km == pytest.approx(2.0)
    assert r.stored_bytes_per_km == pytest.approx(1_500.0)
    assert r.baseline_bytes_per_km == pytest.approx(5_000.0)


def test_the_ratio_is_independent_of_the_distance():
    """Both sides are per-kilometre, so the headline factor must not move when
    only the denominator changes. Guards against dividing one side twice."""
    near = measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=5_000,
                   distance_m=500)
    far = measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=5_000,
                  distance_m=50_000)
    assert near.reduction_factor == pytest.approx(far.reduction_factor)


def test_exactly_the_target_factor_meets_the_criterion():
    r = measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=1_000,
                distance_m=1_000)
    assert r.reduction_factor == pytest.approx(TARGET_FACTOR)
    assert r.meets_target is True


def test_just_under_the_target_factor_fails_the_criterion():
    r = measure(stored_bytes=1_010, n_frames=100, mean_frame_bytes=1_000,
                distance_m=1_000)
    assert r.reduction_factor < TARGET_FACTOR
    assert r.meets_target is False


def test_storing_nothing_gives_an_infinite_factor_rather_than_crashing():
    """A drive down a clean road flags no defects and stores no crops. That is a
    real run, and it must report rather than raise -- but an infinite factor is
    a degenerate measurement, not a triumph, so it stays visible as inf."""
    r = measure(stored_bytes=0, n_frames=100, mean_frame_bytes=1_000,
                distance_m=1_000)
    assert r.reduction_factor == math.inf
    assert r.meets_target is True


def test_zero_distance_is_rejected():
    """No distance means no per-kilometre figure. Returning inf here would put a
    fabricated headline number in front of a marker."""
    with pytest.raises(ValueError, match="distance"):
        measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=1_000,
                distance_m=0.0)


def test_negative_distance_is_rejected():
    with pytest.raises(ValueError, match="distance"):
        measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=1_000,
                distance_m=-5.0)


def test_no_frames_is_rejected():
    with pytest.raises(ValueError, match="frames"):
        measure(stored_bytes=0, n_frames=0, mean_frame_bytes=1_000,
                distance_m=1_000)


def test_the_target_factor_can_be_overridden():
    r = measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=1_000,
                distance_m=1_000, target_factor=200.0)
    assert r.meets_target is False
    assert r.target_factor == pytest.approx(200.0)


def test_the_default_target_is_the_criterion_the_proposal_committed_to():
    assert TARGET_FACTOR == 100.0
