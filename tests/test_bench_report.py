"""Formatting of the bytes-per-kilometre report.

Worth a test because the report *is* the milestone deliverable: a figure quoted
in the wrong unit in the final report is a wrong claim, not a cosmetic bug.
"""
from edgecv.bench.main import human_bytes


def test_bytes_below_a_kilobyte_are_shown_as_bytes():
    assert human_bytes(512) == "512 B"


def test_kilobytes_use_binary_units():
    assert human_bytes(2048) == "2.0 KiB"


def test_megabytes_use_binary_units():
    assert human_bytes(5 * 1024 * 1024) == "5.0 MiB"


def test_gigabytes_use_binary_units():
    assert human_bytes(3 * 1024 ** 3) == "3.0 GiB"


def test_zero_is_shown_as_bytes():
    assert human_bytes(0) == "0 B"


# --- the report has to carry the coverage figures too -----------------------

def _report(**overrides):
    from edgecv.bench.bytes_per_km import measure
    from edgecv.bench.coverage import CoverageReport, Gap
    from edgecv.bench.main import format_report

    result = measure(stored_bytes=1_000, n_frames=100, mean_frame_bytes=5_000,
                     distance_m=1_000)
    coverage = CoverageReport(
        frames=99, frames_without_fix=0, missing_frames=1,
        assessed_m=980.0, gap_m=20.0,
        gaps=[Gap(after_seq=49, before_seq=51, missing_frames=1, metres=20.0)],
    )
    return format_report("run-1", result, coverage=coverage,
                         store_root="/blobs", **overrides)


def test_report_states_metres_of_road_not_assessed():
    text = _report()
    assert "not assessed" in text
    assert "20" in text


def test_report_states_the_worst_single_gap():
    assert "largest gap" in _report().lower()


def test_report_shows_coverage_as_a_percentage():
    assert "98.0%" in _report()
