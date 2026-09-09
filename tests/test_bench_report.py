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
