from datetime import datetime, timezone

import pytest

from edgecv.contracts.frame import FrameEnvelope

BOOT = "22222222-2222-2222-2222-222222222222"


def _env(**kw):
    base = dict(
        run_id="11111111-1111-1111-1111-111111111111",
        seq=7,
        captured_at=datetime(2026, 8, 2, 3, 4, 5, tzinfo=timezone.utc),
        width=256, height=256,
        source_ref="fixtures/defect_003.png",
        sha256="a" * 64,
        transport="reference",
        path="/data/defect_003.png",
    )
    base.update(kw)
    return FrameEnvelope(**base)


def test_roundtrip_reference_transport():
    env = _env()
    assert FrameEnvelope.from_fields(env.to_fields()) == env


def test_roundtrip_inline_transport():
    env = _env(transport="inline", path=None, payload=b"\x89PNG\r\n\x1a\nbytes")
    assert FrameEnvelope.from_fields(env.to_fields()) == env


def test_transport_requires_its_own_payload_or_path():
    with pytest.raises(ValueError, match="inline"):
        _env(transport="inline", path=None, payload=None)
    with pytest.raises(ValueError, match="reference"):
        _env(transport="reference", path=None)


def test_captured_at_must_be_timezone_aware():
    with pytest.raises(ValueError, match="timezone-aware"):
        _env(captured_at=datetime(2026, 8, 2, 3, 4, 5))


def test_fixtures_carry_no_position_and_phones_do():
    """The skeleton runs with no phone attached, so absent position must be
    absent from the wire rather than serialised empty."""
    assert "lat" not in _env().to_fields()

    env = _env(lat=-33.8688, lon=151.2093, heading_deg=275.5,
               speed_mps=16.7, gps_accuracy_m=4.2)
    assert FrameEnvelope.from_fields(env.to_fields()) == env


def test_half_a_fix_is_rejected():
    """Dexter -> Ryan seam: a lat with no lon is a capture bug, not a partial
    result, and silently accepting it puts the defect on the wrong road."""
    with pytest.raises(ValueError, match="together"):
        _env(lat=-33.8688)


def test_monotonic_value_requires_a_boot_id_to_scope_it():
    """Boot-relative values are meaningless across devices and reboots, so one
    must never travel without the id that scopes it."""
    with pytest.raises(ValueError, match="device_boot_id"):
        _env(capture_mono_ns=123_456_789_000)


def test_monotonic_clock_survives_a_wall_clock_step_backwards():
    """The whole reason capture_mono_ns exists. A backwards NTP step must not be
    able to produce a negative frame interval, because intervals feed
    coverage-in-metres."""
    early = _env(seq=1, captured_at=datetime(2026, 8, 2, 3, 4, 5, tzinfo=timezone.utc),
                 capture_mono_ns=1_000_000_000, device_boot_id=BOOT)
    late = _env(seq=2, captured_at=datetime(2026, 8, 2, 3, 4, 4, tzinfo=timezone.utc),
                capture_mono_ns=1_100_000_000, device_boot_id=BOOT)

    assert (late.captured_at - early.captured_at).total_seconds() < 0      # the bug
    assert (late.capture_mono_ns - early.capture_mono_ns) / 1e9 == pytest.approx(0.1)
