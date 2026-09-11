import uuid
import pytest
from edgecv.contracts import CaptureManifest


def test_valid_capture_manifest():
    manifest = CaptureManifest(
        run_id=str(uuid.uuid4()),
        seq=1,
        captured_at="2026-08-19T09:15:03.220000+00:00",
        capture_mono_ns=1234567890123,
        device_boot_id=str(uuid.uuid4()),
        sha256="a" * 64,
        lat=-33.8688,
        lon=151.2093,
    )
    assert manifest.seq == 1
    assert manifest.lat == -33.8688


def test_invalid_lat_lon_pair():
    with pytest.raises(ValueError, match="lat and lon must be provided together"):
        CaptureManifest(
            run_id=str(uuid.uuid4()),
            seq=1,
            captured_at="2026-08-19T09:15:03.220000+00:00",
            capture_mono_ns=1234567890123,
            device_boot_id=str(uuid.uuid4()),
            sha256="a" * 64,
            lat=-33.8688,  # missing lon
        )


def test_invalid_seq():
    with pytest.raises(ValueError, match="seq must be >= 0"):
        CaptureManifest(
            run_id=str(uuid.uuid4()),
            seq=-1,
            captured_at="2026-08-19T09:15:03.220000+00:00",
            capture_mono_ns=1234567890123,
            device_boot_id=str(uuid.uuid4()),
            sha256="a" * 64,
        )


def test_invalid_sha256():
    with pytest.raises(
        ValueError, match="sha256 must be a 64-character lowercase hex string"
    ):
        CaptureManifest(
            run_id=str(uuid.uuid4()),
            seq=0,
            captured_at="2026-08-19T09:15:03.220000+00:00",
            capture_mono_ns=1234567890123,
            device_boot_id=str(uuid.uuid4()),
            sha256="invalid_hex",
        )


def test_dict_serialization_roundtrip():
    data = {
        "run_id": str(uuid.uuid4()),
        "seq": 43,
        "captured_at": "2026-08-19T09:15:03.220000+00:00",
        "capture_mono_ns": 1234567890123,
        "device_boot_id": str(uuid.uuid4()),
        "sha256": "b" * 64,
    }
    manifest = CaptureManifest.from_dict(data)
    assert manifest.heading_deg is None

    serialized = manifest.to_dict()
    assert "heading_deg" not in serialized
    assert serialized["seq"] == 43
