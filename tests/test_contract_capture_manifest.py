"""CONTRACT 5 tests.

The first five are Dexter's, unchanged from PR #3. Everything below them covers
a gap that PR #3's review turned up — each one was reproduced against the merged
code before the test was written.
"""
import dataclasses
import importlib.util
import json
import pathlib
import uuid

import jsonschema
import pytest

from edgecv.contracts import CaptureManifest

SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src/edgecv/contracts/schemas/capture_manifest.schema.json"
)


def _manifest(**overrides):
    """A valid manifest, with fields overridden per test."""
    data = dict(
        run_id=str(uuid.uuid4()),
        seq=1,
        captured_at="2026-08-19T09:15:03.220000+00:00",
        capture_mono_ns=1234567890123,
        device_boot_id=str(uuid.uuid4()),
        sha256="a" * 64,
    )
    data.update(overrides)
    return CaptureManifest(**data)


# --------------------------------------------------------------------------
# Dexter's original tests (PR #3), unchanged.
# --------------------------------------------------------------------------
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
        _manifest(lat=-33.8688)  # missing lon


def test_invalid_seq():
    with pytest.raises(ValueError, match="seq must be >= 0"):
        _manifest(seq=-1)


def test_invalid_sha256():
    with pytest.raises(
        ValueError, match="sha256 must be a 64-character lowercase hex string"
    ):
        _manifest(sha256="invalid_hex")


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


# --------------------------------------------------------------------------
# captured_at — was accepted unvalidated, then failed at ingest instead.
# --------------------------------------------------------------------------
def test_captured_at_rejects_garbage():
    with pytest.raises(ValueError, match="captured_at must be an ISO 8601 datetime"):
        _manifest(captured_at="not-a-date")


def test_captured_at_rejects_naive_datetime():
    # FrameEnvelope raises this at ingest. The device must catch it first.
    with pytest.raises(ValueError, match="captured_at must be timezone-aware"):
        _manifest(captured_at="2026-08-19T09:15:03")


def test_captured_at_rejects_non_string():
    with pytest.raises(ValueError, match="captured_at must be an ISO 8601 string"):
        _manifest(captured_at=1755594903)


def test_captured_at_accepts_zulu_and_offset_forms():
    assert _manifest(captured_at="2026-08-19T09:15:03.220000Z").captured_at
    assert _manifest(captured_at="2026-08-19T19:15:03+10:00").captured_at


# --------------------------------------------------------------------------
# capture_mono_ns — mandatory boot clock, previously unchecked entirely.
# --------------------------------------------------------------------------
def test_capture_mono_ns_rejects_none():
    # An int annotation is not a runtime check, and the schema is non-nullable.
    with pytest.raises(ValueError, match="capture_mono_ns must be an integer"):
        _manifest(capture_mono_ns=None)


def test_capture_mono_ns_rejects_negative():
    with pytest.raises(ValueError, match="capture_mono_ns must be >= 0"):
        _manifest(capture_mono_ns=-5)


def test_seq_rejects_none():
    with pytest.raises(ValueError, match="seq must be an integer"):
        _manifest(seq=None)


# --------------------------------------------------------------------------
# Position bounds — pairing alone cannot see a lat/lon swap.
# --------------------------------------------------------------------------
def test_swapped_lat_lon_is_rejected():
    # Sydney, reversed. Every Australian coordinate has |lon| > 90.
    with pytest.raises(ValueError, match="lat must be between -90 and 90"):
        _manifest(lat=151.2093, lon=-33.8688)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lat", 999.0, "lat must be between -90 and 90"),
        ("lon", 999.0, "lon must be between -180 and 180"),
        ("heading_deg", 400.0, "heading_deg must be between 0 and 360"),
        ("speed_mps", -1.0, "speed_mps must be >= 0"),
        ("gps_accuracy_m", -1.0, "gps_accuracy_m must be >= 0"),
    ],
)
def test_out_of_range_position_fields(field, value, message):
    kwargs = {field: value}
    if field in ("lat", "lon"):
        kwargs.setdefault("lat", 0.0)
        kwargs.setdefault("lon", 0.0)
        kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        _manifest(**kwargs)


def test_sydney_coordinates_are_accepted():
    m = _manifest(lat=-33.8688, lon=151.2093, heading_deg=275.5, speed_mps=16.7,
                  gps_accuracy_m=4.2)
    assert m.lon == 151.2093


# --------------------------------------------------------------------------
# Immutability — validation was a one-time gate on a mutable object.
# --------------------------------------------------------------------------
def test_manifest_is_frozen():
    m = _manifest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.seq = -99


def test_frozen_blocks_breaking_the_lat_lon_pair_after_construction():
    m = _manifest(lat=-33.8688, lon=151.2093)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.lon = None


# --------------------------------------------------------------------------
# sha256 — `$` also matches before a trailing newline; `fullmatch` does not.
# --------------------------------------------------------------------------
def test_sha256_rejects_trailing_newline():
    # A digest read straight off a `sha256sum` line.
    with pytest.raises(ValueError, match="sha256 must be a 64-character"):
        _manifest(sha256="a" * 64 + "\n")


@pytest.mark.parametrize("bad", ["A" * 64, "a" * 63, "a" * 65, None, 12345])
def test_sha256_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError, match="sha256 must be a 64-character"):
        _manifest(sha256=bad)


# --------------------------------------------------------------------------
# UUIDs — wrong exception type, and non-canonical spellings split one run.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["run_id", "device_boot_id"])
@pytest.mark.parametrize("bad", [None, 12345, "not-a-uuid", ""])
def test_uuid_fields_raise_value_error_not_type_error(field, bad):
    # uuid.UUID() raises TypeError for None and AttributeError for an int. An
    # ingest loop quarantining on ValueError would crash on both.
    with pytest.raises(ValueError, match=field):
        _manifest(**{field: bad})


def test_uuid_is_normalised_to_canonical_form():
    run = uuid.uuid4()
    assert _manifest(run_id=run.hex).run_id == str(run)
    assert _manifest(run_id=f"urn:uuid:{run}").run_id == str(run)
    assert _manifest(run_id=f"{{{run}}}").run_id == str(run)


def test_same_run_in_two_spellings_is_one_run():
    run, boot = uuid.uuid4(), uuid.uuid4()
    common = dict(captured_at="2026-08-19T09:15:03.220000+00:00", seq=7,
                  capture_mono_ns=1, sha256="c" * 64, device_boot_id=str(boot))
    assert CaptureManifest(run_id=str(run), **common) == CaptureManifest(
        run_id=run.hex, **common
    )


# --------------------------------------------------------------------------
# from_dict — raised KeyError, which a ValueError handler will not catch.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("missing", list(
    ["run_id", "seq", "captured_at", "capture_mono_ns", "device_boot_id", "sha256"]
))
def test_from_dict_missing_required_field_raises_value_error(missing):
    data = {
        "run_id": str(uuid.uuid4()),
        "seq": 1,
        "captured_at": "2026-08-19T09:15:03.220000+00:00",
        "capture_mono_ns": 1,
        "device_boot_id": str(uuid.uuid4()),
        "sha256": "d" * 64,
    }
    del data[missing]
    with pytest.raises(ValueError, match="missing required field"):
        CaptureManifest.from_dict(data)


def test_from_dict_rejects_non_dict():
    with pytest.raises(ValueError, match="data must be a dict"):
        CaptureManifest.from_dict([1, 2, 3])


def test_full_roundtrip_equality_with_every_optional_set():
    # Matches the strength of test_contract_frame.py's round-trip assertions.
    m = _manifest(lat=-33.8688, lon=151.2093, heading_deg=275.5, speed_mps=16.7,
                  gps_accuracy_m=4.2)
    assert CaptureManifest.from_dict(m.to_dict()) == m


def test_roundtrip_omits_none_fields_entirely():
    m = _manifest()
    assert set(m.to_dict()) == {
        "run_id", "seq", "captured_at", "capture_mono_ns", "device_boot_id", "sha256"
    }
    assert CaptureManifest.from_dict(m.to_dict()) == m


# --------------------------------------------------------------------------
# Schema conformance — nothing checked the dataclass against its own schema.
# --------------------------------------------------------------------------
def _schema():
    return json.loads(SCHEMA_PATH.read_text())


def test_exported_schema_is_frozen():
    assert _schema()["x-status"].startswith("FROZEN")


def test_valid_manifest_validates_against_exported_schema():
    m = _manifest(lat=-33.8688, lon=151.2093, heading_deg=275.5, speed_mps=16.7,
                  gps_accuracy_m=4.2)
    jsonschema.validate(m.to_dict(), _schema())


@pytest.mark.parametrize(
    "payload",
    [
        {"seq": -1},
        {"capture_mono_ns": -1},
        {"sha256": "A" * 64},
        {"lat": 999.0, "lon": 0.0},
        {"lon": 999.0, "lat": 0.0},
        {"heading_deg": 400.0},
        {"speed_mps": -1.0},
    ],
)
def test_schema_rejects_what_the_dataclass_rejects(payload):
    """The schema must mirror __post_init__, so a non-Python device agrees."""
    data = {
        "run_id": str(uuid.uuid4()),
        "seq": 1,
        "captured_at": "2026-08-19T09:15:03.220000+00:00",
        "capture_mono_ns": 1,
        "device_boot_id": str(uuid.uuid4()),
        "sha256": "e" * 64,
    }
    data.update(payload)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema())


def test_schema_requires_lat_and_lon_together():
    data = {
        "run_id": str(uuid.uuid4()),
        "seq": 1,
        "captured_at": "2026-08-19T09:15:03.220000+00:00",
        "capture_mono_ns": 1,
        "device_boot_id": str(uuid.uuid4()),
        "sha256": "e" * 64,
        "lat": -33.8688,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema())


def test_schema_file_matches_its_generator():
    """Regression guard for PR #3: the schema was hand-edited, so the next
    `python scripts/export_schemas.py` silently reverted the freeze.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts/export_schemas.py"
    spec = importlib.util.spec_from_file_location("export_schemas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CAPTURE_MANIFEST == _schema(), (
        "capture_manifest.schema.json is out of sync with CAPTURE_MANIFEST in "
        "scripts/export_schemas.py — edit the generator, then re-run it."
    )
