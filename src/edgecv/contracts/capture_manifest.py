"""CONTRACT 5 — the manifest Dexter's device writes, consumed by Ryan's ingest.

Status is carried by the exported schema and docs/CONTRACTS.md, not by this
docstring. `spool_seq` is DEFERRED, not forgotten — it depends on an unanswered
question about whether inference is deferred to a laptop, so it is deliberately
absent rather than guessed at.

Two clocks, two jobs — the same split as the frame envelope:

    captured_at      absolute time, timezone-aware. GPS-derived UTC where a fix
                     exists at capture time; best-effort device clock otherwise.
    capture_mono_ns  intervals and rates ONLY. Boot-relative, so comparable
                     within one (device_boot_id, run_id) and nowhere else.
                     Never store it as an absolute time.

Unlike FrameEnvelope, BOTH clock fields are required here: the manifest is
device-authored, so the boot-relative clock always exists at capture time.

Validation is deliberately strict, and every failure is a ValueError. The device
is the last place a bad manifest can be caught cheaply — past ingest,
`FrameEnvelope.__post_init__` rejects it and the frame never reaches the pipeline
at all. Catching it here is the difference between one rejected manifest and a
silently missing frame.

The exported JSON Schema mirrors every rule below, so a non-Python producer
validates identically. Change one, change both: scripts/export_schemas.py.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

#: Lowercase hex, exactly 64 characters. fullmatch, never match — Python's `$`
#: also matches before a trailing newline, so `match` would accept a digest
#: copied straight off a `sha256sum` line. The JSON Schema `pattern` uses
#: ECMA-262 semantics, where it would not. That gap is the bug.
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: Required on the wire. Ordered as in docs/CONTRACTS.md §5.
_REQUIRED = (
    "run_id",
    "seq",
    "captured_at",
    "capture_mono_ns",
    "device_boot_id",
    "sha256",
)

#: Optional. Omitted from the wire entirely when None — absent means absent.
_OPTIONAL = ("lat", "lon", "heading_deg", "speed_mps", "gps_accuracy_m")


def _canonical_uuid(value: object, field: str) -> str:
    """Validate and normalise to the canonical hyphenated form.

    `uuid.UUID()` alone is the wrong gate twice over. It raises TypeError for
    None and AttributeError for an int, neither of which an ingest loop that
    quarantines on ValueError will catch. It also accepts braced, `urn:uuid:`
    and dash-free spellings, and the caller keeps whatever it was handed — so
    one run arriving in two spellings becomes two runs.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a UUID string, got {type(value).__name__}"
        )
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid UUID: {value!r}") from exc


def _non_negative_int(value: object, field: str) -> None:
    """Reject None, floats, bools and negatives.

    A dataclass annotation is not a runtime check: `capture_mono_ns=None`
    satisfies `capture_mono_ns: int` happily, and the schema declares a
    non-nullable integer. bool is excluded explicitly because it subclasses int.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field} must be an integer, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field} must be >= 0")


def _in_range(
    value: float | None, field: str, low: float, high: float | None = None
) -> None:
    """Bounds check for an optional number. `high=None` means unbounded above."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {type(value).__name__}")
    if high is None:
        if value < low:
            raise ValueError(f"{field} must be >= {low:g}, got {value}")
        return
    if not low <= value <= high:
        raise ValueError(
            f"{field} must be between {low:g} and {high:g}, got {value}"
        )


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    run_id: str
    seq: int
    captured_at: str
    capture_mono_ns: int
    device_boot_id: str
    sha256: str
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None
    gps_accuracy_m: float | None = None

    def __post_init__(self) -> None:
        # frozen=True, so normalised values are written back the way a frozen
        # dataclass is allowed to: object.__setattr__ during __post_init__.
        object.__setattr__(self, "run_id", _canonical_uuid(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "device_boot_id",
            _canonical_uuid(self.device_boot_id, "device_boot_id"),
        )

        _non_negative_int(self.seq, "seq")
        _non_negative_int(self.capture_mono_ns, "capture_mono_ns")

        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be a 64-character lowercase hex string")

        self._validate_captured_at()
        self._validate_position()

    def _validate_captured_at(self) -> None:
        if not isinstance(self.captured_at, str):
            raise ValueError(
                "captured_at must be an ISO 8601 string, got "
                f"{type(self.captured_at).__name__}"
            )
        try:
            parsed = datetime.fromisoformat(self.captured_at)
        except ValueError as exc:
            raise ValueError(
                f"captured_at must be an ISO 8601 datetime: {self.captured_at!r}"
            ) from exc
        if parsed.tzinfo is None:
            # FrameEnvelope raises exactly this at ingest. Catch it on the device.
            raise ValueError("captured_at must be timezone-aware")

    def _validate_position(self) -> None:
        if (self.lat is None) != (self.lon is None):
            raise ValueError(
                "lat and lon must be provided together or both be None"
            )
        # Bounds exist to catch a lat/lon swap, which pairing alone cannot see.
        # Every Australian coordinate has |lon| > 90, so a swapped pair breaks
        # the lat bound. segment_feature.schema.json names the same trap:
        # "swap them and every point lands in the wrong hemisphere".
        _in_range(self.lat, "lat", -90.0, 90.0)
        _in_range(self.lon, "lon", -180.0, 180.0)
        _in_range(self.heading_deg, "heading_deg", 0.0, 360.0)
        _in_range(self.speed_mps, "speed_mps", 0.0)
        _in_range(self.gps_accuracy_m, "gps_accuracy_m", 0.0)

    def to_dict(self) -> dict:
        """JSON-ready dict, optional fields omitted when None.

        asdict(), not self.__dict__ — slots=True means there is no __dict__.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> CaptureManifest:
        """Build from a decoded JSON object. Every failure is a ValueError.

        A bare data["run_id"] raises KeyError, which an ingest loop that
        quarantines malformed manifests on ValueError does not catch — one bad
        manifest would take the whole consumer down instead of being skipped.
        """
        if not isinstance(data, dict):
            raise ValueError(f"data must be a dict, got {type(data).__name__}")
        missing = [name for name in _REQUIRED if name not in data]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        return cls(
            **{name: data[name] for name in _REQUIRED},
            **{name: data.get(name) for name in _OPTIONAL},
        )
