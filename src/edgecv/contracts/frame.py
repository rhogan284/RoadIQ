"""CONTRACT 1 — the frame envelope published by feed-sim, consumed by workers.

FROZEN. Changing this breaks Shervin's worker. Raise with the team first.

Two clocks, two jobs:

    captured_at      absolute time — partitioning, display, cross-run joins.
                     Prefer GPS-derived UTC when a fix is present, since it is
                     immune to handset clock steps.
    capture_mono_ns  intervals and rates ONLY. Boot-relative, so comparable
                     within one (device_boot_id, run_id) and nowhere else.
                     Never store it as an absolute time.

Position and clock fields are optional: the skeleton runs on generated fixtures
with no phone attached.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Transport = Literal["inline", "reference"]

#: Optional fields, omitted from the wire entirely when None.
_FLOATS = ("lat", "lon", "heading_deg", "speed_mps", "gps_accuracy_m")
_INTS = ("capture_mono_ns",)
_STRS = ("device_boot_id",)


@dataclass(frozen=True, slots=True)
class FrameEnvelope:
    run_id: str
    seq: int
    captured_at: datetime
    width: int
    height: int
    source_ref: str
    sha256: str
    transport: Transport
    payload: bytes | None = None
    path: str | None = None
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None
    gps_accuracy_m: float | None = None
    capture_mono_ns: int | None = None
    device_boot_id: str | None = None

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        if self.transport == "inline" and not self.payload:
            raise ValueError("inline transport requires payload")
        if self.transport == "reference" and not self.path:
            raise ValueError("reference transport requires path")
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be provided together")
        if self.capture_mono_ns is not None and self.device_boot_id is None:
            raise ValueError("capture_mono_ns requires device_boot_id to scope it")

    def to_fields(self) -> dict[str, bytes]:
        """Flatten to Redis stream fields. Bytes because payload is binary.

        Optional fields are omitted when None, not sent empty — absent means absent.
        """
        fields: dict[str, bytes] = {
            "run_id": self.run_id.encode(),
            "seq": str(self.seq).encode(),
            "captured_at": self.captured_at.isoformat().encode(),
            "width": str(self.width).encode(),
            "height": str(self.height).encode(),
            "source_ref": self.source_ref.encode(),
            "sha256": self.sha256.encode(),
            "transport": self.transport.encode(),
        }
        if self.transport == "inline":
            assert self.payload is not None
            fields["payload"] = self.payload
        else:
            assert self.path is not None
            fields["path"] = self.path.encode()

        for name in _FLOATS + _INTS + _STRS:
            value = getattr(self, name)
            if value is not None:
                # repr() for numbers because it round-trips exactly.
                fields[name] = value.encode() if isinstance(value, str) \
                    else repr(value).encode()
        return fields

    @classmethod
    def from_fields(cls, fields: dict) -> FrameEnvelope:
        def get(key: str) -> bytes | None:
            # redis-py returns bytes keys when decode_responses=False, which we
            # need for binary payloads.
            return fields.get(key.encode()) or fields.get(key)

        transport = get("transport").decode()
        raw_path = get("path") if transport == "reference" else None

        optional: dict[str, object] = {}
        for names, cast in ((_FLOATS, float), (_INTS, int), (_STRS, bytes.decode)):
            for name in names:
                raw = get(name)
                if raw is not None:
                    optional[name] = cast(raw)

        return cls(
            run_id=get("run_id").decode(),
            seq=int(get("seq")),
            captured_at=datetime.fromisoformat(get("captured_at").decode()),
            width=int(get("width")),
            height=int(get("height")),
            source_ref=get("source_ref").decode(),
            sha256=get("sha256").decode(),
            transport=transport,  # type: ignore[arg-type]
            payload=get("payload") if transport == "inline" else None,
            path=raw_path.decode() if raw_path else None,
            **optional,  # type: ignore[arg-type]
        )
