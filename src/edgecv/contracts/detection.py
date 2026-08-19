"""CONTRACT 2 — detector output and the record the writer persists.

FROZEN. Changing this breaks both the detector and the writer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

Severity = Literal["minor", "major", "critical"]
Status = Literal["ok", "failed", "skipped"]

MINOR_MAX_AREA_PX = 100
MAJOR_MAX_AREA_PX = 1000


def severity_for(area_px: int) -> Severity:
    if area_px <= MINOR_MAX_AREA_PX:
        return "minor"
    if area_px <= MAJOR_MAX_AREA_PX:
        return "major"
    return "critical"


@dataclass(frozen=True, slots=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError("bbox width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError("bbox origin must be non-negative")

    @property
    def area(self) -> int:
        return self.w * self.h

    def clamped(self, width: int, height: int) -> BBox:
        """Never let a padded crop escape the frame."""
        x = min(self.x, width - 1)
        y = min(self.y, height - 1)
        return BBox(x=x, y=y, w=min(self.w, width - x), h=min(self.h, height - y))

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True, slots=True)
class Detection:
    defect_class: str
    confidence: float
    bbox: BBox
    severity: Severity

    def as_dict(self) -> dict[str, Any]:
        return {
            "defect_class": self.defect_class,
            "confidence": self.confidence,
            "bbox": self.bbox.as_dict(),
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Detection:
        return cls(
            defect_class=d["defect_class"],
            confidence=d["confidence"],
            bbox=BBox(**d["bbox"]),
            severity=d["severity"],
        )


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    name: str
    version: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def params_hash(self) -> str:
        """Stable hash of the config. A retuned threshold is a DIFFERENT detector row."""
        canonical = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "params": self.params}


@dataclass(frozen=True, slots=True)
class InferenceResult:
    run_id: str
    seq: int
    captured_at: datetime
    width: int
    height: int
    source_ref: str
    frame_sha256: str
    detector: DetectorInfo
    worker_id: str
    started_at: datetime
    latency_ms: float
    status: Status
    error: str | None
    detections: list[Detection]
    snippet_sha256s: list[str]
    thumbnail_sha256: str | None
    # Position/clock fields copied off FrameEnvelope by the worker so the
    # writer can persist them. FrameEnvelope already validated these; do not
    # re-validate here.
    lat: float | None = None
    lon: float | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None
    gps_accuracy_m: float | None = None
    capture_mono_ns: int | None = None
    device_boot_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.snippet_sha256s) != len(self.detections):
            raise ValueError("snippet_sha256s must align with detections")

    def to_json(self) -> bytes:
        return json.dumps({
            "run_id": self.run_id,
            "seq": self.seq,
            "captured_at": self.captured_at.isoformat(),
            "width": self.width,
            "height": self.height,
            "source_ref": self.source_ref,
            "frame_sha256": self.frame_sha256,
            "detector": self.detector.as_dict(),
            "worker_id": self.worker_id,
            "started_at": self.started_at.isoformat(),
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
            "detections": [d.as_dict() for d in self.detections],
            "snippet_sha256s": self.snippet_sha256s,
            "thumbnail_sha256": self.thumbnail_sha256,
            "lat": self.lat,
            "lon": self.lon,
            "heading_deg": self.heading_deg,
            "speed_mps": self.speed_mps,
            "gps_accuracy_m": self.gps_accuracy_m,
            "capture_mono_ns": self.capture_mono_ns,
            "device_boot_id": self.device_boot_id,
        }).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> InferenceResult:
        d = json.loads(raw)
        return cls(
            run_id=d["run_id"],
            seq=d["seq"],
            captured_at=datetime.fromisoformat(d["captured_at"]),
            width=d["width"],
            height=d["height"],
            source_ref=d["source_ref"],
            frame_sha256=d["frame_sha256"],
            detector=DetectorInfo(**d["detector"]),
            worker_id=d["worker_id"],
            started_at=datetime.fromisoformat(d["started_at"]),
            latency_ms=d["latency_ms"],
            status=d["status"],
            error=d["error"],
            detections=[Detection.from_dict(x) for x in d["detections"]],
            snippet_sha256s=d["snippet_sha256s"],
            thumbnail_sha256=d["thumbnail_sha256"],
            lat=d.get("lat"),
            lon=d.get("lon"),
            heading_deg=d.get("heading_deg"),
            speed_mps=d.get("speed_mps"),
            gps_accuracy_m=d.get("gps_accuracy_m"),
            capture_mono_ns=d.get("capture_mono_ns"),
            device_boot_id=d.get("device_boot_id"),
        )


@runtime_checkable
class Detector(Protocol):
    """Implement this to add a detector. B owns the implementations."""

    @property
    def info(self) -> DetectorInfo: ...

    def detect(self, image: np.ndarray) -> list[Detection]: ...
