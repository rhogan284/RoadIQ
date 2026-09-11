from dataclasses import dataclass
from typing import Optional
import uuid
import re

@dataclass
class CaptureManifest:
    run_id: str
    seq: int
    captured_at: str
    capture_mono_ns: int
    device_boot_id: str
    sha256: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    heading_deg: Optional[float] = None
    speed_mps: Optional[float] = None
    gps_accuracy_m: Optional[float] = None

    def __post_init__(self):
        
        uuid.UUID(self.run_id)
        uuid.UUID(self.device_boot_id)

    
        if self.seq < 0:
            raise ValueError("seq must be >= 0")

        
        if not re.match(r"^[0-9a-f]{64}$", self.sha256):
            raise ValueError("sha256 must be a 64-character lowercase hex string")

        
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be provided together or both be None")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "CaptureManifest":
        return cls(
            run_id=data["run_id"],
            seq=data["seq"],
            captured_at=data["captured_at"],
            capture_mono_ns=data["capture_mono_ns"],
            device_boot_id=data["device_boot_id"],
            sha256=data["sha256"],
            lat=data.get("lat"),
            lon=data.get("lon"),
            heading_deg=data.get("heading_deg"),
            speed_mps=data.get("speed_mps"),
            gps_accuracy_m=data.get("gps_accuracy_m"),
        )
