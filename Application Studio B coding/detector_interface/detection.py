"""Contract 2 — the frozen detection record shared between Ryan's pipeline and Shervin's detector plugins."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    model_name: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return cls(**data)
