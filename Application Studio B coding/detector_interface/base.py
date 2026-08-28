"""Contract 2 — the frozen detector plugin interface. Every model in plugins/ implements this
so Ryan's pipeline can swap detectors without changing its own code, and so the team can
benchmark accuracy/latency trade-offs across models (§ Research Plan — Shervin, "detector is a
swappable component to benchmark, not a thing to perfect")."""

from abc import ABC, abstractmethod
from typing import Any

from .detection import Detection


class DetectorPlugin(ABC):
    """Base class for a swappable road-damage detector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'yolox-tiny'. Matches the key in registry.AVAILABLE_DETECTORS."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Weights/checkpoint version or tag, written into Detection.model_version."""

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Static metadata for the benchmark comparison table: input_size, num_params,
        published_map, published_latency_ms, license, etc."""

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Called once before the first detect() call."""

    @abstractmethod
    def detect(self, image: Any) -> list[Detection]:
        """Run inference on a single image (H, W, C numpy array, BGR) and return detections."""
