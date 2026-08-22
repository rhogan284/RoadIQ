"""Plain dict-based registry — deliberately not a setuptools entry-point plugin system.
This is five known models to benchmark against each other, not a published/extensible
plugin ecosystem, so a dict + factory function is the right amount of machinery."""

from .base import DetectorPlugin
from .plugins.nanodet_m import NanoDetMDetector
from .plugins.yolo11n import YOLO11nDetector
from .plugins.yolo12s import YOLO12sDetector
from .plugins.yolov4_tiny import YOLOv4TinyDetector
from .plugins.yolox_nano import YOLOXNanoDetector
from .plugins.yolox_tiny import YOLOXTinyDetector

AVAILABLE_DETECTORS: dict[str, type[DetectorPlugin]] = {
    "yolox-tiny": YOLOXTinyDetector,
    "yolox-nano": YOLOXNanoDetector,
    "yolov4-tiny": YOLOv4TinyDetector,
    "yolo11n": YOLO11nDetector,
    "yolo12s": YOLO12sDetector,
    "nanodet-m": NanoDetMDetector,
}


def get_detector(name: str) -> DetectorPlugin:
    try:
        cls = AVAILABLE_DETECTORS[name]
    except KeyError:
        available = ", ".join(sorted(AVAILABLE_DETECTORS))
        raise KeyError(f"Unknown detector '{name}'. Available: {available}") from None
    return cls()
