"""Pick the detector from the environment, so the worker never hard-codes one.

    DETECTOR=threshold            classical baseline (default; the unit tests' detector)
    DETECTOR=yolo12s              YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ optional
"""
from __future__ import annotations

import os
from typing import Mapping

from edgecv.contracts.detection import Detector


def build_detector(env: Mapping[str, str] | None = None) -> Detector:
    e = os.environ if env is None else env
    name = e.get("DETECTOR", "threshold")
    if name == "threshold":
        from edgecv.detectors.threshold import ThresholdDetector
        return ThresholdDetector()
    if name == "yolo12s":
        from edgecv.detectors.yolo12s import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_WEIGHTS, \
            Yolo12sDetector
        return Yolo12sDetector(weights=e.get("YOLO_WEIGHTS", DEFAULT_WEIGHTS),
                               conf=float(e.get("YOLO_CONF", DEFAULT_CONF)),
                               imgsz=int(e.get("YOLO_IMGSZ", DEFAULT_IMGSZ)))
    raise ValueError(f"unknown DETECTOR={name!r} (expected threshold or yolo12s)")
