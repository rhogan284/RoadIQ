"""Classical baseline detector: dark-blob contour detection.

Deliberately simple. It exists so the pipeline is provably end-to-end correct before any
model training happens, and so there is always a working baseline to benchmark against.
Shervin replaces/extends this with the trained detectors behind the same Detector protocol.
"""
from __future__ import annotations

import cv2
import numpy as np

from edgecv.contracts.detection import (
    BBox, Detection, DetectorInfo, severity_for,
)

VERSION = "1.0.0"


class ThresholdDetector:
    # min_area defaults well above any blurred-noise blob: fixture defects are
    # 12-30 px per side (area 144-900), so 60 leaves a safe margin without
    # letting stray 3-sigma noise pixels register as detections.
    def __init__(self, *, k: float = 3.0, min_area: int = 60,
                 blur_ksize: int = 5) -> None:
        self.k = k
        self.min_area = min_area
        self.blur_ksize = blur_ksize

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="threshold", version=VERSION,
            params={"k": self.k, "min_area": self.min_area,
                    "blur_ksize": self.blur_ksize},
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(image, (self.blur_ksize, self.blur_ksize), 0)
        mean, std = float(blurred.mean()), float(blurred.std())
        # Flag pixels k standard deviations DARKER than the frame mean.
        cutoff = max(0.0, mean - self.k * std)
        _used, mask = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY_INV)

        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = image.shape[:2]
        detections: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < self.min_area:
                continue
            bbox = BBox(x=int(x), y=int(y), w=int(w), h=int(h)).clamped(width, height)
            detections.append(Detection(
                defect_class="pothole",
                confidence=min(1.0, (mean - float(blurred[y:y + h, x:x + w].mean()))
                               / max(std, 1e-6) / self.k),
                bbox=bbox,
                severity=severity_for(bbox.area),
            ))
        return detections
