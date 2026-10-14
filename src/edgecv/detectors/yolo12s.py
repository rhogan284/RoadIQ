"""YOLOv12s road-damage detector behind the Contract 2 `Detector` protocol.

Model work: Shervin Sabu (components 3/4) — `origin/sherv_bigtest`,
`Application Studio B coding/detector_interface/plugins/yolo12s.py`. The class-order
remapping and the base-vs-continued checkpoint handling below are his; this module only
changes the output shape to Contract 2 (`defect_class` string, integer `BBox`, `severity`)
instead of merging his `DetectorPlugin` interface into the worker (Week 07 decision).

Two checkpoints, one code path:
- base: `rezzzq/yolo12s-road-damage-rdd2022` (MIT) — names {D00, D10, D20, D40, Repair}
- continued (Shervin's v4): trained on our data.yaml — names {longitudinal crack,
  transverse crack, alligator crack, other corruption, pothole}
Both are mapped by NAME, not index, so the order difference between them cannot silently
relabel potholes as "other corruption".

Classes are emitted as RDD damage codes. The fifth class of either checkpoint has no
equivalent in the 4-class ground truth and is emitted as `other`: stored and shown, never
scored and never counted in the condition index.

Licence: weights MIT; the `ultralytics` runtime is AGPL-3.0 (Licence Register, R-S3).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from edgecv.contracts.detection import BBox, Detection, DetectorInfo, severity_for

DEFAULT_WEIGHTS = "weights/yolo12s_RDD2022_best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640

#: Checkpoint class name (lower-cased) → RDD code. Covers both checkpoints.
NAME_TO_CODE = {
    "d00": "D00", "longitudinal crack": "D00", "longitudinal_crack": "D00",
    "d10": "D10", "transverse crack": "D10", "transverse_crack": "D10",
    "d20": "D20", "alligator crack": "D20", "alligator_crack": "D20",
    "d40": "D40", "pothole": "D40",
}
OTHER = "other"


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Yolo12sDetector:
    def __init__(self, *, weights: str | Path = DEFAULT_WEIGHTS,
                 conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"YOLOv12s weights not found at {self.weights}. Run `make weights`, or set "
                f"YOLO_WEIGHTS to Shervin's yolo12s_rdd2022_continued.pt.")
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self._model = YOLO(str(self.weights))
        self._codes = {i: NAME_TO_CODE.get(str(n).lower(), OTHER)
                       for i, n in self._model.names.items()}
        self._info = DetectorInfo(
            name="yolo12s",
            version="rdd2022-continued-ours" if "continued" in self.weights.name
            else "rdd2022-finetuned-hf-rezzzq",
            # The weights hash is in params so a weights swap is a new detectors row
            # (params_hash) and benchmark results can never mix two models.
            params={"weights": self.weights.name, "weights_sha256_16": _sha16(self.weights),
                    "conf": conf, "imgsz": imgsz,
                    "classes": {str(i): c for i, c in self._codes.items()}},
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image.ndim == 2:  # the model was trained on colour frames
            image = np.stack([image] * 3, axis=-1)
        height, width = image.shape[:2]
        result = self._model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                     verbose=False)[0]
        out: list[Detection] = []
        for (x1, y1, x2, y2), cls, conf in zip(result.boxes.xyxy.tolist(),
                                               result.boxes.cls.tolist(),
                                               result.boxes.conf.tolist()):
            x, y = max(0, int(round(x1))), max(0, int(round(y1)))
            w, h = int(round(x2)) - x, int(round(y2)) - y
            if w <= 0 or h <= 0:
                continue
            bbox = BBox(x=x, y=y, w=w, h=h).clamped(width, height)
            out.append(Detection(defect_class=self._codes.get(int(cls), OTHER),
                                 confidence=round(float(conf), 4), bbox=bbox,
                                 severity=severity_for(bbox.area)))
        return out
