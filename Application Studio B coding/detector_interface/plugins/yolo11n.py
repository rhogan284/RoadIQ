"""YOLO11n adapter. Ultralytics (ultralytics/ultralytics). ⚠️ AGPL-3.0 licensed — this was
flagged in R-S3 as a licensing risk for a product pitched as self-hostable/reusable; keep
this plugin in the benchmark comparison, but the AGPL trade-off needs signing off before it
becomes the shipped default (see R-S3 recommendation).

Base COCO weights at weights/yolo11n/yolo11n.pt; fine-tuned RDD2022 weights (from
training/train_yolo11n.py) land at weights/yolo11n/yolo11n_rdd2022.pt and are preferred
automatically once present. Real inference — no longer a stub.
"""

from pathlib import Path
from typing import Any

from ..base import DetectorPlugin
from ..detection import Detection

BASE_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "yolo11n" / "yolo11n.pt"
FINETUNED_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "yolo11n" / "yolo11n_rdd2022.pt"

RDD2022_CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]


class YOLO11nDetector(DetectorPlugin):
    def __init__(self) -> None:
        self._model = None
        self._weights_path = FINETUNED_WEIGHTS_PATH if FINETUNED_WEIGHTS_PATH.exists() else BASE_WEIGHTS_PATH
        self._fine_tuned = FINETUNED_WEIGHTS_PATH.exists()

    @property
    def name(self) -> str:
        return "yolo11n"

    @property
    def version(self) -> str:
        return "rdd2022-finetuned" if self._fine_tuned else "v8.3.0-coco-pretrained"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "input_size": 416 if self._fine_tuned else 640,
            "license": "AGPL-3.0",
            "license_warning": "Flagged in R-S3 — AGPL conflicts with a self-hostable/reusable product pitch.",
            "weights_path": str(self._weights_path),
            "fine_tuned_on_rdd2022": self._fine_tuned,
        }

    def load(self) -> None:
        if not self._weights_path.exists():
            raise FileNotFoundError(f"YOLO11n weights not found at {self._weights_path}.")
        from ultralytics import YOLO

        self._model = YOLO(str(self._weights_path))

    def detect(self, image: Any) -> list[Detection]:
        if self._model is None:
            self.load()
        results = self._model.predict(image, verbose=False)[0]
        class_names = RDD2022_CLASS_NAMES if self._fine_tuned else self._model.names
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = class_names[class_id] if self._fine_tuned else class_names.get(class_id, str(class_id))
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox_x=x1,
                    bbox_y=y1,
                    bbox_w=x2 - x1,
                    bbox_h=y2 - y1,
                    model_name=self.name,
                    model_version=self.version,
                )
            )
        return detections
