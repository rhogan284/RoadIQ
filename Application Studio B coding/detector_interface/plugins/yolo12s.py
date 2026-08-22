"""YOLOv12s adapter. Base checkpoint: rezzzq/yolo12s-road-damage-rdd2022 on Hugging Face,
already fine-tuned on RDD2022 by its author. training/train_yolo12s.py continues fine-tuning
that checkpoint on our own cleaned RDD2022 subset; once its output
(weights/yolo12s/yolo12s_rdd2022_continued.pt) exists, this plugin prefers it automatically —
same pattern as yolo11n.py's base-vs-finetuned fallback.

Loads with the already-installed stock `ultralytics` package (8.4.121) — confirmed by loading
the checkpoint directly; the model card's "pip install git+https://github.com/sunsmarterjie/
yolov12.git" instruction is unnecessary here since Ultralytics merged official YOLOv12 support
into the mainline package. Using that fork instead would risk clobbering the `ultralytics`
install that yolo11n.py depends on, so deliberately not doing that.

License: MIT (weights). Note the same AGPL-3.0 caveat as yolo11n.py: the *weights* are MIT, but
loading/running them still depends on importing the `ultralytics` package itself, which is
AGPL-3.0 (R-S3).

Class mapping caveat (base checkpoint only): the rezzzq checkpoint's own class order
(model.names) is {0: D00 longitudinal crack, 1: D10 transverse crack, 2: D20 alligator crack,
3: D40 pothole, 4: Repair} — NOT the same order as our RDD2022_CLASS_NAMES (detector_interface
plugins / prepare_dataset.py), which has "other corruption" at index 3 and "pothole" at index 4.
detect() remaps D00/D10/D20/D40 onto our indices 0/1/2/4. "Repair" has no equivalent in our
5-class ground truth at all, so it's remapped to class_id=5 — outside our GT's 0-4 range — so
eval/benchmark.py's per-class AP loop (which only scores class ids present in the ground truth)
silently excludes Repair detections from accuracy metrics instead of miscounting them against an
unrelated class. Once train_yolo12s.py has continued training on data.yaml, the checkpoint's own
model.names already match our class order directly (that's what training against data.yaml
does), so the remap is skipped for that checkpoint.
"""

from pathlib import Path
from typing import Any

from ..base import DetectorPlugin
from ..detection import Detection

BASE_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "yolo12s" / "yolo12s_rdd2022.pt"
CONTINUED_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "weights" / "yolo12s" / "yolo12s_rdd2022_continued.pt"
)

# base checkpoint class_id -> our RDD2022_CLASS_NAMES class_id/name. Repair has no equivalent in
# our ground truth, so it's mapped to an id outside the 0-4 range used by our labels. Only used
# for the base (rezzzq) checkpoint — the continued checkpoint already matches our class order.
BASE_CHECKPOINT_CLASS_MAP: dict[int, tuple[int, str]] = {
    0: (0, "longitudinal crack"),  # D00
    1: (1, "transverse crack"),    # D10
    2: (2, "alligator crack"),     # D20
    3: (4, "pothole"),             # D40
    4: (5, "repair (no GT equivalent)"),
}


class YOLO12sDetector(DetectorPlugin):
    def __init__(self) -> None:
        self._model = None
        self._weights_path = CONTINUED_WEIGHTS_PATH if CONTINUED_WEIGHTS_PATH.exists() else BASE_WEIGHTS_PATH
        self._continued = CONTINUED_WEIGHTS_PATH.exists()

    @property
    def name(self) -> str:
        return "yolo12s"

    @property
    def version(self) -> str:
        return "rdd2022-continued-ours" if self._continued else "rdd2022-finetuned-hf-rezzzq"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "input_size": 640,
            "license": "MIT (weights) / AGPL-3.0 (ultralytics runtime dependency, same caveat as yolo11n)",
            "source": "https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022",
            "weights_path": str(self._weights_path),
            "fine_tuned_on_rdd2022": True,
            "fine_tuned_by": "our training run (continued from rezzzq base)" if self._continued else "third party (not our training run)",
            "class_mapping_caveat": None
            if self._continued
            else "base checkpoint's 'Repair' class has no equivalent in our ground truth; excluded from accuracy scoring, not just mislabeled.",
        }

    def load(self) -> None:
        if not self._weights_path.exists():
            raise FileNotFoundError(f"YOLOv12s weights not found at {self._weights_path}.")
        from ultralytics import YOLO

        self._model = YOLO(str(self._weights_path))

    def detect(self, image: Any) -> list[Detection]:
        if self._model is None:
            self.load()
        results = self._model.predict(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_class_id = int(box.cls[0])
            if self._continued:
                class_id, class_name = raw_class_id, self._model.names[raw_class_id]
            else:
                class_id, class_name = BASE_CHECKPOINT_CLASS_MAP.get(
                    raw_class_id, (raw_class_id, str(raw_class_id))
                )
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
