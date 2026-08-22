"""YOLOv4-tiny adapter. Official figures (AlexeyAB/darknet): 40.2% mAP@0.5, 6.9 BFlops,
23.1 MB weights. Permissively licensed ("YOLO License", AlexeyAB/darknet) — one of the R-S3
survey's low-risk candidates for a self-hostable product.

Loaded via cv2.dnn.readNetFromDarknet directly against the official .cfg/.weights — no darknet
build toolchain needed for inference. Real inference — no longer a stub. Note: these are
COCO-pretrained weights, not fine-tuned on RDD2022 (darknet training needs the darknet C/CUDA
toolchain, which isn't set up on this machine — see README discussion). So it will correctly
detect COCO objects but won't recognise road-damage classes; expect ~0% accuracy against RDD2022
ground truth until it's fine-tuned. It's still wired up for a fair speed/latency comparison
against the other four.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..base import DetectorPlugin
from ..detection import Detection

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / "yolov4_tiny"
CFG_PATH = WEIGHTS_DIR / "yolov4-tiny.cfg"
WEIGHTS_PATH = WEIGHTS_DIR / "yolov4-tiny.weights"
INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YOLOv4TinyDetector(DetectorPlugin):
    def __init__(self) -> None:
        self._net = None

    @property
    def name(self) -> str:
        return "yolov4-tiny"

    @property
    def version(self) -> str:
        return "darknet_yolo_v4_pre-coco-pretrained"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "input_size": INPUT_SIZE,
            "size_mb": 23.1,
            "bflops": 6.9,
            "published_map_at_0.5": 40.2,
            "license": "YOLO License (permissive)",
            "cfg_path": str(CFG_PATH),
            "weights_path": str(WEIGHTS_PATH),
            "fine_tuned_on_rdd2022": False,
            "accuracy_caveat": "COCO-pretrained only — expect ~0% accuracy on RDD2022 classes until fine-tuned via the darknet toolchain.",
        }

    def load(self) -> None:
        missing = [p for p in (CFG_PATH, WEIGHTS_PATH) if not p.exists()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(f"YOLOv4-tiny file(s) not found: {names}.")
        self._net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image: Any) -> list[Detection]:
        if self._net is None:
            self.load()

        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        self._net.setInput(blob)
        output_names = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(output_names)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for row in output:
                scores = row[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                cx, cy, w, h = row[0] * width, row[1] * height, row[2] * width, row[3] * height
                boxes.append([cx - w / 2, cy - h / 2, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        detections = []
        for i in np.array(keep).flatten() if len(keep) else []:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else str(class_id),
                    confidence=confidences[i],
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=w,
                    bbox_h=h,
                    model_name=self.name,
                    model_version=self.version,
                )
            )
        return detections
