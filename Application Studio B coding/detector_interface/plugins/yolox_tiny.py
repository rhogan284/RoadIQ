"""YOLOX-Tiny adapter. R-S1 survey figures: 416x416 input, 5.06M params, 6.45 GFLOPs,
32.8 COCO mAP. Apache-2.0 licensed (Megvii-BaseDetection/YOLOX) — no AGPL concern.

Weights downloaded to weights/yolox_tiny/yolox_tiny.pth (from the official 0.1.1rc0 GitHub
release) — COCO-pretrained only, same caveat as yolov4_tiny.py: expect ~0% accuracy against
RDD2022 ground truth until fine-tuned (no fine-tuning run of our own yet).

The `yolox` PyPI package is broken (its sdist is missing requirements.txt — a known upstream
packaging bug), so it's installed straight from GitHub instead:
    pip install --no-build-isolation --no-deps "git+https://github.com/Megvii-BaseDetection/YOLOX.git"
`--no-deps` because one of its deps, onnx-simplifier, vendors a test-data tree deep enough to
blow past Windows' MAX_PATH during its own build — installed loguru/tabulate/thop/ninja/tqdm/
tensorboard/pycocotools separately instead (the actual runtime deps `detect()` needs).

Preprocessing/postprocessing follow yolox's own tools/demo.py (ValTransform + postprocess) —
that's the reference implementation this adapter mirrors.
"""

from pathlib import Path
from typing import Any

import torch

from ..base import DetectorPlugin
from ..detection import Detection

WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "yolox_tiny" / "yolox_tiny.pth"
EXP_NAME = "yolox-tiny"
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45


class YOLOXTinyDetector(DetectorPlugin):
    def __init__(self) -> None:
        self._model = None
        self._exp = None
        self._preproc = None

    @property
    def name(self) -> str:
        return "yolox-tiny"

    @property
    def version(self) -> str:
        return "0.1.1rc0"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "input_size": 416,
            "num_params_m": 5.06,
            "gflops": 6.45,
            "published_coco_map": 32.8,
            "license": "Apache-2.0",
            "weights_path": str(WEIGHTS_PATH),
            "fine_tuned_on_rdd2022": False,
            "accuracy_caveat": "COCO-pretrained only — expect ~0% accuracy on RDD2022 classes until fine-tuned.",
        }

    def load(self) -> None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"YOLOX-Tiny weights not found at {WEIGHTS_PATH}.")
        from yolox.data.data_augment import ValTransform
        from yolox.exp import get_exp

        self._exp = get_exp(exp_name=EXP_NAME)
        self._exp.test_conf = CONFIDENCE_THRESHOLD
        self._exp.nmsthre = NMS_THRESHOLD
        self._model = self._exp.get_model()
        self._model.eval()
        ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=False)
        self._model.load_state_dict(ckpt["model"])
        self._preproc = ValTransform(legacy=False)

    def detect(self, image: Any) -> list[Detection]:
        if self._model is None:
            self.load()
        from yolox.data.datasets import COCO_CLASSES
        from yolox.utils import postprocess

        test_size = self._exp.test_size
        height, width = image.shape[:2]
        ratio = min(test_size[0] / height, test_size[1] / width)

        preprocessed, _ = self._preproc(image, None, test_size)
        tensor = torch.from_numpy(preprocessed).unsqueeze(0).float()

        with torch.no_grad():
            raw_outputs = self._model(tensor)
            outputs = postprocess(
                raw_outputs, self._exp.num_classes, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, class_agnostic=True
            )[0]

        if outputs is None:
            return []

        outputs = outputs.cpu()
        detections = []
        for row in outputs:
            x1, y1, x2, y2 = (row[0:4] / ratio).tolist()
            obj_conf, cls_conf, class_id = float(row[4]), float(row[5]), int(row[6])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=COCO_CLASSES[class_id],
                    confidence=obj_conf * cls_conf,
                    bbox_x=x1,
                    bbox_y=y1,
                    bbox_w=x2 - x1,
                    bbox_h=y2 - y1,
                    model_name=self.name,
                    model_version=self.version,
                )
            )
        return detections
