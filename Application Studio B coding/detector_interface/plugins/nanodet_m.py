"""NanoDet-m adapter. Official figures (RangiLyu/nanodet): 320x320 input, ShuffleNetV2 1.0x
backbone, 20.6 COCO mAP, 0.95M params, 0.72 GFLOPs — the smallest-footprint candidate in the
R-S1 survey. Apache-2.0 licensed. COCO-pretrained only — same accuracy caveat as yolov4_tiny.py
and the YOLOX plugins: expect ~0% accuracy against RDD2022 ground truth until fine-tuned.

Weights: weights/nanodet_m/nanodet_m.ckpt (RangiLyu/nanodet's Google Drive model zoo link).

Unlike the other four models, there's no usable `nanodet` PyPI package, and even a pip install
straight from GitHub silently produces a broken wheel: nanodet/model/ and nanodet/data/ have no
__init__.py in the upstream repo, so setuptools' find_packages() drops them from the build
entirely (confirmed empirically — `pip install git+...` "succeeds" but `import nanodet.model`
then fails). So this plugin instead vendors the repo as a git clone at
third_party/nanodet/ (this project's root, not weights/ — it's source code, not a downloaded
weight) and imports it via sys.path, which works because Python 3's implicit namespace packages
don't need __init__.py when importing straight off a path:
    git clone --depth 1 https://github.com/RangiLyu/nanodet.git third_party/nanodet
That clone also supplies config/legacy_v0.x_configs/nanodet-m.yml, the architecture config this
checkpoint needs (COCO 80-class, matches the ckpt's state_dict).

Two more upstream quirks worked around here:
- nanodet/data/collate.py does `from torch._six import string_classes` at import time; `torch._six`
  was removed years ago (we're on torch 2.13.0). A tiny shim module is installed into
  sys.modules before that import runs — string_classes was always just `str` in Python 3, so
  this is exactly what real `torch._six` did on this Python version anyway.
- nanodet.util.Logger unconditionally writes a logs.txt (and hijacks the root logging config)
  wherever save_dir points — pointed at weights/nanodet_m/_load_logs so it doesn't dump into
  the project root, but the root-logger hijack itself isn't avoidable without patching the
  library.

Extra runtime deps beyond requirements.txt: pytorch_lightning, termcolor (nanodet's own utils
import both; requirements.txt doesn't pin exact versions since they're only needed for this one
model, `pip install pytorch_lightning termcolor` is enough).
"""

import sys
import types
from pathlib import Path
from typing import Any

import torch

from ..base import DetectorPlugin
from ..detection import Detection

REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "nanodet"
CONFIG_PATH = REPO_DIR / "config" / "legacy_v0.x_configs" / "nanodet-m.yml"
WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "nanodet_m" / "nanodet_m.ckpt"
LOAD_LOG_DIR = WEIGHTS_PATH.parent / "_load_logs"
CONFIDENCE_THRESHOLD = 0.25


def _ensure_importable() -> None:
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    if "torch._six" not in sys.modules:
        shim = types.ModuleType("torch._six")
        shim.string_classes = str
        sys.modules["torch._six"] = shim


class NanoDetMDetector(DetectorPlugin):
    def __init__(self) -> None:
        self._model = None
        self._cfg = None
        self._pipeline = None

    @property
    def name(self) -> str:
        return "nanodet-m"

    @property
    def version(self) -> str:
        return "1.0x-320"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "input_size": 320,
            "backbone": "ShuffleNetV2 1.0x",
            "num_params_m": 0.95,
            "gflops": 0.72,
            "published_coco_map": 20.6,
            "license": "Apache-2.0",
            "weights_path": str(WEIGHTS_PATH),
            "fine_tuned_on_rdd2022": False,
            "accuracy_caveat": "COCO-pretrained only — expect ~0% accuracy on RDD2022 classes until fine-tuned.",
        }

    def load(self) -> None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"NanoDet-m weights not found at {WEIGHTS_PATH}.")
        if not REPO_DIR.exists():
            raise FileNotFoundError(
                f"NanoDet source not found at {REPO_DIR}. Run: "
                f"git clone --depth 1 https://github.com/RangiLyu/nanodet.git {REPO_DIR}"
            )
        _ensure_importable()
        from nanodet.data.transform import Pipeline
        from nanodet.model.arch import build_model
        from nanodet.util import Logger, cfg, load_config, load_model_weight

        load_config(cfg, str(CONFIG_PATH))
        self._cfg = cfg
        logger = Logger(-1, save_dir=str(LOAD_LOG_DIR), use_tensorboard=False)
        model = build_model(cfg.model)
        ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=False)
        load_model_weight(model, ckpt, logger)
        model.eval()
        self._model = model
        self._pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def detect(self, image: Any) -> list[Detection]:
        if self._model is None:
            self.load()
        _ensure_importable()
        from nanodet.data.batch_process import stack_batch_img
        from nanodet.data.collate import naive_collate

        height, width = image.shape[:2]
        meta = dict(img_info={"id": 0, "height": height, "width": width}, raw_img=image, img=image)
        meta = self._pipeline(None, meta, self._cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        with torch.no_grad():
            results = self._model.inference(meta)[0]

        detections = []
        for class_id, boxes in results.items():
            for x1, y1, x2, y2, score in boxes:
                if score < CONFIDENCE_THRESHOLD:
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self._cfg.class_names[class_id],
                        confidence=float(score),
                        bbox_x=float(x1),
                        bbox_y=float(y1),
                        bbox_w=float(x2 - x1),
                        bbox_h=float(y2 - y1),
                        model_name=self.name,
                        model_version=self.version,
                    )
                )
        return detections
