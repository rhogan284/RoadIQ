"""Runs each requested detector plugin against the RDD2022 test split and reports, per model:
accuracy (mAP@0.5, precision, recall) and speed (ms/image, images/sec).

This is the same harness for every model in the registry — it only depends on the
DetectorPlugin.detect() -> list[Detection] contract (Contract 2), so it's a fair comparison
regardless of which framework each plugin uses underneath.

Usage:
    PYTHONPATH=. python eval/benchmark.py                       # all wired-up detectors, 500 test images
    PYTHONPATH=. python eval/benchmark.py --models yolo11n       # just one
    PYTHONPATH=. python eval/benchmark.py --limit 0              # full test split (slow on CPU)
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from detector_interface import AVAILABLE_DETECTORS, get_detector
from detector_interface.detection import Detection

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "data" / "RDD2022" / "clean" / "test" / "images"
TEST_LABELS = ROOT / "data" / "RDD2022" / "clean" / "test" / "labels"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
IOU_THRESHOLD = 0.5


def load_ground_truth(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_id, cx, cy, w, h = line.split()
        class_id = int(class_id)
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append((class_id, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(recall: list[float], precision: list[float]) -> float:
    """Standard all-point interpolated AP (area under the precision envelope)."""
    recall = [0.0] + recall + [1.0]
    precision = [0.0] + precision + [0.0]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return ap


def evaluate_detections(
    all_predictions: list[list[Detection]],
    all_ground_truths: list[list[tuple[int, tuple[float, float, float, float]]]],
) -> dict:
    class_ids = {gt_class for gts in all_ground_truths for gt_class, _ in gts}
    per_class_ap = {}
    total_tp, total_fp, total_gt = 0, 0, sum(len(gts) for gts in all_ground_truths)

    for class_id in sorted(class_ids):
        scored = []  # (confidence, is_tp)
        num_gt_this_class = sum(1 for gts in all_ground_truths for c, _ in gts if c == class_id)
        if num_gt_this_class == 0:
            continue

        for preds, gts in zip(all_predictions, all_ground_truths):
            gt_boxes = [box for c, box in gts if c == class_id]
            matched = [False] * len(gt_boxes)
            class_preds = sorted(
                [p for p in preds if p.class_id == class_id], key=lambda p: p.confidence, reverse=True
            )
            for p in class_preds:
                p_box = (p.bbox_x, p.bbox_y, p.bbox_x + p.bbox_w, p.bbox_y + p.bbox_h)
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    current_iou = iou(p_box, gt_box)
                    if current_iou > best_iou:
                        best_iou, best_j = current_iou, j
                is_tp = best_iou >= IOU_THRESHOLD
                if is_tp:
                    matched[best_j] = True
                scored.append((p.confidence, is_tp))

        scored.sort(key=lambda x: x[0], reverse=True)
        tp_cum, fp_cum = 0, 0
        recalls, precisions = [], []
        for _, is_tp in scored:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            recalls.append(tp_cum / num_gt_this_class)
            precisions.append(tp_cum / (tp_cum + fp_cum))
        total_tp += tp_cum
        total_fp += fp_cum
        per_class_ap[class_id] = average_precision(recalls, precisions) if scored else 0.0

    map50 = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    return {"map50": map50, "precision": precision, "recall": recall, "per_class_ap50": per_class_ap}


def benchmark_model(model_name: str, image_paths: list[Path]) -> dict:
    detector = get_detector(model_name)
    all_predictions, all_ground_truths = [], []
    durations_ms = []

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        start = time.perf_counter()
        try:
            predictions = detector.detect(image)
        except NotImplementedError as e:
            return {"error": str(e)}
        elapsed_ms = (time.perf_counter() - start) * 1000
        if i > 0:  # skip first call — includes model load/warm-up cost
            durations_ms.append(elapsed_ms)

        height, width = image.shape[:2]
        label_path = TEST_LABELS / (image_path.stem + ".txt")
        all_predictions.append(predictions)
        all_ground_truths.append(load_ground_truth(label_path, width, height))

    accuracy = evaluate_detections(all_predictions, all_ground_truths)
    avg_ms = sum(durations_ms) / len(durations_ms) if durations_ms else 0.0
    return {
        "num_images": len(image_paths),
        "avg_ms_per_image": avg_ms,
        "images_per_sec": 1000 / avg_ms if avg_ms > 0 else 0.0,
        "params": detector.params,
        "version": detector.version,
        **accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(AVAILABLE_DETECTORS), choices=list(AVAILABLE_DETECTORS))
    parser.add_argument("--limit", type=int, default=500, help="Number of test images to use (0 = full test split)")
    args = parser.parse_args()

    if not TEST_IMAGES.exists():
        raise RuntimeError(f"{TEST_IMAGES} not found — run training/prepare_dataset.py first.")

    image_paths = sorted(TEST_IMAGES.glob("*"))
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"Evaluating on {len(image_paths)} test images.\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for model_name in args.models:
        print(f"--- {model_name} ---")
        result = benchmark_model(model_name, image_paths)
        results[model_name] = result
        if "error" in result:
            print(f"  skipped: {result['error']}")
            continue
        print(f"  mAP@0.5:   {result['map50']:.4f}")
        print(f"  precision: {result['precision']:.4f}")
        print(f"  recall:    {result['recall']:.4f}")
        print(f"  speed:     {result['avg_ms_per_image']:.1f} ms/image ({result['images_per_sec']:.1f} img/s)")
        print()

    out_path = RESULTS_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
