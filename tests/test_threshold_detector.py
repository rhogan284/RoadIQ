import cv2
import numpy as np
import pytest
from edgecv.contracts.detection import Detector
from edgecv.detectors.threshold import ThresholdDetector
from scripts.make_fixtures import generate


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("fx"), n_clean=5, n_defect=5,
                    size=256, seed=11)


def load(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def test_satisfies_detector_protocol():
    assert isinstance(ThresholdDetector(), Detector)


def test_info_has_name_version_and_params():
    info = ThresholdDetector(min_area=40).info
    assert info.name == "threshold"
    assert info.version
    assert info.params["min_area"] == 40


def test_finds_the_defect_in_a_defect_image(manifest):
    detector = ThresholdDetector()
    detections = detector.detect(load(manifest["defect"][0]["path"]))
    assert len(detections) >= 1


def test_finds_nothing_in_a_clean_image(manifest):
    detector = ThresholdDetector()
    assert detector.detect(load(manifest["clean"][0]["path"])) == []


def test_detected_bbox_overlaps_ground_truth(manifest):
    entry = manifest["defect"][0]
    detection = ThresholdDetector().detect(load(entry["path"]))[0]
    b = detection.bbox
    assert b.x <= entry["x"] + entry["w"] and entry["x"] <= b.x + b.w
    assert b.y <= entry["y"] + entry["h"] and entry["y"] <= b.y + b.h


def test_bbox_stays_inside_the_frame(manifest):
    image = load(manifest["defect"][0]["path"])
    height, width = image.shape
    for detection in ThresholdDetector().detect(image):
        b = detection.bbox
        assert 0 <= b.x and b.x + b.w <= width
        assert 0 <= b.y and b.y + b.h <= height


def test_min_area_filters_small_blobs(manifest):
    image = load(manifest["defect"][0]["path"])
    assert ThresholdDetector(min_area=1_000_000).detect(image) == []


def test_severity_is_populated(manifest):
    detections = ThresholdDetector().detect(load(manifest["defect"][0]["path"]))
    assert detections[0].severity in {"minor", "major", "critical"}
