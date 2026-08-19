import json

import cv2

from scripts.make_fixtures import generate


def test_generates_requested_counts(tmp_path):
    manifest = generate(tmp_path, n_clean=5, n_defect=3, size=128, seed=7)
    assert len(list((tmp_path / "clean").glob("*.png"))) == 5
    assert len(list((tmp_path / "defect").glob("*.png"))) == 3
    assert len(manifest["defect"]) == 3


def test_manifest_records_ground_truth_bbox(tmp_path):
    manifest = generate(tmp_path, n_clean=1, n_defect=1, size=128, seed=7)
    entry = manifest["defect"][0]
    for key in ("path", "x", "y", "w", "h"):
        assert key in entry
    assert entry["w"] > 0 and entry["h"] > 0


def test_manifest_written_to_disk(tmp_path):
    generate(tmp_path, n_clean=1, n_defect=1, size=128, seed=7)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert "defect" in on_disk and "clean" in on_disk


def test_deterministic_for_same_seed(tmp_path):
    a = generate(tmp_path / "a", n_clean=2, n_defect=2, size=128, seed=42)
    b = generate(tmp_path / "b", n_clean=2, n_defect=2, size=128, seed=42)
    assert [d["x"] for d in a["defect"]] == [d["x"] for d in b["defect"]]


def test_defect_region_is_darker_than_surroundings(tmp_path):
    manifest = generate(tmp_path, n_clean=0, n_defect=1, size=128, seed=7)
    entry = manifest["defect"][0]
    img = cv2.imread(entry["path"], cv2.IMREAD_GRAYSCALE)
    patch = img[entry["y"]:entry["y"] + entry["h"], entry["x"]:entry["x"] + entry["w"]]
    assert patch.mean() < img.mean() - 20
