from datetime import datetime, timezone
import pytest
from edgecv.contracts.detection import (
    BBox, Detection, DetectorInfo, InferenceResult, severity_for,
)

def test_bbox_area():
    assert BBox(x=10, y=20, w=4, h=5).area == 20

def test_bbox_rejects_negative_size():
    with pytest.raises(ValueError):
        BBox(x=0, y=0, w=0, h=5)

def test_severity_thresholds():
    assert severity_for(50) == "minor"
    assert severity_for(500) == "major"
    assert severity_for(5000) == "critical"

def test_params_hash_is_order_independent():
    a = DetectorInfo(name="threshold", version="1.0.0", params={"k": 3, "min_area": 40})
    b = DetectorInfo(name="threshold", version="1.0.0", params={"min_area": 40, "k": 3})
    assert a.params_hash == b.params_hash
    assert len(a.params_hash) == 16

def test_params_hash_changes_with_params():
    a = DetectorInfo(name="threshold", version="1.0.0", params={"k": 3})
    b = DetectorInfo(name="threshold", version="1.0.0", params={"k": 4})
    assert a.params_hash != b.params_hash

def _result(**kw):
    base = dict(
        run_id="11111111-1111-1111-1111-111111111111",
        seq=3,
        captured_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        width=256, height=256,
        source_ref="fixtures/defect_001.png",
        frame_sha256="b" * 64,
        detector=DetectorInfo(name="threshold", version="1.0.0", params={"k": 3}),
        worker_id="worker-1",
        started_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        latency_ms=12.5,
        status="ok",
        error=None,
        detections=[Detection(defect_class="scratch", confidence=0.9,
                              bbox=BBox(x=1, y=2, w=10, h=10), severity="minor")],
        snippet_sha256s=["c" * 64],
        thumbnail_sha256="d" * 64,
    )
    base.update(kw)
    return InferenceResult(**base)

def test_inference_result_json_roundtrip():
    result = _result()
    assert InferenceResult.from_json(result.to_json()) == result

def test_clean_frame_has_no_detections():
    result = _result(detections=[], snippet_sha256s=[], thumbnail_sha256=None)
    assert InferenceResult.from_json(result.to_json()).detections == []

def test_snippet_list_must_align_with_detections():
    with pytest.raises(ValueError, match="align"):
        _result(snippet_sha256s=[])
