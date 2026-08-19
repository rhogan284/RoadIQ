import pytest
from edgecv.contracts.detection import InferenceResult
from edgecv.worker.main import process_one

pytestmark = pytest.mark.integration


@pytest.fixture
def fixtures(tmp_path_factory):
    from scripts.make_fixtures import generate
    d = tmp_path_factory.mktemp("wfx")
    return d, generate(d, n_clean=3, n_defect=3, size=128, seed=13)


@pytest.fixture
def blobstore(tmp_path):
    from edgecv.blobstore.store import BlobStore
    return BlobStore(root=tmp_path / "blobs")


def envelope_for(path: str, seq: int = 0):
    import hashlib
    from datetime import datetime, timezone
    from pathlib import Path as P
    from edgecv.contracts.frame import FrameEnvelope
    import cv2
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    return FrameEnvelope(
        run_id="11111111-1111-1111-1111-111111111111", seq=seq,
        captured_at=datetime.now(timezone.utc), width=int(w), height=int(h),
        source_ref=path, sha256=hashlib.sha256(P(path).read_bytes()).hexdigest(),
        transport="reference", path=path,
    )


def test_defect_frame_yields_detections_and_snippets(fixtures, blobstore):
    _d, manifest = fixtures
    path = manifest["defect"][0]["path"]
    result = process_one(envelope_for(path), blobstore=blobstore,
                         worker_id="w1")
    assert isinstance(result, InferenceResult)
    assert result.status == "ok"
    assert len(result.detections) >= 1
    assert len(result.snippet_sha256s) == len(result.detections)
    assert result.thumbnail_sha256 is not None


def test_clean_frame_stores_nothing(fixtures, blobstore):
    _d, manifest = fixtures
    result = process_one(envelope_for(manifest["clean"][0]["path"]),
                         blobstore=blobstore, worker_id="w1")
    assert result.detections == []
    assert result.snippet_sha256s == []
    assert result.thumbnail_sha256 is None


def test_crops_are_written_to_the_blobstore(fixtures, blobstore):
    _d, manifest = fixtures
    result = process_one(envelope_for(manifest["defect"][0]["path"]),
                         blobstore=blobstore, worker_id="w1")
    for sha in result.snippet_sha256s:
        assert blobstore.exists(sha, kind="crop", fmt="webp")


def test_crop_is_far_smaller_than_the_source_frame(fixtures, blobstore):
    from pathlib import Path
    _d, manifest = fixtures
    path = manifest["defect"][0]["path"]
    result = process_one(envelope_for(path), blobstore=blobstore, worker_id="w1")
    crop_bytes = len(blobstore.get(result.snippet_sha256s[0], kind="crop",
                                   fmt="webp"))
    assert crop_bytes < Path(path).stat().st_size


def test_undecodable_frame_returns_failed_status(blobstore, tmp_path):
    bogus = tmp_path / "not_an_image.png"
    bogus.write_bytes(b"not an image")
    from datetime import datetime, timezone
    from edgecv.contracts.frame import FrameEnvelope
    envelope = FrameEnvelope(
        run_id="11111111-1111-1111-1111-111111111111", seq=1,
        captured_at=datetime.now(timezone.utc), width=1, height=1,
        source_ref=str(bogus), sha256="a" * 64, transport="reference",
        path=str(bogus),
    )
    result = process_one(envelope, blobstore=blobstore, worker_id="w1")
    assert result.status == "failed"
    assert result.error
    assert result.detections == []


def test_identical_frames_produce_identical_crop_hashes(fixtures, blobstore):
    _d, manifest = fixtures
    path = manifest["defect"][0]["path"]
    a = process_one(envelope_for(path, seq=1), blobstore=blobstore, worker_id="w1")
    b = process_one(envelope_for(path, seq=2), blobstore=blobstore, worker_id="w1")
    assert a.snippet_sha256s == b.snippet_sha256s
