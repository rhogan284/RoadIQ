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


# --- position pass-through ---------------------------------------------------
#
# Contract 2 says the worker copies the frame envelope's position and clock
# fields onto every result, and `Repository.write_results` inserts them into
# `frames`. The worker did not actually copy them. Nothing caught it because the
# fixtures carried no fix, so the column was null either way -- the bug only
# became visible once feed-sim started emitting a synthetic GPS track and
# `frames_with_fix` came back 0 out of 600 on a real run.

def _envelope_with_fix(path: str, seq: int = 0):
    from dataclasses import replace
    from edgecv.feedsim.gpstrack import SyntheticTrack

    track = SyntheticTrack(start=(-33.8688, 151.2093), bearing_deg=90.0,
                           speed_mps=13.89, fps=15.0)
    fix = track.fix_for(seq)
    return replace(envelope_for(path, seq),
                   lat=fix.lat, lon=fix.lon, heading_deg=fix.heading_deg,
                   speed_mps=fix.speed_mps, gps_accuracy_m=fix.gps_accuracy_m,
                   capture_mono_ns=123_456_789,
                   device_boot_id="22222222-2222-2222-2222-222222222222")


def test_result_carries_the_frames_position_on_a_defect_frame(fixtures, blobstore):
    _d, manifest = fixtures
    envelope = _envelope_with_fix(manifest["defect"][0]["path"], seq=7)
    result = process_one(envelope, blobstore=blobstore, worker_id="w1")
    assert result.lat == pytest.approx(envelope.lat)
    assert result.lon == pytest.approx(envelope.lon)
    assert result.heading_deg == pytest.approx(envelope.heading_deg)
    assert result.speed_mps == pytest.approx(envelope.speed_mps)
    assert result.gps_accuracy_m == pytest.approx(envelope.gps_accuracy_m)


def test_result_carries_the_position_on_a_clean_frame_too(fixtures, blobstore):
    """Clean frames are the coverage denominator, so losing their position loses
    the distance -- the exact failure that blocked the Week 6 milestone."""
    _d, manifest = fixtures
    envelope = _envelope_with_fix(manifest["clean"][0]["path"], seq=3)
    result = process_one(envelope, blobstore=blobstore, worker_id="w1")
    assert result.lat == pytest.approx(envelope.lat)
    assert result.lon == pytest.approx(envelope.lon)


def test_result_carries_the_monotonic_clock_and_boot_id(fixtures, blobstore):
    _d, manifest = fixtures
    envelope = _envelope_with_fix(manifest["clean"][0]["path"])
    result = process_one(envelope, blobstore=blobstore, worker_id="w1")
    assert result.capture_mono_ns == 123_456_789
    assert result.device_boot_id == "22222222-2222-2222-2222-222222222222"


def test_position_is_none_when_the_envelope_had_no_fix(fixtures, blobstore):
    _d, manifest = fixtures
    result = process_one(envelope_for(manifest["clean"][0]["path"]),
                         blobstore=blobstore, worker_id="w1")
    assert result.lat is None and result.lon is None
    assert result.capture_mono_ns is None


def test_position_survives_a_frame_that_fails_to_decode(fixtures, blobstore,
                                                        tmp_path):
    """A failed frame still gets a frames row, and that row is still part of the
    coverage denominator, so it still needs its position."""
    from dataclasses import replace
    _d, manifest = fixtures
    broken = tmp_path / "not-an-image.png"
    broken.write_bytes(b"garbage")
    envelope = replace(_envelope_with_fix(manifest["clean"][0]["path"]),
                       source_ref=str(broken), path=str(broken))
    result = process_one(envelope, blobstore=blobstore, worker_id="w1")
    assert result.status == "failed"
    assert result.lat == pytest.approx(envelope.lat)
