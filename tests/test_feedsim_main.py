import json
import pytest
from edgecv.feedsim.main import load_pools, run_feed

pytestmark = pytest.mark.integration


@pytest.fixture
def fixtures(tmp_path):
    from scripts.make_fixtures import generate
    generate(tmp_path, n_clean=10, n_defect=5, size=64, seed=5)
    return tmp_path


def test_load_pools_reads_manifest(fixtures):
    clean, defect = load_pools(fixtures / "manifest.json")
    assert len(clean) == 10 and len(defect) == 5


def test_run_feed_publishes_requested_frame_count(rds, fixtures):
    stats = run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
                     n_frames=20, fps=200, prevalence=0.2, maxlen=1000, seed=1,
                     stream="frames")
    assert stats.offered == 20
    assert stats.dropped == 0
    assert rds.xlen("frames") == 20


def test_run_feed_assigns_monotonic_seq(rds, fixtures):
    run_feed(rds, manifest=fixtures / "manifest.json", run_id=None, n_frames=10,
             fps=200, prevalence=0.2, maxlen=1000, seed=1, stream="frames")
    from edgecv.contracts.frame import FrameEnvelope
    seqs = [FrameEnvelope.from_fields(f).seq for _i, f in rds.xrange("frames")]
    assert seqs == list(range(10))


def test_run_feed_drops_when_maxlen_reached(rds, fixtures):
    stats = run_feed(rds, manifest=fixtures / "manifest.json", run_id=None,
                     n_frames=20, fps=500, prevalence=0.2, maxlen=5, seed=1,
                     stream="frames")
    assert stats.offered == 20
    assert stats.dropped == 15
    assert rds.xlen("frames") == 5


def test_run_feed_is_deterministic_for_a_seed(rds, fixtures):
    from edgecv.contracts.frame import FrameEnvelope
    def refs():
        rds.flushdb()
        run_feed(rds, manifest=fixtures / "manifest.json", run_id=None, n_frames=15,
                 fps=500, prevalence=0.3, maxlen=1000, seed=99, stream="frames")
        return [FrameEnvelope.from_fields(f).source_ref
                for _i, f in rds.xrange("frames")]
    assert refs() == refs()
