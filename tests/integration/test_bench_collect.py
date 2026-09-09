"""Gathering the three bytes-per-kilometre inputs out of a real database.

Frames are inserted directly rather than driven through the pipeline: what is
under test is the SQL and the distance arithmetic, and a full pipeline run would
make a failure here ambiguous between the two.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from edgecv.bench.collect import collect_run, mean_frame_bytes, run_fixes
from edgecv.blobstore.store import BlobStore
from edgecv.feedsim.gpstrack import SyntheticTrack

pytestmark = pytest.mark.integration

SYD = (-33.8688, 151.2093)
FPS = 10.0
SPEED = 13.89


@pytest.fixture
def track():
    return SyntheticTrack(start=SYD, bearing_deg=90.0, speed_mps=SPEED, fps=FPS)


@pytest.fixture
def sources(tmp_path):
    """Three fixture images of known, differing sizes."""
    sizes = (1_000, 2_000, 3_000)
    paths = []
    for i, size in enumerate(sizes):
        p = tmp_path / f"frame_{i}.png"
        p.write_bytes(b"\0" * size)
        paths.append(str(p))
    return paths


def _seed_run(conn, run_id, *, track, sources, n_frames, with_fix=True):
    start = datetime(2026, 9, 9, 3, 0, tzinfo=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO survey_runs (run_id, authority_id, started_at,
                   source_kind, source_ref, target_fps, prevalence, transport)
               VALUES (%s, 'test-council', %s, 'synthetic', 'manifest.json',
                       %s, 0.2, 'reference')""",
            (run_id, start, FPS),
        )
        for seq in range(n_frames):
            fix = track.fix_for(seq) if with_fix else None
            cur.execute(
                """INSERT INTO frames (run_id, seq, captured_at, enqueued_at,
                       width, height, source_ref, sha256, lat, lon,
                       heading_deg, speed_mps, gps_accuracy_m)
                   VALUES (%s, %s, %s, %s, 64, 64, %s, %s, %s, %s, %s, %s, %s)""",
                (run_id, seq, start + timedelta(seconds=seq / FPS),
                 start + timedelta(seconds=seq / FPS),
                 sources[seq % len(sources)], f"{seq:064d}",
                 fix.lat if fix else None, fix.lon if fix else None,
                 fix.heading_deg if fix else None,
                 fix.speed_mps if fix else None,
                 fix.gps_accuracy_m if fix else None),
            )


def test_run_fixes_come_back_in_seq_order(clean_db, track, sources):
    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=50)
    fixes = run_fixes(clean_db, run_id)
    assert len(fixes) == 50
    # Heading due east, so longitude must increase monotonically.
    lons = [lon for _lat, lon in fixes]
    assert lons == sorted(lons)


def test_distance_matches_the_track_that_generated_the_frames(clean_db, track,
                                                              sources):
    """The end-to-end check on the denominator: what the database gives back has
    to agree with speed x time to within GPS rounding."""
    run_id = str(uuid.uuid4())
    n = 601  # 60 s at 10 fps, so 600 intervals
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=n)
    inputs = collect_run(clean_db, run_id)
    assert inputs.distance_m == pytest.approx(SPEED * 60.0, rel=1e-3)
    assert inputs.n_frames == n
    assert inputs.n_frames_with_fix == n


def test_frames_without_a_fix_are_counted_but_not_measured(clean_db, track,
                                                           sources):
    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=20,
              with_fix=False)
    inputs = collect_run(clean_db, run_id)
    assert inputs.n_frames == 20
    assert inputs.n_frames_with_fix == 0
    assert inputs.distance_m == 0.0


def test_mean_frame_bytes_is_the_mean_over_frames_not_over_files(clean_db, track,
                                                                 sources):
    """Sources are 1000/2000/3000 bytes and 4 frames cycle through them, so the
    mean per *frame* is (1000+2000+3000+1000)/4 = 1750 -- not the 2000 you get
    by averaging the three distinct files. Replay weighting is the point."""
    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=4)
    assert mean_frame_bytes(clean_db, run_id) == pytest.approx(1750.0)


def test_a_missing_source_file_is_an_error_not_a_silent_skip(clean_db, track,
                                                             sources, tmp_path):
    """Skipping a deleted fixture would quietly shrink the baseline and flatter
    the reduction factor."""
    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=3)
    (tmp_path / "frame_1.png").unlink()
    with pytest.raises(FileNotFoundError, match="frame_1.png"):
        mean_frame_bytes(clean_db, run_id)


def test_an_unknown_run_is_an_error(clean_db):
    with pytest.raises(LookupError, match="run"):
        collect_run(clean_db, str(uuid.uuid4()))


def test_measure_run_produces_the_headline_figure(clean_db, track, sources,
                                                  tmp_path):
    from edgecv.bench.collect import measure_run

    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=601)
    store = BlobStore(root=tmp_path / "blobs")
    store.put(b"x" * 4_000, kind="crop", fmt="png", width=8, height=8)

    result = measure_run(clean_db, store, run_id)

    # 601 frames over sources of 1000/2000/3000 bytes, cycled by seq % 3:
    # 201 x 1000 + 200 x 2000 + 200 x 3000 = 1,201,000 bytes to keep them all.
    baseline = 201 * 1_000 + 200 * 2_000 + 200 * 3_000

    assert result.stored_bytes == 4_000
    assert result.distance_km == pytest.approx(0.83334, rel=1e-3)
    assert result.baseline_bytes == baseline
    assert result.reduction_factor == pytest.approx(baseline / 4_000, rel=1e-6)
    assert result.meets_target is True


def test_latest_run_id_returns_the_most_recently_started_run(clean_db, track,
                                                              sources):
    from edgecv.bench.collect import latest_run_id

    older, newer = str(uuid.uuid4()), str(uuid.uuid4())
    _seed_run(clean_db, older, track=track, sources=sources, n_frames=2)
    _seed_run(clean_db, newer, track=track, sources=sources, n_frames=2)
    with clean_db.cursor() as cur:
        cur.execute("UPDATE survey_runs SET started_at = %s WHERE run_id = %s",
                    (datetime(2026, 9, 9, 5, 0, tzinfo=timezone.utc), newer))
    assert latest_run_id(clean_db) == newer


def test_latest_run_id_on_an_empty_database_is_an_error(clean_db):
    from edgecv.bench.collect import latest_run_id

    with clean_db.cursor() as cur:
        cur.execute("TRUNCATE survey_runs CASCADE")
    with pytest.raises(LookupError, match="no survey runs"):
        latest_run_id(clean_db)


# --- coverage gaps in metres (success criterion 2) --------------------------

def test_run_coverage_reports_a_dropped_frame_as_metres_not_assessed(clean_db,
                                                                     track,
                                                                     sources):
    from edgecv.bench.collect import run_coverage

    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=101)
    # Drop seq 50, as a full stream buffer would have.
    with clean_db.cursor() as cur:
        cur.execute("DELETE FROM frames WHERE run_id = %s AND seq = 50", (run_id,))

    report = run_coverage(clean_db, run_id)

    assert report.frames == 100
    assert report.missing_frames == 1
    assert report.gap_count == 1
    # Two frame intervals at 13.89 m/s and 10 fps.
    assert report.gap_m == pytest.approx(2 * SPEED / FPS, rel=1e-3)
    assert report.gaps[0].after_seq == 49
    assert report.gaps[0].before_seq == 51


def test_run_coverage_on_a_clean_run_has_no_gaps(clean_db, track, sources):
    from edgecv.bench.collect import run_coverage

    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=101)
    report = run_coverage(clean_db, run_id)
    assert report.gap_count == 0
    assert report.coverage_pct == pytest.approx(100.0)
    assert report.assessed_m == pytest.approx(SPEED * 10.0, rel=1e-3)


def test_bytes_per_km_divides_by_assessed_road_not_attempted(clean_db, track,
                                                             sources, tmp_path):
    """Stored bytes come from the frames we actually processed, so the per-km
    figure has to divide by the road those frames covered. Dividing by the whole
    attempted track would spread the bytes over road nobody surveyed and report a
    smaller number than the truth."""
    from edgecv.bench.collect import measure_run, run_coverage

    run_id = str(uuid.uuid4())
    _seed_run(clean_db, run_id, track=track, sources=sources, n_frames=101)
    # A 10-frame hole: 9 intervals of road attempted but never assessed.
    with clean_db.cursor() as cur:
        cur.execute("DELETE FROM frames WHERE run_id = %s AND seq BETWEEN 40 AND 49",
                    (run_id,))

    coverage = run_coverage(clean_db, run_id)
    store = BlobStore(root=tmp_path / "blobs")
    store.put(b"x" * 1_000, kind="crop", fmt="png", width=8, height=8)

    result = measure_run(clean_db, store, run_id)

    assert coverage.gap_m > 0                       # there really is a hole
    assert result.distance_km == pytest.approx(coverage.assessed_m / 1000, rel=1e-9)
    assert result.distance_km < coverage.attempted_m / 1000
