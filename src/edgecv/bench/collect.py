"""Gather the bytes-per-kilometre inputs for a run out of Postgres and the store.

The arithmetic lives in `bytes_per_km.measure`; this is only the gathering, kept
separate so the headline formula has no database in it.

Where each number comes from, and why:

  distance      great-circle length of the run's GPS fixes, read from `frames`.
                Not `speed x duration` from `survey_runs.target_fps`: that is
                the rate we *asked* for, and a run that dropped frames or paced
                badly did not travel it.
  frame count   `frames` rows. Every frame gets one, clean ones included, which
                is exactly why the table can be the coverage denominator.
  mean frame    stat() of each frame's `source_ref`, weighted per frame rather
    size        than per distinct file, because a replayed image is a frame the
                real device would have had to store again.
  stored bytes  `BlobStore.total_bytes()`, not `sum(snippets.bytes)` -- that
                column is a 0 placeholder on every row (see
                `Repository._snippet_id`).

Known limitation, carried into the Week 8 milestone: `total_bytes()` measures the
whole store, so the figure is only per-run if the store holds one run. Snippets
are content-addressed with no run linkage -- `snippets` has no `run_id` and
cannot sensibly gain one, since dedup means one blob may serve many runs. Correct
per-run attribution needs a `run_snippets` join table written by the worker.
Until then, measure on a fresh store (or a before/after delta) and say so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psycopg

from edgecv.bench.bytes_per_km import TARGET_FACTOR, BytesPerKm, measure
from edgecv.bench.coverage import CoverageReport, coverage_from_rows
from edgecv.blobstore.store import BlobStore
from edgecv.geo import LatLon, path_length_m


@dataclass(frozen=True, slots=True)
class RunInputs:
    n_frames: int
    n_frames_with_fix: int
    distance_m: float
    mean_frame_bytes: float


def _assert_run_exists(conn: psycopg.Connection, run_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM survey_runs WHERE run_id = %s", (run_id,))
        if cur.fetchone() is None:
            raise LookupError(f"no survey run {run_id!r}")


def latest_run_id(conn: psycopg.Connection) -> str:
    """The most recently started survey run, so the harness can be run with no
    arguments straight after a demo."""
    with conn.cursor() as cur:
        cur.execute("SELECT run_id::text FROM survey_runs "
                    "ORDER BY started_at DESC, run_id DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            raise LookupError("no survey runs recorded -- run the feed first")
        return row[0]


def run_fixes(conn: psycopg.Connection, run_id: str) -> list[LatLon]:
    """The run's positions in capture order, skipping frames with no fix.

    Ordered by seq, not by captured_at: seq is the monotonic counter feed-sim
    assigns, while two frames can share a timestamp at high frame rates.
    """
    with conn.cursor() as cur:
        cur.execute(
            """SELECT lat, lon FROM frames
               WHERE run_id = %s AND lat IS NOT NULL AND lon IS NOT NULL
               ORDER BY seq""",
            (run_id,),
        )
        return [(float(lat), float(lon)) for lat, lon in cur.fetchall()]


def frame_counts(conn: psycopg.Connection, run_id: str) -> tuple[int, int]:
    """(frames, frames carrying a fix)."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT count(*), count(*) FILTER (WHERE lat IS NOT NULL)
               FROM frames WHERE run_id = %s""",
            (run_id,),
        )
        total, with_fix = cur.fetchone()
        return int(total), int(with_fix)


def mean_frame_bytes(conn: psycopg.Connection, run_id: str) -> float:
    """Mean encoded size of the frames this run replayed, in bytes.

    Weighted per frame: `source_ref` repeats when the sampler replays an image,
    and each replay is a frame the device would have had to store.

    Each distinct file is stat()ed once and the size reused, so a 10,000-frame
    run over 70 fixtures does 70 stat calls, not 10,000.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_ref, count(*) FROM frames WHERE run_id = %s "
            "GROUP BY source_ref",
            (run_id,),
        )
        rows = cur.fetchall()
    if not rows:
        raise LookupError(f"run {run_id!r} has no frames")

    total_bytes = 0
    total_frames = 0
    for source_ref, n in rows:
        path = Path(source_ref)
        if not path.is_file():
            # Silently skipping would shrink the baseline and flatter the
            # reduction factor, so this is fatal rather than a warning.
            raise FileNotFoundError(
                f"frame source {source_ref!r} is gone, so the every-frame "
                f"baseline cannot be measured for run {run_id!r}"
            )
        total_bytes += path.stat().st_size * n
        total_frames += n
    return total_bytes / total_frames


def collect_run(conn: psycopg.Connection, run_id: str) -> RunInputs:
    _assert_run_exists(conn, run_id)
    n_frames, n_with_fix = frame_counts(conn, run_id)
    return RunInputs(
        n_frames=n_frames,
        n_frames_with_fix=n_with_fix,
        distance_m=path_length_m(run_fixes(conn, run_id)),
        mean_frame_bytes=mean_frame_bytes(conn, run_id) if n_frames else 0.0,
    )


def run_coverage(conn: psycopg.Connection, run_id: str) -> CoverageReport:
    """Metres of road assessed vs lost for one run (success criterion 2).

    Reads every frame row including the ones with no fix, because the two kinds
    of absence mean opposite things — see `coverage.py`. A missing seq is road
    nobody assessed; a null lat on a present row is road we assessed but cannot
    place.
    """
    _assert_run_exists(conn, run_id)
    with conn.cursor() as cur:
        cur.execute("SELECT seq, lat, lon FROM frames WHERE run_id = %s "
                    "ORDER BY seq", (run_id,))
        rows = [(int(seq), None if lat is None else float(lat),
                 None if lon is None else float(lon))
                for seq, lat, lon in cur.fetchall()]
    return coverage_from_rows(rows)


def measure_run(conn: psycopg.Connection, store: BlobStore, run_id: str, *,
                target_factor: float = TARGET_FACTOR) -> BytesPerKm:
    """The Week 6 milestone figure for one run.

    Divides by the **assessed** distance, not the whole attempted track. The
    stored bytes and the every-frame baseline both come from the frames we
    actually processed, so the road those frames covered is the matching
    denominator. Spreading them over road nobody surveyed would report a smaller
    number than the truth — invisible in the ratio, since both sides divide by
    the same figure, but wrong in the per-kilometre absolutes that go in the
    report.
    """
    inputs = collect_run(conn, run_id)
    coverage = run_coverage(conn, run_id)
    return measure(
        stored_bytes=store.total_bytes(),
        n_frames=inputs.n_frames,
        mean_frame_bytes=inputs.mean_frame_bytes,
        distance_m=coverage.assessed_m,
        target_factor=target_factor,
    )
