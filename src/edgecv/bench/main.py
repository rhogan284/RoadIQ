"""`python -m edgecv.bench.main` -- report bytes per kilometre for a run.

Success criterion 1 of the C3 proposal, and the Week 6 product milestone:
"first bytes-per-kilometre measured and recorded".

Lives in the package rather than in `scripts/` because the Dockerfile only
copies `src/`, and the blob store is a named docker volume -- so the harness has
to run in a container that mounts it (`make bench-bytes`), exactly like the
writer and the worker do.

Not written to `bench_runs` yet, though `bench_runs.bytes_stored` is the column
it eventually belongs in: that table is contract 4 and Ilana owns its shape.
Agreeing the grid point for a storage run comes before writing rows into it.
"""
from __future__ import annotations

import argparse

import psycopg

from edgecv.bench.bytes_per_km import TARGET_FACTOR, BytesPerKm
from edgecv.bench.collect import latest_run_id, measure_run, run_coverage
from edgecv.bench.coverage import CoverageReport
from edgecv.blobstore.store import BlobStore
from edgecv.config import Settings

_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


def human_bytes(n: float) -> str:
    """Binary units, because this is disk and stream size, not marketing."""
    size = float(n)
    for unit in _UNITS:
        if size < 1024 or unit == _UNITS[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def format_report(run_id: str, result: BytesPerKm, *,
                  coverage: CoverageReport, store_root: str) -> str:
    missing = coverage.frames_without_fix
    verdict = "PASS" if result.meets_target else "FAIL"
    factor = ("inf (nothing stored)" if result.reduction_factor == float("inf")
              else f"{result.reduction_factor:.1f}x")
    lines = [
        "RoadIQ -- bytes per kilometre (success criterion 1)",
        "=" * 58,
        f"  run_id          {run_id}",
        f"  frames          {result.n_frames}  "
        f"({result.n_frames - missing} with a GPS fix, {missing} without)",
        f"  distance        {result.distance_km:.3f} km",
        "",
        f"  stored          {human_bytes(result.stored_bytes):>10}"
        f"   {human_bytes(result.stored_bytes_per_km):>10} /km",
        f"  every frame     {human_bytes(result.baseline_bytes):>10}"
        f"   {human_bytes(result.baseline_bytes_per_km):>10} /km",
        "",
        f"  reduction       {factor}   (target {result.target_factor:.0f}x)"
        f"   {verdict}",
        "",
        "  -- coverage (success criterion 2) " + "-" * 24,
        f"  assessed        {coverage.assessed_m:>10.1f} m",
        f"  not assessed    {coverage.gap_m:>10.1f} m"
        f"   ({coverage.missing_frames} frame(s) lost in "
        f"{coverage.gap_count} gap(s))",
        f"  largest gap     {coverage.largest_gap_m:>10.1f} m",
        f"  coverage        {coverage.coverage_pct:>9.1f}%",
        "",
        f"  stored bytes measured from {store_root}",
    ]
    if missing:
        lines.append(
            f"  WARNING: {missing} frame(s) carried no fix and contributed no "
            f"distance, so the per-km figures are pessimistic."
        )
    return "\n".join(lines)


def main() -> None:
    settings = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None,
                    help="defaults to the most recently started run")
    ap.add_argument("--target", type=float, default=TARGET_FACTOR,
                    help="reduction factor the run must meet")
    args = ap.parse_args()

    store = BlobStore(root=settings.blob_root)
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        run_id = args.run_id or latest_run_id(conn)
        result = measure_run(conn, store, run_id, target_factor=args.target)
        coverage = run_coverage(conn, run_id)

    print(format_report(run_id, result, coverage=coverage,
                        store_root=str(settings.blob_root)))
    raise SystemExit(0 if result.meets_target else 1)


if __name__ == "__main__":
    main()
