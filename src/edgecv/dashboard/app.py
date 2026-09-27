"""Throwaway Streamlit coverage panel.

Query functions are kept free of Streamlit so they stay unit-testable without
a browser. This page is replaced in Week 9 by the map, worst-N segments and
review queue behind the read API -- do not invest in it beyond the coverage
panel spec §5 actually asks for.
"""
from __future__ import annotations

import psycopg


def run_options(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id::text, authority_id, started_at, target_fps "
            "FROM survey_runs ORDER BY started_at DESC"
        )
        return [{"run_id": r[0], "authority_id": r[1], "started_at": r[2],
                 "target_fps": float(r[3])} for r in cur.fetchall()]


def coverage_series(conn: psycopg.Connection, run_id: str) -> list[dict]:
    """Spec §5 view 6. `frames_ingested` is the denominator."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bucket, frames_ingested, frames_processed, frames_flagged, "
            "       frames_without_fix "
            "FROM run_coverage_1min WHERE run_id = %s ORDER BY bucket",
            (run_id,),
        )
        return [{"bucket": r[0], "frames_ingested": r[1], "frames_processed": r[2],
                 "frames_flagged": r[3], "frames_without_fix": r[4]}
                for r in cur.fetchall()]


def _bus_panel(st, settings) -> None:
    """Live bus health. Separate from the Postgres panels on purpose: those show
    what already landed, this shows what is still in flight, which is the half
    that was invisible until now.

    Wrapped in try/except because the dashboard must still render the coverage
    panel when Redis is unreachable — the survey data is in Postgres and does not
    depend on the bus being up to be readable."""
    import redis

    from edgecv.bus.observe import consumer_health, group_health, stuck_entries

    st.subheader("Bus health")
    try:
        client = redis.from_url(settings.redis_url, decode_responses=False)
        health = group_health(client, stream_name=settings.frames_stream,
                              group="workers")
    except Exception as exc:  # noqa: BLE001 - a panel must not take the page down
        st.info(f"Bus not readable: {exc}")
        return

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Stream length", f"{health.stream_length:,}")
    # "unknown", never 0: nil lag means Redis cannot compute the backlog, which
    # is a different thing from being caught up.
    b2.metric("Lag (undelivered)",
              f"{health.lag:,}" if health.lag_known else "unknown")
    b3.metric("Pending (in flight)", f"{health.pending:,}")
    b4.metric("Consumers", f"{health.consumers:,}")

    consumers = consumer_health(client, stream_name=settings.frames_stream,
                                group="workers")
    if consumers:
        st.dataframe([{"consumer": c.name, "pending": c.pending,
                       "idle_ms": c.idle_ms} for c in consumers])

    # Above a healthy frame's processing time, so anything here is a worker that
    # died holding work rather than one that is merely busy.
    stuck = stuck_entries(client, stream_name=settings.frames_stream,
                          group="workers", min_idle_ms=30_000)
    if stuck:
        st.warning(f"{len(stuck)} entry(s) idle over 30 s — a worker may have "
                   f"died holding work. XAUTOCLAIM should reclaim these.")
        st.dataframe([{"entry": e.entry_id, "consumer": e.consumer,
                       "idle_ms": e.idle_ms, "deliveries": e.delivery_count}
                      for e in stuck])


def main() -> None:
    import pandas as pd
    import streamlit as st

    from edgecv.bench.collect import run_coverage
    from edgecv.config import Settings

    st.set_page_config(page_title="Road condition pipeline", layout="wide")
    st.title("Road condition pipeline — walking skeleton")
    st.caption("Skeleton coverage panel. Replaced in Week 9 by the map, worst-N "
               "segments and review queue behind the read API.")

    settings = Settings.from_env()
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        runs = run_options(conn)
        if not runs:
            st.info("No runs yet. Start feed-sim.")
            return

        labels = {f"{r['authority_id']} · {r['started_at']:%H:%M:%S}": r["run_id"]
                  for r in runs}
        run_id = labels[st.selectbox("Survey run", list(labels))]
        _bus_panel(st, settings)
        rows = coverage_series(conn, run_id)
        if not rows:
            st.warning("No frames for this run yet.")
            return

        frame = pd.DataFrame(rows).set_index("bucket")
        offered = int(frame["frames_ingested"].sum())
        processed = int(frame["frames_processed"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Frames ingested", f"{offered:,}")
        c2.metric("Processed", f"{processed:,}",
                  f"{(processed / offered if offered else 0):.1%} coverage")
        c3.metric("Flagged", f"{int(frame['frames_flagged'].sum()):,}")

        # Success criterion 2: a dropped frame is a gap in the survey, reported
        # in metres of road not assessed rather than as a frame count.
        cov = run_coverage(conn, run_id)
        st.subheader("Road assessed")
        m1, m2, m3 = st.columns(3)
        m1.metric("Assessed", f"{cov.assessed_m / 1000:.3f} km",
                  f"{cov.coverage_pct:.1f}% of attempted")
        m2.metric("Not assessed", f"{cov.gap_m:.1f} m",
                  f"{cov.missing_frames} frame(s) in {cov.gap_count} gap(s)",
                  delta_color="inverse")
        m3.metric("Largest gap", f"{cov.largest_gap_m:.1f} m")
        if cov.gaps:
            st.dataframe([{"after_seq": g.after_seq, "before_seq": g.before_seq,
                           "frames_lost": g.missing_frames,
                           "metres": round(g.metres, 1)} for g in cov.gaps])

        st.subheader("Coverage over time")
        st.line_chart(frame[["frames_ingested", "frames_processed"]])
        st.dataframe(frame)


if __name__ == "__main__":
    main()
