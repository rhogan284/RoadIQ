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


def main() -> None:
    import pandas as pd
    import streamlit as st

    from edgecv.config import Settings

    st.set_page_config(page_title="Road condition pipeline", layout="wide")
    st.title("Road condition pipeline — walking skeleton")
    st.caption("Skeleton coverage panel. Replaced in Week 9 by the map, worst-N "
               "segments and review queue behind the read API.")

    with psycopg.connect(Settings.from_env().pg_dsn, autocommit=True) as conn:
        runs = run_options(conn)
        if not runs:
            st.info("No runs yet. Start feed-sim.")
            return

        labels = {f"{r['authority_id']} · {r['started_at']:%H:%M:%S}": r["run_id"]
                  for r in runs}
        rows = coverage_series(conn, labels[st.selectbox("Survey run", list(labels))])
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

        st.subheader("Coverage over time")
        st.line_chart(frame[["frames_ingested", "frames_processed"]])
        st.dataframe(frame)


if __name__ == "__main__":
    main()
