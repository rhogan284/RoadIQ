"""frames is partitioned in practice: rows land in monthly partitions, DEFAULT stays empty.

The W8 milestone, and the failure from the T2 reading on 9 Sep: with only a DEFAULT
partition, every row sits in the way of the partition that should hold it, and creating
that partition later fails. These tests run against live Postgres (`make up`) and, like
the rest of the suite, TRUNCATE it.
"""
from __future__ import annotations

import datetime as dt

RUN_ID = "00000000-0000-4000-8000-0000000000a8"


def _insert_frame(conn, seq: int, captured_at: dt.datetime) -> None:
    conn.execute(
        "INSERT INTO frames (run_id, seq, captured_at, enqueued_at, width, height, "
        "source_ref, sha256) VALUES (%s, %s, %s, %s, 640, 640, 'test', %s)",
        (RUN_ID, seq, captured_at, captured_at, "0" * 64))


def _where(conn, seq: int) -> str:
    return conn.execute(
        "SELECT tableoid::regclass::text FROM frames WHERE run_id = %s AND seq = %s",
        (RUN_ID, seq)).fetchone()[0]


def _default_rows(conn) -> int:
    return conn.execute("SELECT count(*) FROM frames_default").fetchone()[0]


def test_partitions_exist_ahead_of_the_data(clean_db):
    from edgecv.db.migrate import PARTITION_MONTHS_AHEAD
    month = dt.date.today().replace(day=1)
    names = {r[0] for r in clean_db.execute(
        "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE i.inhparent = 'frames'::regclass").fetchall()}
    for _ in range(PARTITION_MONTHS_AHEAD + 1):
        assert f"frames_{month:%Y_%m}" in names
        month = (month + dt.timedelta(days=32)).replace(day=1)


def test_a_new_frame_lands_in_its_month_not_default(clean_db):
    now = dt.datetime.now(dt.timezone.utc)
    _insert_frame(clean_db, 1, now)
    assert _where(clean_db, 1) == f"frames_{now:%Y_%m}"
    assert _default_rows(clean_db) == 0


def test_rows_stranded_in_default_are_moved_to_their_partition(clean_db):
    # A month with no partition is exactly the 001 situation: the row falls to DEFAULT.
    old = dt.datetime(2020, 1, 15, tzinfo=dt.timezone.utc)
    _insert_frame(clean_db, 2, old)
    assert _where(clean_db, 2) == "frames_default"
    try:
        moved = clean_db.execute("SELECT ensure_frame_partitions()").fetchone()[0]
        assert moved == 1
        assert _where(clean_db, 2) == "frames_2020_01"
        assert _default_rows(clean_db) == 0
        # Idempotent: nothing left to rescue, nothing new to create.
        assert clean_db.execute("SELECT ensure_frame_partitions()").fetchone()[0] == 0
    finally:
        clean_db.execute("DROP TABLE IF EXISTS frames_2020_01")


def test_a_time_bounded_read_prunes_to_one_partition(clean_db):
    now = dt.datetime.now(dt.timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = (start + dt.timedelta(days=32)).replace(day=1)
    plan = "\n".join(r[0] for r in clean_db.execute(
        "EXPLAIN (COSTS OFF) SELECT count(*) FROM frames "
        "WHERE run_id = %s AND captured_at >= %s AND captured_at < %s",
        (RUN_ID, start, end)).fetchall())
    assert f"frames_{now:%Y_%m}" in plan
    assert "Append" not in plan, plan
