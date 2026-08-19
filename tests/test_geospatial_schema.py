"""Proves the geospatial schema is LIVE — it accepts the shapes Ilana's segmenter
will write — rather than merely present. Every DDL claim in R-R1/R-R2 was
design-time and unverified; this is where it stops being a claim.
"""
import pytest

pytestmark = pytest.mark.integration

RUN = "44444444-4444-4444-4444-444444444444"
# George Street, Sydney. Southern hemisphere and east of Greenwich, which is what
# lets the transposition test below actually fail.
SYD_LAT, SYD_LON = -33.8688, 151.2093


@pytest.fixture
def run_id(clean_db):
    with clean_db.cursor() as cur:
        cur.execute("""
            INSERT INTO survey_runs (run_id, authority_id, started_at, source_kind,
                                     source_ref, target_fps, transport)
            VALUES (%s, 'demo-council', '2026-08-19T10:00:00+10:00', 'synthetic',
                    'generated', 10, 'reference')
            ON CONFLICT (run_id) DO NOTHING
        """, (RUN,))
    return RUN


def test_postgis_is_available_at_the_version_we_pinned(clean_db):
    with clean_db.cursor() as cur:
        cur.execute("SELECT postgis_lib_version()")
        version = cur.fetchone()[0]
    major, minor = (int(p) for p in version.split(".")[:2])
    # ST_LineSubstring(geography) arrived in 3.4.0 and we need it to cut
    # centrelines into 100 m segments.
    assert (major, minor) >= (3, 4), f"PostGIS {version} < 3.4"


def test_a_sydney_defect_does_not_land_in_xinjiang(clean_db, run_id):
    """THE transposition test. ST_MakePoint(x, y) takes (lon, lat) — the reverse
    of how a coordinate is spoken and of our column order. R-R1 names this the
    most likely defect in the project. Swapped arguments put this at 151N, 33E.
    """
    with clean_db.cursor() as cur:
        cur.execute("""
            INSERT INTO defect_instances
                (cluster_key, run_id, defect_class, lat, lon, first_seen_at,
                 last_seen_at, observation_count, severity)
            VALUES (%s, %s, 'pothole', %s, %s, now(), now(), 4, 'major')
            RETURNING ST_Y(geog::geometry), ST_X(geog::geometry)
        """, ("a" * 32, run_id, SYD_LAT, SYD_LON))
        y, x = cur.fetchone()

    assert y == pytest.approx(SYD_LAT), "latitude is not in Y — arguments transposed"
    assert x == pytest.approx(SYD_LON), "longitude is not in X — arguments transposed"


def test_segment_attribution_query_works(clean_db, run_id):
    """The query the segmenter depends on: point -> which segment, and how far
    along it. Proving it runs and returns a plausible distance is the point; the
    clustering that feeds it is Ilana's."""
    with clean_db.cursor() as cur:
        cur.execute("""
            INSERT INTO segments (authority_id, road_name, geom, length_m)
            VALUES ('demo-council', 'George Street',
                    ST_SetSRID(ST_MakeLine(ST_MakePoint(%s, %s),
                                           ST_MakePoint(%s, %s)), 4326)::geography,
                    92.0)
            RETURNING segment_id
        """, (SYD_LON, SYD_LAT, SYD_LON + 0.001, SYD_LAT))
        segment_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO defect_instances
                (cluster_key, run_id, defect_class, lat, lon, first_seen_at,
                 last_seen_at, observation_count, severity)
            VALUES (%s, %s, 'pothole', %s, %s, now(), now(), 3, 'minor')
            RETURNING instance_id
        """, ("b" * 32, run_id, SYD_LAT, SYD_LON + 0.0005))
        instance_id = cur.fetchone()[0]

        cur.execute("""
            SELECT s.segment_id,
                   ST_LineLocatePoint(s.geom::geometry, d.geog::geometry) * s.length_m
            FROM   segments s
            JOIN   defect_instances d ON ST_DWithin(s.geom, d.geog, 25)
            WHERE  d.instance_id = %s
            ORDER  BY ST_Distance(s.geom, d.geog)
            LIMIT  1
        """, (instance_id,))
        row = cur.fetchone()

    assert row is not None, "ST_DWithin found no segment within 25 m"
    assert row[0] == segment_id
    assert 30 < float(row[1]) < 65, "defect should sit near the middle of the segment"


def test_recompute_and_replace_is_idempotent_and_keeps_review_state(clean_db, run_id):
    """R-R2's guarantee as a test, and the reason instance_reviews is separate.
    Re-running the segmenter must leave the cluster_key set identical AND must not
    delete a council officer's decision."""
    def recompute(observation_count):
        with clean_db.transaction(), clean_db.cursor() as cur:
            cur.execute("DELETE FROM defect_instances WHERE run_id = %s", (run_id,))
            for i, key in enumerate(("c" * 32, "d" * 32)):
                cur.execute("""
                    INSERT INTO defect_instances
                        (cluster_key, run_id, defect_class, lat, lon, first_seen_at,
                         last_seen_at, observation_count, severity)
                    VALUES (%s, %s, 'pothole', %s, %s, now(), now(), %s, 'minor')
                """, (key, run_id, SYD_LAT + i * 0.0001, SYD_LON, observation_count))

    recompute(3)
    with clean_db.cursor() as cur:
        cur.execute("""
            INSERT INTO instance_reviews (run_id, cluster_key, review_state, reviewed_by)
            VALUES (%s, %s, 'confirmed', 'officer@council.nsw.gov.au')
        """, (run_id, "c" * 32))
        cur.execute("SELECT cluster_key FROM defect_instances WHERE run_id=%s "
                    "ORDER BY cluster_key", (run_id,))
        first = [r[0] for r in cur.fetchall()]

    recompute(5)          # the segmenter runs again with more observations
    with clean_db.cursor() as cur:
        cur.execute("SELECT cluster_key FROM defect_instances WHERE run_id=%s "
                    "ORDER BY cluster_key", (run_id,))
        second = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT review_state FROM instance_reviews "
                    "WHERE run_id=%s AND cluster_key=%s", (run_id, "c" * 32))
        review = cur.fetchone()

    assert first == second, "recompute changed the cluster identity set"
    assert review == ("confirmed",), "a re-run destroyed human review state"
