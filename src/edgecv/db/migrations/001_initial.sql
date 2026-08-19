-- R-R1: geography on segments and defect_instances. The image pin in
-- docker-compose.yml is the other half of this requirement.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS detectors (
    detector_id  bigserial PRIMARY KEY,
    name         text NOT NULL,
    version      text NOT NULL,
    params       jsonb NOT NULL DEFAULT '{}',
    params_hash  text NOT NULL,
    UNIQUE (name, version, params_hash)
);

-- Moved above survey_runs (A3's ordering trap): survey_runs.bench_run_id
-- references this table, so it must exist first or the migration fails on an
-- unknown relation.
CREATE TABLE IF NOT EXISTS bench_runs (
    bench_run_id     uuid PRIMARY KEY,
    label            text,
    grid_point       jsonb NOT NULL DEFAULT '{}',
    frames_offered   integer,
    frames_processed integer,
    frames_dropped   integer,
    p50_ms           numeric,
    p95_ms           numeric,
    p99_ms           numeric,
    precision        numeric,
    recall           numeric,
    f1               numeric,
    bytes_stored     bigint,
    peak_rss_mb      numeric,
    cpu_pct          numeric,
    started_at       timestamptz,
    ended_at         timestamptz
);

-- Renamed: a "run" is now one drive, not a factory shift.
CREATE TABLE IF NOT EXISTS survey_runs (
    run_id       uuid PRIMARY KEY,
    authority_id text NOT NULL,          -- council/authority owning this drive
    vehicle_ref  text,
    started_at   timestamptz NOT NULL,
    ended_at     timestamptz,
    source_kind  text NOT NULL
                 CHECK (source_kind IN ('dataset-replay', 'drive', 'synthetic')),
    source_ref   text NOT NULL,
    target_fps   numeric NOT NULL,
    prevalence   numeric,
    transport    text NOT NULL CHECK (transport IN ('inline', 'reference')),
    config       jsonb NOT NULL DEFAULT '{}',
    device       jsonb NOT NULL DEFAULT '{}',   -- handset, OS, camera params
    bench_run_id uuid REFERENCES bench_runs (bench_run_id)
);

-- Range-partitioned on captured_at so retention is a partition DROP, not a mass
-- DELETE. A partitioned table's PK must contain the partition key (R-R5).
--
-- Position is PLAIN float here, not PostGIS: on this hot, high-volume table the
-- coordinates are only telemetry, and keeping the extension off it is deliberate
-- (R-R1 hybrid storage). Geometry lives on segments and defect_instances.
--
-- Only a DEFAULT partition for now. Monthly partitions and the retention reaper
-- are the W9 compaction task; DEFAULT routes every row correctly until then.
CREATE TABLE IF NOT EXISTS frames (
    run_id          uuid NOT NULL,
    seq             bigint NOT NULL,
    captured_at     timestamptz NOT NULL,
    enqueued_at     timestamptz NOT NULL,
    width           integer NOT NULL,
    height          integer NOT NULL,
    source_ref      text NOT NULL,
    sha256          char(64) NOT NULL,
    lat             double precision,
    lon             double precision,
    heading_deg     real,
    speed_mps       real,
    gps_accuracy_m  real,
    capture_mono_ns bigint,
    device_boot_id  uuid,
    PRIMARY KEY (run_id, seq, captured_at)
) PARTITION BY RANGE (captured_at);

CREATE TABLE IF NOT EXISTS frames_default PARTITION OF frames DEFAULT;

-- R-R5: "all frames for run X" prunes nothing unless the query also constrains
-- time. The read path adds that from survey_runs; this index covers the rest.
CREATE INDEX IF NOT EXISTS frames_run_seq_idx ON frames (run_id, seq);

CREATE TABLE IF NOT EXISTS snippets (
    snippet_id  bigserial PRIMARY KEY,
    sha256      char(64) NOT NULL,
    kind        text NOT NULL CHECK (kind IN ('crop', 'thumbnail')),
    format      text NOT NULL,
    bytes       integer NOT NULL,
    width       integer NOT NULL,
    height      integer NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (sha256, kind)
);

CREATE TABLE IF NOT EXISTS inferences (
    inference_id  bigserial PRIMARY KEY,
    run_id        uuid NOT NULL,
    seq           bigint NOT NULL,
    captured_at   timestamptz NOT NULL,
    detector_id   bigint NOT NULL REFERENCES detectors (detector_id),
    worker_id     text NOT NULL,
    started_at    timestamptz NOT NULL,
    latency_ms    numeric NOT NULL,
    status        text NOT NULL CHECK (status IN ('ok', 'failed', 'skipped')),
    error         text,
    UNIQUE (run_id, seq, detector_id)          -- the idempotency key
);

CREATE INDEX IF NOT EXISTS inferences_captured_at_idx ON inferences (captured_at);

CREATE TABLE IF NOT EXISTS detections (
    detection_id  bigserial PRIMARY KEY,
    inference_id  bigint NOT NULL REFERENCES inferences (inference_id) ON DELETE CASCADE,
    defect_class  text NOT NULL,
    confidence    numeric,
    bbox_x        integer NOT NULL,
    bbox_y        integer NOT NULL,
    bbox_w        integer NOT NULL,
    bbox_h        integer NOT NULL,
    area_px       integer NOT NULL,
    severity      text NOT NULL CHECK (severity IN ('minor', 'major', 'critical')),
    snippet_id    bigint REFERENCES snippets (snippet_id)
);

CREATE INDEX IF NOT EXISTS detections_inference_idx ON detections (inference_id);

CREATE TABLE IF NOT EXISTS ground_truth (
    gt_id         bigserial PRIMARY KEY,
    source_ref    text NOT NULL,
    defect_class  text NOT NULL,
    bbox_x        integer NOT NULL,
    bbox_y        integer NOT NULL,
    bbox_w        integer NOT NULL,
    bbox_h        integer NOT NULL,
    UNIQUE (source_ref, defect_class, bbox_x, bbox_y, bbox_w, bbox_h)
);

-- Spec §5 view 6 — the coverage panel, per minute. Replaces the old defect-rate
-- materialized view (A4): that matview fed an SPC p-chart spec §5 no longer
-- has. This view feeds the panel spec §5 actually specifies: "km surveyed vs
-- km attempted, frames dropped, GPS gaps".
--
-- A plain VIEW, not materialized — no periodic refresh function, no manual
-- refresh fallback, no unique index to support one. Instant at skeleton
-- volumes.
--
-- `frames` is the denominator, which is why every frame gets a row including the
-- clean ones: "we assessed 14.2 km of the 15 km segment" is only provable if
-- every frame is accounted for. Storage discipline applies to PIXELS, not rows.
--
-- Per-frame booleans are computed with EXISTS in the inner SELECT, then counted in
-- the outer aggregate. Do NOT join frames to inferences here: inferences is
-- UNIQUE(run_id, seq, detector_id), so one frame processed by N detectors would
-- fan out to N rows and inflate every count -- silently overstating coverage.
CREATE OR REPLACE VIEW run_coverage_1min AS
SELECT
    date_trunc('minute', f.captured_at)     AS bucket,
    f.run_id                                AS run_id,
    count(*)                                AS frames_ingested,
    count(*) FILTER (WHERE f.processed)     AS frames_processed,
    count(*) FILTER (WHERE f.flagged)       AS frames_flagged,
    count(*) FILTER (WHERE f.lat IS NULL)   AS frames_without_fix
FROM (
    SELECT fr.run_id, fr.seq, fr.captured_at, fr.lat,
           EXISTS (SELECT 1 FROM inferences i
                    WHERE i.run_id = fr.run_id AND i.seq = fr.seq
                      AND i.captured_at = fr.captured_at
                      AND i.status = 'ok')                       AS processed,
           EXISTS (SELECT 1 FROM inferences i
                    JOIN detections d ON d.inference_id = i.inference_id
                    WHERE i.run_id = fr.run_id AND i.seq = fr.seq
                      AND i.captured_at = fr.captured_at)         AS flagged
    FROM frames fr
) f
GROUP BY 1, 2;

-- Geospatial tables (segments, defect_instances, instance_reviews,
-- segment_condition) are appended below by the next migration task, after
-- `detections` (their best_detection_id FK target already exists above).

-- ===========================================================================
-- Geospatial and defect-instance model. R-R1 (PostGIS geography) + R-R2
-- (recompute-and-replace). Empty in the skeleton: Ilana's segmenter fills
-- defect_instances and segment_condition, Joseph's read API writes reviews.
-- Created NOW so neither arrival needs a migration.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS segments (
    segment_id   bigserial PRIMARY KEY,
    authority_id text NOT NULL,
    road_name    text NOT NULL,
    road_ref     text,
    geom         geography(LineString, 4326) NOT NULL,
    length_m     double precision NOT NULL,
    surface_type text NOT NULL DEFAULT 'sealed'
);

CREATE INDEX IF NOT EXISTS segments_geom_gix ON segments USING GIST (geom);

-- One row per PHYSICAL defect, clustered from the 2-9 detections that saw it.
-- DERIVED DATA: the segmenter recomputes all rows for a (run_id, segment_id) and
-- swaps them in one transaction, which is what makes it idempotent regardless of
-- which clustering algorithm runs inside (R-R2).
--
-- Note the absence of review columns — recompute-and-replace churns instance_id,
-- so human decisions live in instance_reviews instead.
CREATE TABLE IF NOT EXISTS defect_instances (
    instance_id       bigserial PRIMARY KEY,
    cluster_key       char(32) NOT NULL,   -- deterministic identity: hash of
                                           -- (run_id, class, min(seq), ordinal)
    run_id            uuid NOT NULL REFERENCES survey_runs (run_id),
    segment_id        bigint REFERENCES segments (segment_id),
    defect_class      text NOT NULL,
    lat               double precision NOT NULL,
    lon               double precision NOT NULL,
    -- Generated geography, legal because ST_MakePoint, ST_SetSRID and
    -- geography(geometry) are each IMMUTABLE (checked against the PostGIS
    -- source, not a blog). ST_MakePoint takes (lon, lat) — X before Y.
    geog              geography(Point, 4326)
                      GENERATED ALWAYS AS (
                          ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography
                      ) STORED,
    along_m           double precision,    -- ST_LineLocatePoint * length_m
    first_seen_at     timestamptz NOT NULL,
    last_seen_at      timestamptz NOT NULL,
    observation_count integer NOT NULL DEFAULT 1,
    peak_confidence   numeric,
    severity          text NOT NULL,
    extent_m          double precision,
    best_detection_id bigint REFERENCES detections (detection_id) ON DELETE SET NULL,
    UNIQUE (run_id, cluster_key)
);

CREATE INDEX IF NOT EXISTS defect_instances_geog_gix
    ON defect_instances USING GIST (geog);

-- Human input, not derived data. Keyed on the stable identity so it survives a
-- segmenter re-run. No FK to defect_instances: a review may outlive the instance
-- row it was made against, which is the entire point of this table.
CREATE TABLE IF NOT EXISTS instance_reviews (
    run_id       uuid NOT NULL REFERENCES survey_runs (run_id),
    cluster_key  char(32) NOT NULL,
    review_state text NOT NULL CHECK (review_state IN
                     ('pending', 'confirmed', 'rejected', 'reclassified')),
    new_class    text,
    reviewed_by  text,
    reviewed_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, cluster_key)
);

-- One score per segment per survey, so re-surveying ADDS a row rather than
-- overwriting one and change-over-time is a self-join.
CREATE TABLE IF NOT EXISTS segment_condition (
    segment_id      bigint NOT NULL REFERENCES segments (segment_id),
    run_id          uuid NOT NULL REFERENCES survey_runs (run_id),
    assessed_at     timestamptz NOT NULL,
    condition_index numeric,
    condition_band  text,
    counts          jsonb NOT NULL DEFAULT '{}',   -- per-class instance counts
    frames_assessed integer,
    coverage_m      double precision,
    PRIMARY KEY (segment_id, run_id)
);
