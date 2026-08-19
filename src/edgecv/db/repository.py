"""Idempotent batched writer from worker InferenceResults into Postgres.

Insert order follows the schema's own FK dependency graph: detectors before
inferences, inferences before detections, snippets before the detections
that reference them. Every insert lands on the idempotency key the schema
already enforces (frames on (run_id, seq, captured_at), inferences on
(run_id, seq, detector_id)).

Every connection this class is used with is autocommit=True, so without an
explicit transaction each cur.execute() would commit the instant it runs.
write_results wraps its whole loop in self.conn.transaction() specifically
so a crash between an inference's commit and its detections' commits can
never happen: either the entire batch lands, or none of it does. That is
what makes "replaying a batch is a no-op rather than a duplicate" true --
without it, a crash mid-batch commits the inference alone, and the
replay's ON CONFLICT DO NOTHING sees that inference as already-recorded and
silently skips the detections that were never written, losing them
permanently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import psycopg

from edgecv.contracts.detection import DetectorInfo, InferenceResult


@dataclass(frozen=True, slots=True)
class WriteStats:
    frames: int
    inferences: int
    detections: int
    skipped_duplicates: int


class Repository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def upsert_run(self, *, run_id: str, authority_id: str, started_at: datetime,
                   source_kind: str, source_ref: str, target_fps: float,
                   prevalence: float | None, transport: str,
                   vehicle_ref: str | None = None,
                   config: dict | None = None,
                   device: dict | None = None) -> None:
        """`authority_id` replaces the old production-line identifier: a run
        is one drive for one road authority, not a shift on a production
        line."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO survey_runs (run_id, authority_id, vehicle_ref, started_at,
                                         source_kind, source_ref, target_fps,
                                         prevalence, transport, config, device)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (run_id, authority_id, vehicle_ref, started_at, source_kind,
                 source_ref, target_fps, prevalence, transport,
                 json.dumps(config or {}), json.dumps(device or {})),
            )

    def finish_run(self, run_id: str, ended_at: datetime, *,
                   config: dict | None = None) -> None:
        """`config` (I1) records feed-level counts the caller already has --
        e.g. feedsim's `{"frames_offered": n, "frames_dropped": n}` -- into the
        existing `survey_runs.config jsonb`, so no migration is needed."""
        with self.conn.cursor() as cur:
            if config is None:
                cur.execute("UPDATE survey_runs SET ended_at=%s WHERE run_id=%s",
                            (ended_at, run_id))
            else:
                cur.execute(
                    "UPDATE survey_runs SET ended_at=%s, config=%s WHERE run_id=%s",
                    (ended_at, json.dumps(config), run_id),
                )

    def _upsert_detector(self, cur: psycopg.Cursor, detector: DetectorInfo) -> int:
        """Detector identity is (name, version, params_hash): a retuned
        threshold is a DIFFERENT detector row, which is what makes the
        benchmark grid in detectors.params_hash meaningful."""
        cur.execute(
            """
            INSERT INTO detectors (name, version, params, params_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name, version, params_hash) DO NOTHING
            """,
            (detector.name, detector.version, json.dumps(detector.params),
             detector.params_hash),
        )
        cur.execute(
            "SELECT detector_id FROM detectors WHERE name=%s AND version=%s "
            "AND params_hash=%s",
            (detector.name, detector.version, detector.params_hash),
        )
        return cur.fetchone()[0]

    def _snippet_id(self, cur: psycopg.Cursor, sha256: str, kind: str, *,
                     fmt: str = "png", bytes_len: int = 0, width: int = 0,
                     height: int = 0) -> int:
        """Snippets are content-addressed by (sha256, kind); the writer only
        has the hash and dimensions from the contract (bbox for a crop, the
        frame size for a thumbnail), not the actual byte size or encoding, so
        `format`/`bytes` take safe defaults rather than blocking the write.

        `bytes`, `width` and `height` are all 0 placeholders (I2): the crop's
        real dimensions are the padded blob `BlobStore.put()` actually wrote
        (`CROP_PADDING_PX` per side larger than the bbox), and the thumbnail's
        real dimensions are its resized 256px-long-edge, not the full frame --
        `InferenceResult` carries only content hashes, so none of that survives
        to the writer. The real values exist transiently in the worker's
        `BlobRef` from `BlobStore.put()` and are discarded. A later milestone
        reporting bytes-stored-per-km must either widen contract 2 to carry
        them, or have the worker write these rows itself instead of the
        writer inferring them from wire fields that were never meant to
        describe the blob."""
        cur.execute(
            """
            INSERT INTO snippets (sha256, kind, format, bytes, width, height)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (sha256, kind) DO NOTHING
            """,
            (sha256, kind, fmt, bytes_len, width, height),
        )
        cur.execute("SELECT snippet_id FROM snippets WHERE sha256=%s AND kind=%s",
                    (sha256, kind))
        return cur.fetchone()[0]

    def write_results(self, results: list[InferenceResult]) -> WriteStats:
        """Persist a batch of worker results. Idempotent: replaying the same
        batch inserts nothing twice, because every insert is keyed on the
        idempotency key the schema enforces. `skipped_duplicates` counts the
        (run_id, seq, detector_id) pairs already recorded by an earlier call.

        The whole loop runs inside one self.conn.transaction(): on
        autocommit=True connections that is what stops a crash between an
        inference's commit and its detections' commits from permanently
        losing those detections (see module docstring)."""
        frames = inferences = detections = duplicates = 0
        with self.conn.transaction(), self.conn.cursor() as cur:
            for result in results:
                # Frame row: the coverage denominator. Every frame gets one,
                # clean ones included. Position is null for generated fixtures.
                cur.execute(
                    """
                    INSERT INTO frames (run_id, seq, captured_at, enqueued_at,
                                        width, height, source_ref, sha256,
                                        lat, lon, heading_deg, speed_mps,
                                        gps_accuracy_m, capture_mono_ns,
                                        device_boot_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, seq, captured_at) DO NOTHING
                    """,
                    (result.run_id, result.seq, result.captured_at, result.started_at,
                     result.width, result.height, result.source_ref,
                     result.frame_sha256,
                     result.lat, result.lon, result.heading_deg, result.speed_mps,
                     result.gps_accuracy_m, result.capture_mono_ns,
                     result.device_boot_id),
                )
                frames += cur.rowcount

                detector_id = self._upsert_detector(cur, result.detector)

                # RETURNING tells us whether WE created the inference row --
                # NULL back means (run_id, seq, detector_id) already existed.
                cur.execute(
                    """
                    INSERT INTO inferences (run_id, seq, captured_at, detector_id,
                                            worker_id, started_at, latency_ms,
                                            status, error)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, seq, detector_id) DO NOTHING
                    RETURNING inference_id
                    """,
                    (result.run_id, result.seq, result.captured_at, detector_id,
                     result.worker_id, result.started_at, result.latency_ms,
                     result.status, result.error),
                )
                row = cur.fetchone()
                if row is None:
                    # Already present -- its detections are already stored.
                    duplicates += 1
                    continue
                inference_id = row[0]
                inferences += 1

                if result.thumbnail_sha256:
                    self._snippet_id(cur, result.thumbnail_sha256, "thumbnail",
                                      width=0, height=0)

                for detection, snippet_sha in zip(result.detections,
                                                   result.snippet_sha256s,
                                                   strict=True):
                    snippet_id = self._snippet_id(
                        cur, snippet_sha, "crop",
                        width=0, height=0,
                    )
                    cur.execute(
                        """
                        INSERT INTO detections (inference_id, defect_class, confidence,
                                                bbox_x, bbox_y, bbox_w, bbox_h,
                                                area_px, severity, snippet_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (inference_id, detection.defect_class, detection.confidence,
                         detection.bbox.x, detection.bbox.y, detection.bbox.w,
                         detection.bbox.h, detection.bbox.area, detection.severity,
                         snippet_id),
                    )
                    detections += 1

        return WriteStats(frames=frames, inferences=inferences,
                           detections=detections, skipped_duplicates=duplicates)
