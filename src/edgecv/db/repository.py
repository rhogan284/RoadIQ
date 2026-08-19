"""Idempotent batched writer from worker InferenceResults into Postgres.

Insert order follows the schema's own FK dependency graph: detectors before
inferences, inferences before detections. Every insert lands on the
idempotency key the schema already enforces (frames on
(run_id, seq, captured_at), inferences on (run_id, seq, detector_id)), so
replaying a batch -- e.g. after a worker crash and stream redelivery -- is a
no-op rather than a duplicate.
"""
from __future__ import annotations

import json
from datetime import datetime

import psycopg

from edgecv.contracts.detection import DetectorInfo, InferenceResult


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

    def finish_run(self, run_id: str, ended_at: datetime) -> None:
        with self.conn.cursor() as cur:
            cur.execute("UPDATE survey_runs SET ended_at=%s WHERE run_id=%s",
                        (ended_at, run_id))

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

    def write_results(self, results: list[InferenceResult]) -> dict[str, int]:
        """Persist a batch of worker results. Idempotent: replaying the same
        batch inserts nothing twice, because every insert is keyed on the
        idempotency key the schema enforces."""
        frames = inferences = detections = 0
        with self.conn.cursor() as cur:
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
                    # (run_id, seq, detector_id) already recorded -- a
                    # redelivered batch. Detections were written the first
                    # time; do not duplicate them.
                    continue
                inferences += 1
                inference_id = row[0]

                # snippet_id is left NULL: InferenceResult carries only the
                # snippet's content hash (snippet_sha256s), not the
                # width/height/format/bytes the `snippets` table requires.
                # Resolving those needs a blob-metadata lookup that is not
                # part of this contract -- out of scope for this pass, see
                # the dispatch report.
                for d in result.detections:
                    cur.execute(
                        """
                        INSERT INTO detections (inference_id, defect_class, confidence,
                                                bbox_x, bbox_y, bbox_w, bbox_h,
                                                area_px, severity, snippet_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)
                        """,
                        (inference_id, d.defect_class, d.confidence,
                         d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h,
                         d.bbox.area, d.severity),
                    )
                    detections += cur.rowcount

        return {"frames": frames, "inferences": inferences, "detections": detections}
