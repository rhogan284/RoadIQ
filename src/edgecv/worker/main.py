"""Inference worker: read frames, detect, store crops, emit results.

Stateless and CPU-bound by design — scale it with replicas. It never touches Postgres;
metadata goes onto the results stream for the single writer to persist.
"""
from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import redis

from edgecv.blobstore.store import BlobStore
from edgecv.bus.consumer import FrameConsumer
from edgecv.config import Settings
from edgecv.contracts.detection import Detector, InferenceResult
from edgecv.contracts.frame import FrameEnvelope
from edgecv.detectors.registry import build_detector
from edgecv.detectors.threshold import ThresholdDetector

CROP_PADDING_PX = 16
THUMBNAIL_LONG_EDGE = 256
CROP_FMT = "webp"
THUMB_FMT = "webp"
RECLAIM_IDLE_MS = 30_000
READ_COUNT = 10
RECLAIM_COUNT = 10
# Frames destroyed after a worker accepted them. Deliberately not keyed by
# run_id the way the producer's `stats:{run_id}:dropped` is: the entry's
# payload is gone, so there is no run_id or seq left to key it by.
LOST_IN_FLIGHT_KEY = "stats:lost_in_flight"


def _decode(envelope: FrameEnvelope) -> np.ndarray:
    # Colour, not grayscale: the YOLO detector was trained on colour frames, and the
    # threshold detector converts to gray itself. Crops and thumbnails are therefore
    # colour too, so bytes-per-km is higher than the M2 (grayscale) figure.
    if envelope.transport == "inline":
        assert envelope.payload is not None
        buffer = np.frombuffer(envelope.payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    else:
        assert envelope.path is not None
        image = cv2.imread(envelope.path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode frame {envelope.source_ref}")
    return image


def _encode(image: np.ndarray, fmt: str, *, lossless: bool) -> bytes:
    params = [cv2.IMWRITE_WEBP_QUALITY, 101 if lossless else 80]
    ok, buffer = cv2.imencode(f".{fmt}", image, params)
    if not ok:
        raise ValueError(f"could not encode {fmt}")
    return buffer.tobytes()


def process_one(envelope: FrameEnvelope, *, blobstore: BlobStore, worker_id: str,
                detector: Detector | None = None) -> InferenceResult:
    detector = detector or ThresholdDetector()
    started_at = datetime.now(timezone.utc)
    began = time.perf_counter()

    def result(status: str, error: str | None, detections, snippets, thumbnail):
        return InferenceResult(
            run_id=envelope.run_id, seq=envelope.seq,
            captured_at=envelope.captured_at, width=envelope.width,
            height=envelope.height, source_ref=envelope.source_ref,
            frame_sha256=envelope.sha256,
            detector=detector.info,
            worker_id=worker_id, started_at=started_at,
            latency_ms=(time.perf_counter() - began) * 1000,
            status=status, error=error, detections=detections,
            snippet_sha256s=snippets, thumbnail_sha256=thumbnail,
            # Contract 2: position and clock are copied straight off the frame
            # envelope. The worker is the only component that sees both the
            # envelope and the result, so if it drops them here they are gone --
            # the writer has nothing else to insert into `frames`. Copied for
            # every status, failed frames included: a failed frame still gets a
            # frames row and is still part of the coverage denominator, so it
            # still has to carry the metres it covered.
            lat=envelope.lat, lon=envelope.lon,
            heading_deg=envelope.heading_deg, speed_mps=envelope.speed_mps,
            gps_accuracy_m=envelope.gps_accuracy_m,
            capture_mono_ns=envelope.capture_mono_ns,
            device_boot_id=envelope.device_boot_id,
        )

    try:
        image = _decode(envelope)
    except ValueError as exc:
        return result("failed", str(exc), [], [], None)

    detections = detector.detect(image)
    if not detections:
        # Clean frames store no pixels at all. This is the storage policy's core.
        return result("ok", None, [], [], None)

    height, width = image.shape[:2]
    snippet_shas: list[str] = []
    for detection in detections:
        bbox = detection.bbox
        x0 = max(0, bbox.x - CROP_PADDING_PX)
        y0 = max(0, bbox.y - CROP_PADDING_PX)
        x1 = min(width, bbox.x + bbox.w + CROP_PADDING_PX)
        y1 = min(height, bbox.y + bbox.h + CROP_PADDING_PX)
        crop = image[y0:y1, x0:x1]
        # Crops are lossless: they may be evidence for audit or retraining.
        ref = blobstore.put(_encode(crop, CROP_FMT, lossless=True), kind="crop",
                            fmt=CROP_FMT, width=x1 - x0, height=y1 - y0)
        snippet_shas.append(ref.sha256)

    scale = THUMBNAIL_LONG_EDGE / max(height, width)
    thumb = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image
    thumb_ref = blobstore.put(_encode(thumb, THUMB_FMT, lossless=False),
                              kind="thumbnail", fmt=THUMB_FMT,
                              width=thumb.shape[1], height=thumb.shape[0])

    return result("ok", None, detections, snippet_shas, thumb_ref.sha256)


def _handle(entry_id, envelope, *, consumer: FrameConsumer, blobstore: BlobStore,
            worker_id: str, client: redis.Redis, results_stream: str,
            detector: Detector | None = None) -> None:
    outcome = process_one(envelope, blobstore=blobstore, worker_id=worker_id,
                          detector=detector)
    client.xadd(results_stream, {"json": outcome.to_json()})
    consumer.ack(entry_id)


def _record_lost(client: redis.Redis, entry_ids: list[str], *,
                 worker_id: str) -> None:
    """Count frames XAUTOCLAIM found destroyed under the pending-entries list.

    A refused frame is a *known* gap: the producer counted it and the run knows
    which frame it was. This is the unknown kind — a worker had accepted the
    frame and the entry was destroyed beneath it, taking run_id, seq and
    position with it. The count is the alarm. The metres come from the coverage
    report, which sees the same loss as a missing `frames` row and measures the
    hole between its surviving neighbours.
    """
    client.incrby(LOST_IN_FLIGHT_KEY, len(entry_ids))
    print(f"{worker_id} LOST {len(entry_ids)} in-flight frame(s) — stream entries "
          f"destroyed while pending: {', '.join(entry_ids)}", flush=True)


def _drain_reclaimed(consumer: FrameConsumer, *, blobstore: BlobStore,
                      worker_id: str, client: redis.Redis,
                      results_stream: str, detector: Detector | None = None) -> int:
    """Fully claim every currently-idle-enough pending entry, not just one
    XAUTOCLAIM's worth.

    reclaim() makes a single XAUTOCLAIM call and does not loop its cursor —
    it drains at most RECLAIM_COUNT idle entries per call, by design (a
    worker's main loop must not block on an unbounded drain). That means the
    caller decides how hard to drain. Calling it once per outer-loop pass,
    only when a normal read comes back empty, can strand frames a dead
    worker left behind for many seconds if there are more idle entries than
    one call covers — every extra batch waits for another empty read, and
    FrameConsumer.read() blocks for block_ms (2s default) each time there is
    nothing new. Looping here until reclaim() stops making progress drains the
    entire backlog in one pass instead, with no added delay.

    The loop condition is `progressed`, not "returned deliveries". An
    XAUTOCLAIM page that only purges destroyed entries recovers no work but has
    still advanced through the PEL, and stopping on it strands the live orphans
    sitting behind it — which is the bug this looked like before the purged-id
    list was read at all.
    """
    processed = 0
    outcome = consumer.reclaim(min_idle_ms=RECLAIM_IDLE_MS, count=RECLAIM_COUNT)
    while outcome.progressed:
        for entry_id, envelope in outcome.deliveries:
            _handle(entry_id, envelope, consumer=consumer, blobstore=blobstore,
                    worker_id=worker_id, client=client, results_stream=results_stream,
                    detector=detector)
            processed += 1
        if outcome.lost_entry_ids:
            _record_lost(client, outcome.lost_entry_ids, worker_id=worker_id)
        outcome = consumer.reclaim(min_idle_ms=RECLAIM_IDLE_MS, count=RECLAIM_COUNT)
    return processed


def main() -> None:
    settings = Settings.from_env()
    worker_id = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")
    client = redis.from_url(settings.redis_url, decode_responses=False)
    consumer = FrameConsumer(client, stream=settings.frames_stream,
                             group="workers", consumer=worker_id)
    consumer.ensure_group()
    blobstore = BlobStore(root=settings.blob_root)
    # Built once: loading YOLO weights per frame would dominate latency.
    detector = build_detector()

    running = True

    def stop(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"{worker_id} started detector={detector.info.name}/{detector.info.version}",
          flush=True)
    while running:
        batch = consumer.read(count=READ_COUNT)
        if not batch:
            # Nothing new — take over anything a dead worker left in flight,
            # draining the whole backlog rather than one XAUTOCLAIM's worth.
            _drain_reclaimed(consumer, blobstore=blobstore, worker_id=worker_id,
                             client=client, results_stream=settings.results_stream,
                             detector=detector)
            continue
        for entry_id, envelope in batch:
            _handle(entry_id, envelope, consumer=consumer, blobstore=blobstore,
                    worker_id=worker_id, client=client,
                    results_stream=settings.results_stream, detector=detector)


if __name__ == "__main__":
    main()
