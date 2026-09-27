"""Real RDD2022 images through every component, in-process, against live Postgres/Redis.

feed-sim (route track) → bus → worker (YOLOv12s) → writer → segments → segmenter →
bench → read API → review → re-score. The containerised run (`make e2e`) is the same code
at full scale; this is the 40-image version that can run in a minute.

Needs `make weights dataset` and Postgres/Redis up. Like the rest of the integration
suite it TRUNCATES the database it points at — point PG_DSN at a scratch stack.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import pytest

from edgecv.blobstore.store import BlobStore
from edgecv.bus.consumer import FrameConsumer
from edgecv.db.repository import Repository
from edgecv.feedsim.main import run_feed
from edgecv.feedsim.route import RouteTrack
from edgecv.roads import DEFAULT_NETWORK
from edgecv.worker.main import process_one
from edgecv.writer.main import drain_once

pytestmark = pytest.mark.integration

MANIFEST = Path("data/rdd2022/manifest.json")
GT = Path("data/rdd2022/ground_truth.json")
WEIGHTS = Path("weights/yolo12s_RDD2022_best.pt")
FPS = 8.0

if not (MANIFEST.exists() and WEIGHTS.exists()):
    pytest.skip("needs `make weights dataset`", allow_module_level=True)


@pytest.fixture(scope="module")
def detector():
    from edgecv.detectors.yolo12s import Yolo12sDetector
    return Yolo12sDetector(weights=WEIGHTS)


@pytest.fixture
def mini_manifest(tmp_path):
    full = json.loads(MANIFEST.read_text())
    rng = random.Random(7)
    sub = {"clean": rng.sample(full["clean"], 20), "defect": rng.sample(full["defect"], 20)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(sub))
    return path


def _scalar(conn, sql, *args):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()[0]


def test_rdd2022_through_every_component(rds, clean_db, mini_manifest, detector, tmp_path,
                                         monkeypatch):
    repo = Repository(clean_db)
    blobstore = BlobStore(root=tmp_path / "blobs")
    run_id = "00000000-0000-4000-8000-00000000e2e0"
    repo.upsert_run(run_id=run_id, authority_id="demo-council",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    source_kind="dataset-replay", source_ref=str(mini_manifest),
                    target_fps=FPS, prevalence=None, transport="reference")

    # 1. feed-sim: every image once, along the real street network
    track = RouteTrack.from_network(DEFAULT_NETWORK, n_frames=40, seed=1,
                                    speed_mps=13.89, fps=FPS)
    stats = run_feed(rds, manifest=mini_manifest, run_id=run_id, n_frames=0, fps=1000,
                     prevalence=0.0, maxlen=10_000, seed=1, stream="frames",
                     track=track, order="all")
    assert (stats.offered, stats.dropped) == (40, 0)
    repo.finish_run(run_id, dt.datetime.now(dt.timezone.utc),
                    config={"frames_offered": 40, "frames_dropped": 0,
                            "raw_bytes_offered": stats.raw_bytes})

    # 2. worker with Shervin's detector
    consumer = FrameConsumer(rds, stream="frames", group="workers", consumer="w1",
                             block_ms=100)
    consumer.ensure_group()
    while batch := consumer.read(count=20):
        for entry_id, envelope in batch:
            out = process_one(envelope, blobstore=blobstore, worker_id="w1", detector=detector)
            rds.xadd("results", {"json": out.to_json()})
            consumer.ack(entry_id)

    # 3. writer
    while drain_once(rds, repo, stream="results", group="writers", consumer="wr1",
                     batch=50).frames:
        pass
    assert _scalar(clean_db, "SELECT count(*) FROM frames") == 40
    assert _scalar(clean_db, "SELECT count(*) FROM inferences WHERE status = 'ok'") == 40
    n_det = _scalar(clean_db, "SELECT count(*) FROM detections")
    assert n_det > 0, "YOLOv12s found nothing on 20 labelled damage images"
    assert _scalar(clean_db, "SELECT count(DISTINCT name) FROM detectors") == 1

    # 4. segmenter: every located detection becomes exactly one instance (replay = 0 m merge)
    from edgecv.segmenter.main import load_segments, segment_run
    assert load_segments(clean_db, authority_id="demo-council") > 0
    first = segment_run(clean_db, run_id)
    assert first["instances"] + first["unlocated_detections"] == n_det
    assert first["segments_scored"] > 0
    keys = lambda: {r[0] for r in clean_db.execute(  # noqa: E731
        "SELECT cluster_key FROM defect_instances").fetchall()}
    before = keys()
    assert segment_run(clean_db, run_id)["instances"] == first["instances"]
    assert keys() == before, "re-running the segmenter must reproduce the same identities"

    # 5. bench against ground truth
    from edgecv.bench.evaluate import evaluate, load_ground_truth, record
    load_ground_truth(clean_db, GT)
    report = evaluate(clean_db, run_id)
    assert report["frames_processed"] == 40
    assert sum(m["gt"] for m in report["per_class"].values()) > 0
    record(clean_db, report)

    # 6. read API + review, exactly as the dashboard calls it
    monkeypatch.setenv("BLOB_ROOT", str(tmp_path / "blobs"))
    from fastapi.testclient import TestClient

    import edgecv.api.main as api
    monkeypatch.setattr(api, "SETTINGS", api.Settings.from_env())
    client = TestClient(api.app)
    summary = client.get(f"/api/runs/{run_id}/summary").json()
    assert summary["frames_accounted_pct"] == 100.0
    assert summary["assessed_km"] > 0
    geo = client.get(f"/api/runs/{run_id}/segments.geojson").json()
    assert any(f["properties"]["condition_band"] for f in geo["features"])
    assert client.get(f"/api/runs/{run_id}/worklist").json()
    assert client.get(f"/api/runs/{run_id}/bench").json()["frames_processed"] == 40

    scored = [i for i in client.get(f"/api/runs/{run_id}/instances").json()
              if i["segment_id"] and i["defect_class"] != "other"]
    target = scored[0]
    detail = client.get(f"/api/instances/{run_id}/{target['cluster_key']}").json()
    assert client.get(f"/api/frames/{run_id}/{detail['seq']}/image").status_code == 200
    if detail["crop_sha256"]:
        assert client.get(f"/api/blobs/crop/{detail['crop_sha256']}").status_code == 200

    def band_index():
        f = next(f for f in client.get(f"/api/runs/{run_id}/segments.geojson").json()["features"]
                 if f["properties"]["segment_id"] == target["segment_id"])
        return f["properties"]["condition_index"]

    rejected = client.post(f"/api/instances/{run_id}/{target['cluster_key']}/review",
                           json={"state": "rejected"}).json()
    after_reject = band_index()
    confirmed = client.post(f"/api/instances/{run_id}/{target['cluster_key']}/review",
                            json={"state": "confirmed"}).json()
    assert rejected["state"] == "rejected" and confirmed["state"] == "confirmed"
    # A confirmed defect deducts; a rejected one does not. The review survives the
    # segmenter's recompute-and-replace because it is keyed on cluster_key.
    if after_reject is not None and confirmed["condition_index"] is not None:
        assert confirmed["condition_index"] <= after_reject
    states = {i["cluster_key"]: i["state"]
              for i in client.get(f"/api/runs/{run_id}/instances").json()}
    assert states[target["cluster_key"]] == "confirmed"
