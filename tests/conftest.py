"""Shared fixtures for the e2e suite.

Every test runs real components against live Postgres and Redis (`make up`), on real
RDD2022 images (`make dataset`) through the YOLOv12s detector (`make weights`). The
fixtures TRUNCATE Postgres and FLUSH Redis — point PG_DSN / REDIS_URL at a scratch
stack, never at a demo you want to keep.
"""
import json
import os
import random
from pathlib import Path

import psycopg
import pytest
import redis as redis_lib

PG_DSN = os.environ.get("PG_DSN", "postgresql://edgecv:edgecv@localhost:5432/edgecv")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

MANIFEST = Path("data/rdd2022/manifest.json")
GT = Path("data/rdd2022/ground_truth.json")
WEIGHTS = Path("weights/yolo12s_RDD2022_best.pt")


@pytest.fixture
def pg():
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        yield conn


@pytest.fixture
def clean_db(pg):
    from edgecv.db.migrate import apply_migrations
    apply_migrations(pg)
    with pg.cursor() as cur:
        cur.execute("TRUNCATE defect_instances, segment_condition, instance_reviews, "
                    "segments, detections, inferences, snippets, frames, "
                    "detectors, survey_runs, ground_truth, bench_runs")
    return pg


@pytest.fixture
def rds():
    client = redis_lib.from_url(REDIS_URL, decode_responses=False)
    client.flushdb()
    yield client
    client.flushdb()


@pytest.fixture(scope="session")
def detector():
    if not WEIGHTS.exists():
        pytest.skip("needs `make weights`")
    from edgecv.detectors.yolo12s import Yolo12sDetector
    return Yolo12sDetector(weights=WEIGHTS)


@pytest.fixture
def rdd_subset(tmp_path):
    """Write a manifest of `n_clean` + `n_defect` random RDD2022 test images."""
    if not MANIFEST.exists():
        pytest.skip("needs `make dataset`")
    full = json.loads(MANIFEST.read_text())

    def make(n_clean: int, n_defect: int, seed: int = 7) -> Path:
        rng = random.Random(seed)
        sub = {"clean": rng.sample(full["clean"], n_clean),
               "defect": rng.sample(full["defect"], n_defect)}
        path = tmp_path / f"manifest-{n_clean}-{n_defect}-{seed}.json"
        path.write_text(json.dumps(sub))
        return path
    return make


@pytest.fixture
def api(tmp_path, monkeypatch):
    """The read API exactly as the dashboard calls it, over this test's blob store."""
    monkeypatch.setenv("BLOB_ROOT", str(tmp_path / "blobs"))
    from fastapi.testclient import TestClient

    import edgecv.api.main as api_main
    monkeypatch.setattr(api_main, "SETTINGS", api_main.Settings.from_env())
    return TestClient(api_main.app)
