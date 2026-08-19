from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    pg_dsn: str = "postgresql://edgecv:edgecv@localhost:5432/edgecv"
    redis_url: str = "redis://localhost:6379/0"
    frames_stream: str = "frames"
    results_stream: str = "results"
    frames_maxlen: int = 1000
    blob_root: Path = Path("./blobs")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        defaults = cls()
        return cls(
            pg_dsn=e.get("PG_DSN", defaults.pg_dsn),
            redis_url=e.get("REDIS_URL", defaults.redis_url),
            frames_stream=e.get("FRAMES_STREAM", defaults.frames_stream),
            results_stream=e.get("RESULTS_STREAM", defaults.results_stream),
            frames_maxlen=int(e.get("FRAMES_MAXLEN", defaults.frames_maxlen)),
            blob_root=Path(e.get("BLOB_ROOT", str(defaults.blob_root))),
        )
