"""Content-addressed blob storage.

Snippets are keyed by sha256 of their bytes, so replaying the same source image across
many benchmark runs stores its crop exactly once. That dedup is what makes repeat runs
nearly free on disk.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["crop", "thumbnail"]
VALID_KINDS: frozenset[str] = frozenset({"crop", "thumbnail"})


@dataclass(frozen=True, slots=True)
class BlobRef:
    sha256: str
    kind: Kind
    fmt: str
    bytes_len: int
    width: int
    height: int
    deduped: bool


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, sha256: str, *, kind: str, fmt: str) -> Path:
        return self.root / kind / sha256[:2] / sha256[2:4] / f"{sha256}.{fmt}"

    def put(self, data: bytes, *, kind: str, fmt: str, width: int,
            height: int) -> BlobRef:
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(VALID_KINDS)}")
        sha256 = hashlib.sha256(data).hexdigest()
        path = self.path_for(sha256, kind=kind, fmt=fmt)
        deduped = path.exists()
        if not deduped:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file then rename, so a crash never leaves a partial
            # object at a path whose name claims a hash it doesn't have.
            tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            tmp.replace(path)
        return BlobRef(sha256=sha256, kind=kind, fmt=fmt, bytes_len=len(data),
                       width=width, height=height, deduped=deduped)

    def get(self, sha256: str, *, kind: str, fmt: str) -> bytes:
        return self.path_for(sha256, kind=kind, fmt=fmt).read_bytes()

    def exists(self, sha256: str, *, kind: str, fmt: str) -> bool:
        return self.path_for(sha256, kind=kind, fmt=fmt).exists()
