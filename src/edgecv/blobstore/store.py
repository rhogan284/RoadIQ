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

    def total_bytes(self, *, kind: str | None = None) -> int:
        """Bytes on disk, optionally for one kind only.

        This is the numerator of the bytes-per-kilometre figure. It is measured
        here and not summed from `snippets.bytes`, because that column is
        written as a 0 placeholder for every row -- the writer only ever sees
        content hashes on the wire, never blob sizes (see
        `Repository._snippet_id`). The store is what actually holds the bytes,
        so it is the honest place to ask, and asking it needs no change to
        frozen contract 2.

        Dedup falls out for free: a replayed run writes no new files, so it adds
        no bytes, which is the behaviour the crop store exists to produce.

        `.tmp*` files are skipped. `put()` writes to one and renames, so any
        left behind are the debris of a crashed write, not stored data.
        """
        root = self.root / kind if kind else self.root
        if not root.exists():
            return 0
        return sum(f.stat().st_size for f in root.rglob("*")
                   if f.is_file() and not f.suffix.startswith(".tmp"))
