import pytest
from edgecv.blobstore.store import BlobStore

@pytest.fixture
def store(tmp_path):
    return BlobStore(root=tmp_path)

def test_put_returns_sha256_of_content(store):
    ref = store.put(b"hello", kind="crop", fmt="png", width=1, height=1)
    assert ref.sha256 == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert ref.bytes_len == 5
    assert ref.deduped is False

def test_fanned_out_path_layout(store, tmp_path):
    ref = store.put(b"hello", kind="crop", fmt="png", width=1, height=1)
    expected = tmp_path / "crop" / ref.sha256[:2] / ref.sha256[2:4] / f"{ref.sha256}.png"
    assert expected.exists()

def test_identical_content_is_deduped(store):
    first = store.put(b"same", kind="crop", fmt="png", width=1, height=1)
    second = store.put(b"same", kind="crop", fmt="png", width=1, height=1)
    assert first.sha256 == second.sha256
    assert second.deduped is True

def test_dedup_does_not_rewrite_file(store):
    ref = store.put(b"same", kind="crop", fmt="png", width=1, height=1)
    path = store.path_for(ref.sha256, kind="crop", fmt="png")
    mtime = path.stat().st_mtime_ns
    store.put(b"same", kind="crop", fmt="png", width=1, height=1)
    assert path.stat().st_mtime_ns == mtime

def test_get_roundtrip(store):
    ref = store.put(b"payload", kind="thumbnail", fmt="webp", width=2, height=2)
    assert store.get(ref.sha256, kind="thumbnail", fmt="webp") == b"payload"

def test_kinds_are_separate_namespaces(store):
    crop = store.put(b"x", kind="crop", fmt="png", width=1, height=1)
    thumb = store.put(b"x", kind="thumbnail", fmt="png", width=1, height=1)
    assert crop.sha256 == thumb.sha256
    assert thumb.deduped is False

def test_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="kind"):
        store.put(b"x", kind="bogus", fmt="png", width=1, height=1)

def test_get_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        store.get("0" * 64, kind="crop", fmt="png")

# --- total_bytes: the numerator of the bytes-per-kilometre figure ------------
#
# Measured off the store rather than off `snippets.bytes`, because that column
# is a 0 placeholder for every row (see Repository._snippet_id). The store is
# the thing actually holding the bytes, so it is the honest place to ask.

def test_total_bytes_of_an_empty_store_is_zero(store):
    assert store.total_bytes() == 0


def test_total_bytes_sums_every_blob(store):
    store.put(b"12345", kind="crop", fmt="png", width=1, height=1)
    store.put(b"678", kind="thumbnail", fmt="webp", width=1, height=1)
    assert store.total_bytes() == 8


def test_total_bytes_does_not_double_count_a_deduped_put(store):
    store.put(b"12345", kind="crop", fmt="png", width=1, height=1)
    store.put(b"12345", kind="crop", fmt="png", width=1, height=1)
    assert store.total_bytes() == 5


def test_total_bytes_can_be_filtered_by_kind(store):
    store.put(b"12345", kind="crop", fmt="png", width=1, height=1)
    store.put(b"678", kind="thumbnail", fmt="webp", width=1, height=1)
    assert store.total_bytes(kind="crop") == 5
    assert store.total_bytes(kind="thumbnail") == 3


def test_total_bytes_ignores_partial_temp_files(store):
    """`put` writes a .tmp file then renames. A crashed write leaves one behind,
    and it is not stored data -- counting it would inflate the figure."""
    ref = store.put(b"12345", kind="crop", fmt="png", width=1, height=1)
    path = store.path_for(ref.sha256, kind="crop", fmt="png")
    path.with_suffix(path.suffix + ".tmp99999").write_bytes(b"garbage")
    assert store.total_bytes() == 5
