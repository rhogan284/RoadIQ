from edgecv.config import Settings


def test_defaults_when_env_empty():
    s = Settings.from_env({})
    assert s.frames_stream == "frames"
    assert s.results_stream == "results"
    assert s.frames_maxlen == 1000
    assert s.blob_root.name == "blobs"


def test_env_overrides():
    s = Settings.from_env({"FRAMES_MAXLEN": "50", "FRAMES_STREAM": "f2"})
    assert s.frames_maxlen == 50
    assert s.frames_stream == "f2"
