"""Local model lookup and atomic download failure handling."""

import io
from unittest.mock import patch

import pytest

from wyoming_whisper_cpp.models import resolve_model, resolve_name


def test_aliases():
    assert resolve_name("small-int8") == "small-q8_0"
    assert resolve_name("tiny.en-int8") == "tiny.en-q8_0"
    with patch("platform.machine", return_value="aarch64"):
        assert resolve_name("auto") == "tiny-q8_0"


def test_search_all_dirs_and_cache(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    second.mkdir()
    model = second / "ggml-tiny.bin"
    model.write_bytes(b"model")
    assert resolve_model("tiny", [first, second], first, True) == model
    assert resolve_model("tiny", [first], second, True) == model
    assert resolve_model(str(model), [], first, True) == model


def test_offline_and_missing_file(tmp_path):
    with patch("wyoming_whisper_cpp.models.urlopen") as download:
        with pytest.raises(FileNotFoundError):
            resolve_model("tiny", [], tmp_path, True)
        with pytest.raises(FileNotFoundError):
            resolve_model("missing.bin", [], tmp_path)
        download.assert_not_called()


@pytest.mark.parametrize("length,success", [(5, True), (10, False)])
def test_atomic_download(tmp_path, length, success):
    response = io.BytesIO(b"model")
    response.headers = {"Content-Length": str(length)}
    with patch("wyoming_whisper_cpp.models.urlopen", return_value=response):
        if success:
            assert resolve_model("tiny", [], tmp_path).read_bytes() == b"model"
        else:
            with pytest.raises(OSError):
                resolve_model("tiny", [], tmp_path)
            assert list(tmp_path.iterdir()) == []
