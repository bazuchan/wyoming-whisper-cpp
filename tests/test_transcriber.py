"""Native parameter mapping and optional real-model smoke test."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from wyoming_whisper_cpp.whisper_cpp_handler import (
    WhisperCppTranscriber,
    normalize_language,
)

WAV = Path(__file__).parent / "speech.wav"


def test_language():
    assert normalize_language("ru_RU") == "ru"
    assert normalize_language("auto") is None


@pytest.mark.parametrize("beam,strategy", [(1, 0), (5, 1)])
def test_binding_parameters_and_serialization(beam, strategy, tmp_path):
    model_path = tmp_path / "local.bin"
    model_path.write_bytes(b"lmgg")
    active = 0
    guard = threading.Lock()

    def transcribe(audio, **kwargs):
        nonlocal active
        assert audio.ndim == 1 and str(audio.dtype) == "float32"
        assert kwargs["language"] == "en"
        assert kwargs["initial_prompt"] == ""
        with guard:
            active += 1
            assert active == 1
        time.sleep(0.01)
        with guard:
            active -= 1
        return [SimpleNamespace(text=" hello ")]

    with (
        patch("pywhispercpp.model.Model") as model,
        patch("_pywhispercpp.whisper_is_multilingual", return_value=False),
    ):
        model.return_value.transcribe.side_effect = transcribe
        backend = WhisperCppTranscriber(model_path, beam_size=beam)
        assert model.call_args.kwargs["params_sampling_strategy"] == strategy
        assert model.call_args.kwargs["context_params"]["use_gpu"] is False
        assert backend.languages == ["en"]
        with ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(lambda _: backend.transcribe(WAV, "auto", beam), range(2))
            )
        assert results == ["hello", "hello"]


def test_invalid_model_file(tmp_path):
    model_path = tmp_path / "invalid.bin"
    model_path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="GGML"):
        WhisperCppTranscriber(model_path)
    with pytest.raises(FileNotFoundError):
        WhisperCppTranscriber(tmp_path / "missing.bin")


@pytest.mark.integration
def test_real_model():
    model = os.environ.get("WHISPER_CPP_TEST_MODEL")
    if not model:
        pytest.skip("Set WHISPER_CPP_TEST_MODEL to a local GGML file")
    backend = WhisperCppTranscriber(Path(model), beam_size=1)
    text = backend.transcribe(WAV, "en", beam_size=1)
    assert "living room" in text.lower()
    assert "lamp" in text.lower()
