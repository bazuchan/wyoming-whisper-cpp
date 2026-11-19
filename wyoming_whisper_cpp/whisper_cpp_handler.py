"""Synchronous transcriber, matching the upstream batch backend interface."""

import threading
import wave
from pathlib import Path

import numpy as np


def normalize_language(language: str | None) -> str | None:
    """Accept Home Assistant locale tags as well as bare Whisper codes."""
    if not language or language.lower() == "auto":
        return None
    return language.lower().replace("_", "-").split("-", 1)[0]


class WhisperCppTranscriber:
    """Own one native context; serialize access even after async cancellation."""

    def __init__(
        self,
        model_path: Path,
        *,
        beam_size: int = 5,
        cpu_threads: int = 4,
        device: str = "cpu",
        gpu_device: int = 0,
        flash_attn: bool = False,
        task: str = "transcribe",
    ) -> None:
        import _pywhispercpp as native
        from pywhispercpp.model import Model

        # Fail in Python rather than passing unreadable/non-GGML files to native code.
        with model_path.open("rb") as model_file:
            if model_file.read(4) != b"lmgg":
                raise ValueError(f"Not a whisper.cpp GGML model: {model_path}")
        self._lock = threading.Lock()
        self.beam_size = beam_size
        self.model = Model(
            str(model_path),
            params_sampling_strategy=0 if beam_size == 1 else 1,
            context_params={
                "use_gpu": device == "gpu",
                "gpu_device": gpu_device,
                "flash_attn": flash_attn,
            },
            n_threads=cpu_threads,
            translate=task == "translate",
            no_context=True,
            no_timestamps=True,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
        )
        # The binding exposes this query only through its native context API.
        self.languages = (
            Model.available_languages()
            if native.whisper_is_multilingual(self.model._ctx)
            else ["en"]
        )

    def transcribe(
        self,
        wav_path: str | Path,
        language: str | None,
        beam_size: int = 5,
        initial_prompt: str | None = None,
    ) -> str:
        if beam_size != self.beam_size:
            raise ValueError("Sampling strategy is fixed when the model is loaded")
        language = normalize_language(language)
        if self.languages == ["en"]:
            language = "en"
        if language is not None and language not in self.languages:
            raise ValueError(f"Unsupported language: {language}")
        with wave.open(str(wav_path), "rb") as wav:
            if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) != (
                16000,
                2,
                1,
            ):
                raise ValueError("Expected 16 kHz, 16-bit, mono PCM")
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if audio.size == 0:
            return ""
        samples = audio.astype(np.float32) / 32768.0
        # The lock lives in the worker, not in the cancelled asyncio caller.
        with self._lock:
            segments = self.model.transcribe(
                samples,
                language=language or "auto",
                initial_prompt=initial_prompt or "",
                beam_search={"beam_size": beam_size, "patience": -1.0},
                greedy={"best_of": 1},
            )
            return (
                " ".join(segment.text.strip() for segment in segments)
                .replace("[BLANK_AUDIO]", "")
                .strip()
            )
