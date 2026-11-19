"""Incremental Silero VAD with speech filtering and optional endpoint detection."""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2


class Detector(Protocol):
    def reset(self) -> None: ...

    def process_array(self, samples: np.ndarray) -> float: ...


@dataclass(frozen=True)
class VadSettings:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    pad_ms: int = 200
    endpoint_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("VAD threshold must be between 0 and 1")
        if self.min_speech_ms <= 0 or self.min_silence_ms <= 0 or self.pad_ms < 0:
            raise ValueError("Invalid VAD speech/silence/padding duration")
        if self.endpoint_seconds is not None and (
            not math.isfinite(self.endpoint_seconds) or self.endpoint_seconds <= 0
        ):
            raise ValueError("VAD endpoint must be positive and finite")


def create_detector() -> Detector:
    from pysilero_vad import SileroVoiceActivityDetector

    return SileroVoiceActivityDetector()


class VadSession:
    """Per-client state. Positions refer to the original normalized PCM stream."""

    def __init__(self, settings: VadSettings, detector: Detector | None = None):
        self.settings = settings
        self.detector = detector if detector is not None else create_detector()
        self.detector.reset()
        self._pending = bytearray()
        self._offset = 0
        self._start: int | None = None
        self._last_voice = 0
        self._confirmed = False
        self._spans: list[tuple[int, int]] = []
        self.has_speech = False
        self.ended = False

    def process(self, audio: bytes) -> bool:
        """Consume complete frames; return False after a speech endpoint."""
        if len(audio) % 2:
            raise ValueError("Incomplete PCM sample")
        if self.ended:
            return False
        self._pending.extend(audio)
        consumed = 0
        while len(self._pending) - consumed >= FRAME_BYTES:
            frame = self._pending[consumed : consumed + FRAME_BYTES]
            consumed += FRAME_BYTES
            self._score(frame, FRAME_SAMPLES)
            if self.ended:
                break
        del self._pending[:consumed]
        if self.ended:
            self._pending.clear()
        return not self.ended

    def finish(self) -> None:
        """Score a final short frame with zero padding, without inventing duration."""
        if self._pending and not self.ended:
            size = len(self._pending) // 2
            frame = bytes(self._pending).ljust(FRAME_BYTES, b"\0")
            self._score(frame, size)
        self._pending.clear()

    def _score(self, frame: bytes, samples: int) -> None:
        values = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        probability = float(self.detector.process_array(values))
        start = self._offset
        self._offset += samples
        if probability >= self.settings.threshold:
            if self._start is None:
                self._start = start
            self._last_voice = self._offset
            if self._offset - self._start >= self.settings.min_speech_ms * 16:
                self._confirmed = True
                self.has_speech = True
        elif self._start is not None:
            if not self._confirmed:
                self._start = None
            elif self._offset - self._last_voice >= self.settings.min_silence_ms * 16:
                self._spans.append((self._start, self._last_voice))
                self._start = None
                self._confirmed = False
        endpoint = self.settings.endpoint_seconds
        if (
            endpoint is not None
            and self.has_speech
            and self._offset - self._last_voice >= endpoint * RATE
        ):
            self.ended = True

    @property
    def processed_samples(self) -> int:
        """Sample offset, stopping exactly at an endpoint within a large chunk."""
        return self._offset

    def speech_audio(self, audio: bytes) -> bytes:
        """Join confirmed speech regions with padding, merging overlaps."""
        spans = list(self._spans)
        if self._confirmed and self._start is not None:
            spans.append((self._start, self._last_voice))
        pad = self.settings.pad_ms * 16
        limit = min(len(audio) // 2, self._offset)
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            start, end = max(0, start - pad), min(limit, end + pad)
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return b"".join(audio[start * 2 : end * 2] for start, end in merged)
