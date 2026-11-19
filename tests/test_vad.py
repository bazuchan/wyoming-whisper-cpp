"""Frame boundaries, filtering, endpointing, and real Silero smoke tests."""

import wave
from pathlib import Path

import pytest

from wyoming_whisper_cpp.vad import VadSession, VadSettings

FRAME = b"\x01\0" * 512


class Scores:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def process_array(self, samples):
        assert len(samples) == 512
        self.calls += 1
        return next(self.scores)


def test_chunk_boundaries_and_filter():
    detector = Scores([0, 0, 1, 1, 0, 0, 0])
    session = VadSession(
        VadSettings(min_speech_ms=64, min_silence_ms=64, pad_ms=0), detector
    )
    audio = FRAME * 7
    for offset in range(0, len(audio), 258):
        assert session.process(audio[offset : offset + 258])
    session.finish()
    assert session.has_speech
    assert session.speech_audio(audio) == FRAME * 2
    assert detector.calls == 7 and detector.resets == 1


def test_noise_does_not_trigger_speech_or_endpoint():
    session = VadSession(
        VadSettings(min_speech_ms=64, endpoint_seconds=0.064), Scores([1, 0, 0, 0, 0])
    )
    assert session.process(FRAME * 5)
    assert not session.has_speech
    assert session.speech_audio(FRAME * 5) == b""


def test_endpoint_only_after_confirmed_speech():
    session = VadSession(
        VadSettings(min_speech_ms=64, endpoint_seconds=0.096),
        Scores([0, 0, 1, 1, 0, 0, 0]),
    )
    assert session.process(FRAME * 6)
    assert not session.process(FRAME)
    assert session.has_speech and session.ended
    assert not session.process(FRAME)


def test_endpoint_inside_large_chunk():
    session = VadSession(
        VadSettings(min_speech_ms=32, endpoint_seconds=0.064), Scores([1, 0, 0])
    )
    assert not session.process(FRAME * 100)
    assert session.processed_samples == 512 * 3


def test_short_tail_is_scored_without_padding_duration():
    session = VadSession(VadSettings(min_speech_ms=16, pad_ms=0), Scores([1]))
    session.process(FRAME[:512])
    assert not session.has_speech
    session.finish()
    assert session.has_speech
    assert session.processed_samples == 256
    assert session.speech_audio(FRAME[:512]) == FRAME[:512]


def test_padding_merges_overlaps():
    session = VadSession(
        VadSettings(min_speech_ms=32, min_silence_ms=32, pad_ms=32),
        Scores([0, 1, 0, 1, 0]),
    )
    session.process(FRAME * 5)
    assert session.speech_audio(FRAME * 5) == FRAME * 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": float("nan")},
        {"min_speech_ms": 0},
        {"min_silence_ms": 0},
        {"pad_ms": -1},
        {"endpoint_seconds": float("nan")},
        {"endpoint_seconds": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        VadSettings(**kwargs)


def test_real_silero_silence_and_speech():
    session = VadSession(VadSettings(endpoint_seconds=0.5))
    assert session.process(b"\0" * 32000)
    assert not session.has_speech
    with wave.open(str(Path(__file__).parent / "speech.wav"), "rb") as wav:
        audio = wav.readframes(wav.getnframes())
    session.process(audio)
    session.process(b"\0" * 32000)
    session.finish()
    assert session.has_speech and session.ended
    assert session.speech_audio(b"\0" * 32000 + audio + b"\0" * 32000)
