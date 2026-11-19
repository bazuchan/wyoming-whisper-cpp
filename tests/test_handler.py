"""Exercise protocol lifecycle without downloading a model."""

import asyncio
import threading
import wave
from unittest.mock import AsyncMock

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Describe, Info

from wyoming_whisper_cpp.dispatch_handler import DispatchEventHandler


class FakeTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, language, **kwargs):
        with wave.open(str(path), "rb") as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (
                16000,
                2,
                1,
            )
            audio = wav.readframes(wav.getnframes())
        self.calls.append((audio, language, kwargs))
        return "Turn on the lamp."


def handler(model=None, **kwargs):
    instance = DispatchEventHandler(
        Info(),
        model or FakeTranscriber(),
        asyncio.StreamReader(),
        AsyncMock(),
        **kwargs,
    )
    instance.write_event = AsyncMock()
    return instance


def chunk(rate=16000, channels=1, width=2, audio=None):
    return AudioChunk(
        rate=rate,
        width=width,
        channels=channels,
        audio=audio if audio is not None else b"\0" * (rate // 10 * width * channels),
    ).event()


async def test_describe_and_empty_stop():
    h = handler()
    assert await h.handle_event(Describe().event())
    assert Info.is_type(h.write_event.call_args.args[0].type)
    assert not await h.handle_event(AudioStop().event())
    assert Transcript.from_event(h.write_event.call_args.args[0]).text == ""
    assert not h._transcriber.calls


@pytest.mark.parametrize(
    "rate,channels,width",
    [(16000, 1, 2), (44100, 2, 2), (48000, 1, 2), (16000, 1, 1), (16000, 2, 4)],
)
async def test_audio_normalization(rate, channels, width):
    h = handler(language="en", initial_prompt="lamp")
    await h.handle_event(Transcribe(language="ru-RU").event())
    await h.handle_event(AudioStart(rate=rate, width=width, channels=channels).event())
    await h.handle_event(chunk(rate, channels, width))
    assert not await h.handle_event(AudioStop().event())
    audio, language, options = h._transcriber.calls[0]
    assert 3100 <= len(audio) <= 3200
    assert language == "ru"
    assert options["initial_prompt"] == "lamp"


async def test_auto_overrides_default_and_duplicate_stop():
    h = handler(language="en")
    await h.handle_event(Transcribe(language="auto").event())
    await h.handle_event(chunk())
    await h.handle_event(AudioStop().event())
    await h.handle_event(AudioStop().event())
    assert h._transcriber.calls[0][1] is None
    assert h.write_event.await_count == 1


async def test_limits_and_invalid_frames():
    for h, event, code in [
        (handler(max_audio_seconds=0.01), chunk(), "audio-too-long"),
        (handler(), chunk(audio=b"x"), "invalid-audio"),
    ]:
        assert not await h.handle_event(event)
        assert Error.from_event(h.write_event.call_args.args[0]).code == code


async def test_backend_failure():
    class Broken:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("broken")

    h = handler(Broken())
    await h.handle_event(chunk())
    assert not await h.handle_event(AudioStop().event())
    assert Error.is_type(h.write_event.call_args.args[0].type)


async def test_clients_are_isolated():
    model = FakeTranscriber()
    first, second = handler(model), handler(model)
    await first.handle_event(chunk(audio=b"\x01\0" * 100))
    await second.handle_event(chunk(audio=b"\x02\0" * 100))
    await asyncio.gather(
        first.handle_event(AudioStop().event()),
        second.handle_event(AudioStop().event()),
    )
    assert {call[0] for call in model.calls} == {b"\x01\0" * 100, b"\x02\0" * 100}


async def test_inference_does_not_block_loop_and_cancel_keeps_wav():
    started, release = threading.Event(), threading.Event()

    class Blocking(FakeTranscriber):
        def transcribe(self, *args, **kwargs):
            started.set()
            assert release.wait(5)
            return super().transcribe(*args, **kwargs)

    model = Blocking()
    h = handler(model)
    await h.handle_event(chunk())
    pending = asyncio.create_task(h.handle_event(AudioStop().event()))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        await handler().handle_event(Describe().event())
        pending.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert len(model.calls) == 1
