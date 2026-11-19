"""Streaming protocol, bounded background jobs, and VAD/name integration."""

import asyncio
import threading
from unittest.mock import patch

from test_handler import FakeTranscriber, chunk, handler
from test_name_cache import FakeHass
from test_vad import FRAME, Scores
from wyoming.asr import Transcript
from wyoming.audio import AudioStop
from wyoming.info import Describe

from wyoming_whisper_cpp.__main__ import build_info
from wyoming_whisper_cpp.name_cache import HassNameCache
from wyoming_whisper_cpp.vad import VadSession, VadSettings


def events(h):
    return [call.args[0] for call in h.write_event.call_args_list]


async def test_partial_before_audio_stop_and_final_order():
    h = handler(streaming=True, streaming_interval=0.01)
    await h.handle_event(chunk())
    await asyncio.wait_for(h._partial_task, 1)
    assert [event.type for event in events(h)] == [
        "transcript-start",
        "transcript-chunk",
    ]
    assert not await h.handle_event(AudioStop().event())
    assert [event.type for event in events(h)][-2:] == ["transcript", "transcript-stop"]
    assert len(h._transcriber.calls) == 2
    assert sum(event.type == "transcript" for event in events(h)) == 1
    assert not await h.handle_event(AudioStop().event())


async def test_partial_chunks_replace_previous_text():
    class Revising(FakeTranscriber):
        def transcribe(self, *args, **kwargs):
            super().transcribe(*args, **kwargs)
            return ["Turn on a", "Turn on the lamp."][min(len(self.calls) - 1, 1)]

    h = handler(Revising(), streaming=True, streaming_interval=0.01)
    await h.handle_event(chunk())
    await h._partial_task
    await h.handle_event(AudioStop().event())
    texts = [
        event.data["text"] for event in events(h) if event.type == "transcript-chunk"
    ]
    assert texts == ["Turn on a", "Turn on the lamp."]
    assert Transcript.from_event(events(h)[-2]).text == texts[-1]


async def test_slow_partial_coalesces_and_does_not_block_reader():
    started, release = threading.Event(), threading.Event()

    class Slow(FakeTranscriber):
        def transcribe(self, *args, **kwargs):
            started.set()
            assert release.wait(5)
            return super().transcribe(*args, **kwargs)

    h = handler(Slow(), streaming=True, streaming_interval=0.01)
    await h.handle_event(chunk())
    assert await asyncio.to_thread(started.wait, 2)
    pending = h._partial_task
    try:
        for _ in range(10):
            await h.handle_event(chunk())
        await h.handle_event(Describe().event())
        assert pending is h._partial_task
        ending = asyncio.create_task(h.handle_event(AudioStop().event()))
        await asyncio.sleep(0)
    finally:
        release.set()
    await asyncio.wait_for(ending, 2)
    assert len(h._transcriber.calls) == 2  # One partial and one final, not eleven jobs.
    assert len(h._transcriber.calls[-1][0]) == 11 * 3200
    assert events(h)[-1].type == "transcript-stop"


async def test_disconnect_drains_worker_and_suppresses_late_partial():
    started, release = threading.Event(), threading.Event()

    class Slow(FakeTranscriber):
        def transcribe(self, *args, **kwargs):
            started.set()
            assert release.wait(5)
            return super().transcribe(*args, **kwargs)

    h = handler(Slow(), streaming=True, streaming_interval=0.01)
    await h.handle_event(chunk())
    assert await asyncio.to_thread(started.wait, 2)
    cleanup = asyncio.create_task(h.disconnect())
    try:
        await asyncio.sleep(0)
    finally:
        release.set()
    await asyncio.wait_for(cleanup, 2)
    assert [event.type for event in events(h)] == ["transcript-start"]
    assert h._partial_task is None and not h._audio


async def test_partial_failure_can_recover_at_final_decode():
    class OnceBroken(FakeTranscriber):
        def transcribe(self, *args, **kwargs):
            super().transcribe(*args, **kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("partial failed")
            return "Recovered"

    h = handler(OnceBroken(), streaming=True, streaming_interval=0.01)
    await h.handle_event(chunk())
    await h._partial_task
    await h.handle_event(AudioStop().event())
    assert events(h)[-2].data["text"] == "Recovered"
    assert events(h)[-1].type == "transcript-stop"


async def test_silent_stream_never_calls_whisper():
    settings = VadSettings(min_speech_ms=32)
    session = VadSession(settings, Scores([0] * 4))
    h = handler(
        streaming=True, streaming_interval=0.01, vad_settings=settings, vad_filter=True
    )
    with patch("wyoming_whisper_cpp.dispatch_handler.VadSession", return_value=session):
        await h.handle_event(chunk(audio=FRAME * 4))
        await h._partial_task
        await h.handle_event(AudioStop().event())
    assert not h._transcriber.calls
    assert events(h)[-2].data["text"] == ""


async def test_endpoint_finishes_without_audio_stop_and_ignores_trailing_audio():
    settings = VadSettings(min_speech_ms=32, endpoint_seconds=0.064)
    session = VadSession(settings, Scores([1, 0, 0]))
    h = handler(vad_settings=settings)
    with patch("wyoming_whisper_cpp.dispatch_handler.VadSession", return_value=session):
        assert not await h.handle_event(chunk(audio=FRAME * 100))
    assert len(h._transcriber.calls[0][0]) == len(FRAME) * 3
    await h.handle_event(AudioStop().event())
    assert sum(event.type == "transcript" for event in events(h)) == 1


async def test_vad_filter_and_names_reach_backend():
    settings = VadSettings(min_speech_ms=32, min_silence_ms=32, pad_ms=0)
    session = VadSession(settings, Scores([0, 1, 0]))
    cache = HassNameCache(FakeHass())
    h = handler(names=cache, vad_settings=settings, vad_filter=True)
    try:
        with patch(
            "wyoming_whisper_cpp.dispatch_handler.VadSession", return_value=session
        ):
            await h.handle_event(chunk(audio=FRAME * 3))
            await h.handle_event(AudioStop().event())
        audio, _, kwargs = h._transcriber.calls[0]
        assert audio == FRAME
        assert kwargs["initial_prompt"] == "Office Lamp, Desk Lamp."
        assert cache._hass.calls == 1
    finally:
        await cache.close()


async def test_prompt_is_frozen_across_partial_and_final():
    hass = FakeHass()
    hass.gate.clear()
    cache = HassNameCache(hass, prefix="Names:")
    h = handler(names=cache, streaming=True, streaming_interval=0.01)
    try:
        await h.handle_event(chunk())
        await h._partial_task
        hass.gate.set()
        await cache._task
        await h.handle_event(AudioStop().event())
        assert [call[2]["initial_prompt"] for call in h._transcriber.calls] == [
            "Names:",
            "Names:",
        ]
    finally:
        await cache.close()


def test_advertised_capabilities():
    program = build_info("base", ["en"], streaming=True, endpointing=True).asr[0]
    assert program.supports_transcript_streaming
    assert not program.requires_external_vad
    program = build_info("base", ["en"]).asr[0]
    assert not program.supports_transcript_streaming and program.requires_external_vad
