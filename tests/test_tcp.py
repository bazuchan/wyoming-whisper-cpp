"""Real server lifecycle and TCP protocol test with a provisioned model."""

import asyncio
import os
import socket
import sys
import wave
from pathlib import Path

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

from wyoming_whisper_cpp.health_check import check


@pytest.mark.integration
@pytest.mark.parametrize("features", [False, True])
async def test_tcp_server(tmp_path, features):
    model = os.environ.get("WHISPER_CPP_TEST_MODEL")
    if not model:
        pytest.skip("Set WHISPER_CPP_TEST_MODEL to a local GGML file")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uri = f"tcp://127.0.0.1:{port}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_whisper_cpp",
        "--uri",
        uri,
        "--model",
        model,
        "--data-dir",
        str(tmp_path),
        "--local-files-only",
        "--beam-size",
        "1",
        *(
            [
                "--streaming",
                "--streaming-interval",
                "0.5",
                "--vad-filter",
                "--vad-endpointing",
                "1.0",
            ]
            if features
            else []
        ),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                await asyncio.wait_for(check(uri), 1)
                break
            except (OSError, asyncio.TimeoutError):
                assert process.returncode is None, "Server exited during startup"
                await asyncio.sleep(0.05)
        else:
            pytest.fail("Server did not start")
        async with AsyncClient.from_uri(uri) as client:
            await client.write_event(Transcribe(language="en-US").event())
            with wave.open(str(Path(__file__).parent / "speech.wav"), "rb") as wav:
                await client.write_event(
                    AudioStart(rate=16000, width=2, channels=1).event()
                )
                if features:
                    # A partial must arrive while the client is still recording,
                    # not just as a disguised final response after AudioStop.
                    await client.write_event(
                        AudioChunk(
                            rate=16000,
                            width=2,
                            channels=1,
                            audio=wav.readframes(32000),
                        ).event()
                    )
                    start = await asyncio.wait_for(client.read_event(), 30)
                    partial = await asyncio.wait_for(client.read_event(), 30)
                    assert start.type == "transcript-start"
                    assert partial.type == "transcript-chunk"
                while audio := wav.readframes(1600):
                    await client.write_event(
                        AudioChunk(
                            rate=16000,
                            width=2,
                            channels=1,
                            audio=audio,
                        ).event()
                    )
            if features:
                # No audio-stop: VAD must detect the endpoint itself.
                await client.write_event(
                    AudioChunk(
                        rate=16000,
                        width=2,
                        channels=1,
                        audio=b"\0" * 48000,
                    ).event()
                )
            else:
                await client.write_event(AudioStop().event())
            event = await asyncio.wait_for(client.read_event(), 30)
            while event is not None and event.type == "transcript-chunk":
                event = await asyncio.wait_for(client.read_event(), 30)
            assert event is not None and Transcript.is_type(event.type)
            text = Transcript.from_event(event).text.lower()
            assert "living room" in text and "lamp" in text
            if features:
                stop = await asyncio.wait_for(client.read_event(), 5)
                assert stop.type == "transcript-stop"
            assert await asyncio.wait_for(client.read_event(), 5) is None
        await check(uri)
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                pytest.fail("Server did not stop on SIGTERM")
    assert process.returncode == 0
