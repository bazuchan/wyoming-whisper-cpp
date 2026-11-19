"""Wyoming audio lifecycle, VAD, vocabulary, and revisable streaming results."""

import asyncio
import logging
import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .vad import VadSession, VadSettings
from .whisper_cpp_handler import WhisperCppTranscriber, normalize_language

if TYPE_CHECKING:
    from .name_cache import HassNameCache

_LOGGER = logging.getLogger(__name__)


class DispatchEventHandler(AsyncEventHandler):
    """One isolated utterance per connection; one final Transcript in every mode."""

    def __init__(
        self,
        info: Info,
        transcriber: WhisperCppTranscriber,
        *args,
        language: str | None = None,
        beam_size: int = 5,
        initial_prompt: str | None = None,
        max_audio_seconds: float = 120,
        names: "HassNameCache | None" = None,
        vad_settings: VadSettings | None = None,
        vad_filter: bool = False,
        streaming: bool = False,
        streaming_interval: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info = info.event()
        self._transcriber = transcriber
        self._language = normalize_language(language)
        self._beam_size = beam_size
        self._initial_prompt = initial_prompt
        self._names = names
        self._refreshed_names = False
        self._prompt_ready = False
        self._vad_settings = vad_settings
        self._vad_filter = vad_filter
        self._vad: VadSession | None = None
        self._vad_busy = False
        self._max_bytes = int(max_audio_seconds * 16000) * 2
        self._audio = bytearray()
        self._converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._finished = False
        self._streaming = streaming
        self._stream_started = False
        self._stream_step = max(2, int(streaming_interval * 16000) * 2)
        self._stream_offset = 0
        self._partial_task: asyncio.Task | None = None
        self._partial_failed = False
        self._last_partial: str | None = None
        self._write_lock = asyncio.Lock()

    async def write_event(self, event: Event) -> None:
        # Describe replies and the streaming worker share the same socket.
        async with self._write_lock:
            await super().write_event(event)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info)
            return True
        if self._finished:
            return False
        if Transcribe.is_type(event.type):
            if self._audio:
                return await self._error("Request already started", "invalid-request")
            request = Transcribe.from_event(event)
            if request.language is not None:
                self._language = normalize_language(request.language)
            if self._vad_settings and self._vad_settings.endpoint_seconds:
                sensitivity = request.vad_sensitivity
                factor = {"relaxed": 1.5, "aggressive": 0.5}.get(sensitivity, 1.0)
                self._vad_settings = replace(
                    self._vad_settings,
                    endpoint_seconds=self._vad_settings.endpoint_seconds * factor,
                )
            return True
        if AudioStart.is_type(event.type):
            if self._audio:
                return await self._error("Audio already started", "invalid-request")
            self._refresh_names()
            self._converter = AudioChunkConverter(rate=16000, width=2, channels=1)
            return True
        if AudioChunk.is_type(event.type):
            self._refresh_names()
            try:
                chunk = AudioChunk.from_event(event)
                if (
                    chunk.rate <= 0
                    or chunk.width not in (1, 2, 3, 4)
                    or chunk.channels not in (1, 2)
                    or len(chunk.audio) % (chunk.width * chunk.channels)
                ):
                    raise ValueError("Expected complete mono/stereo PCM frames")
                chunk = self._converter.convert(chunk)
            except (ValueError, KeyError, TypeError) as err:
                return await self._error(str(err), "invalid-audio")
            if len(self._audio) + len(chunk.audio) > self._max_bytes:
                return await self._error(
                    "Audio duration limit exceeded", "audio-too-long"
                )
            self._audio.extend(chunk.audio)
            if self._vad_settings is not None:
                self._vad_busy = True
                try:
                    if self._vad is None:
                        self._vad = await asyncio.to_thread(
                            VadSession, self._vad_settings
                        )
                    keep_listening = await asyncio.to_thread(
                        self._vad.process, chunk.audio
                    )
                except Exception:
                    _LOGGER.exception("VAD failed")
                    return await self._error("VAD failed; see server log", "vad-failed")
                finally:
                    self._vad_busy = False
                if not keep_listening:
                    return await self._finish()
            if self._streaming:
                await self._start_stream()
                self._schedule_partial()
            return True
        if AudioStop.is_type(event.type):
            return await self._finish()
        return True

    def _refresh_names(self) -> None:
        if self._names is not None and not self._refreshed_names:
            self._refreshed_names = True
            self._names.schedule_refresh()

    async def _prompt(self, wait: bool) -> str | None:
        if not self._prompt_ready:
            if self._names is not None:
                self._initial_prompt = await self._names.initial_prompt(
                    self._transcriber, wait=wait
                )
            # All partial hypotheses and the final decode use the same prompt.
            self._prompt_ready = True
        return self._initial_prompt

    async def _start_stream(self) -> None:
        if not self._stream_started:
            self._stream_started = True
            await self.write_event(TranscriptStart().event())

    def _schedule_partial(self) -> None:
        if self._partial_failed or self._finished:
            return
        if self._partial_task is not None and not self._partial_task.done():
            return
        if len(self._audio) - self._stream_offset >= self._stream_step:
            self._partial_task = asyncio.create_task(self._partial_loop())

    def _decode_audio(self) -> bytes:
        audio = bytes(self._audio)
        if self._vad is not None:
            if not self._vad.has_speech:
                return b""
            if self._vad_filter:
                return self._vad.speech_audio(audio)
            if self._vad.ended:
                return audio[: self._vad.processed_samples * 2]
        return audio

    async def _partial_loop(self) -> None:
        try:
            # At most one pending decode per client: slow inference coalesces
            # incoming chunks into the newest snapshot instead of queueing jobs.
            while (
                not self._finished
                and not self._vad_busy
                and len(self._audio) - self._stream_offset >= self._stream_step
            ):
                self._stream_offset = len(self._audio)
                audio = self._decode_audio()
                if not audio:
                    continue
                prompt = await self._prompt(wait=False)
                text = await self._decode(audio, prompt)
                if not self._finished and text != self._last_partial:
                    await self.write_event(TranscriptChunk(text=text).event())
                    self._last_partial = text
        except Exception:
            self._partial_failed = True
            _LOGGER.exception(
                "Partial decode failed; retaining final-only transcription"
            )

    async def _finish(self) -> bool:
        self._finished = True
        try:
            await self._drain_partial()
            if self._vad is not None:
                await asyncio.to_thread(self._vad.finish)
            audio = self._decode_audio()
            text = (
                await self._decode(audio, await self._prompt(wait=True))
                if audio
                else ""
            )
            _LOGGER.info("%s", text)
            if self._streaming:
                await self._start_stream()
                if text != self._last_partial:
                    await self.write_event(TranscriptChunk(text=text).event())
            await self.write_event(Transcript(text=text).event())
            if self._streaming:
                await self.write_event(TranscriptStop().event())
        except Exception:
            _LOGGER.exception("Transcription failed")
            await self.write_event(
                Error(
                    "Transcription failed; see server log", code="transcribe-failed"
                ).event()
            )
        finally:
            self._audio.clear()
        return False

    async def _error(self, text: str, code: str) -> bool:
        self._finished = True
        await self._drain_partial()
        self._audio.clear()
        await self.write_event(Error(text=text, code=code).event())
        return False

    async def _drain_partial(self) -> None:
        if self._partial_task is not None:
            await self._partial_task
            self._partial_task = None

    async def _decode(self, audio: bytes, prompt: str | None) -> str:
        with tempfile.TemporaryDirectory(prefix="wyoming-whisper-") as directory:
            path = Path(directory) / "speech.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(audio)
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._transcriber.transcribe,
                    path,
                    self._language,
                    beam_size=self._beam_size,
                    initial_prompt=prompt,
                )
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Native inference outlives a cancelled awaiter. Keep its WAV alive.
                try:
                    await task
                finally:
                    raise

    async def disconnect(self) -> None:
        self._finished = True
        await self._drain_partial()
        self._audio.clear()
