"""Command-line entry point with upstream-compatible core options."""

import argparse
import asyncio
import logging
import math
import platform
from functools import partial
from pathlib import Path

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .dispatch_handler import DispatchEventHandler
from .env_args import parse_args
from .models import resolve_model
from .vad import VadSettings
from .whisper_cpp_handler import WhisperCppTranscriber

_LOGGER = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError("must be nonnegative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be positive and finite")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("must be nonnegative and finite")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise ValueError("must be between 0 and 1")
    return number


def prompt_budget(value: str) -> int:
    number = positive_int(value)
    if number > 223:
        raise ValueError("Whisper prompt budget cannot exceed 223")
    return number


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wyoming speech-to-text using whisper.cpp",
        epilog="Options also read WYO_WHISPER_<OPTION> and <OPTION>_FILE. CLI wins.",
    )
    parser.add_argument("--uri", required=True, help="tcp://HOST:PORT or unix:///PATH")
    parser.add_argument("--model", default="auto", help="GGML model name or .bin path")
    parser.add_argument("--data-dir", required=True, action="append", type=Path)
    parser.add_argument(
        "--download-dir", type=Path, help="Default: first data directory"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--language", default="auto")
    parser.add_argument(
        "--beam-size",
        type=nonnegative_int,
        default=0,
        help="0: auto (ARM 1, otherwise 5); 1: greedy; >1: beam search",
    )
    parser.add_argument("--cpu-threads", type=positive_int, default=4)
    parser.add_argument("--initial-prompt")
    parser.add_argument(
        "--whisper-task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="GPU requires a GPU-enabled pywhispercpp build",
    )
    parser.add_argument("--gpu-device", type=nonnegative_int, default=0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--max-audio-seconds", type=positive_float, default=120)
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Filter non-speech with Silero VAD (extra: vad)",
    )
    parser.add_argument(
        "--vad-endpointing",
        type=positive_float,
        help="Finish after this many seconds of silence following speech (extra: vad)",
    )
    parser.add_argument("--vad-threshold", type=probability, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=positive_int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=positive_int, default=500)
    parser.add_argument("--vad-pad-ms", type=nonnegative_int, default=200)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Emit revisable Wyoming transcript chunks while receiving audio",
    )
    parser.add_argument(
        "--streaming-interval",
        type=positive_float,
        default=1.0,
        help="Minimum new audio seconds between partial decodes",
    )
    parser.add_argument(
        "--hass-token",
        help="HA token; prefer WYO_WHISPER_HASS_TOKEN_FILE (extra: hass)",
    )
    parser.add_argument("--hass-api", default="http://homeassistant.local:8123/api")
    parser.add_argument("--hass-api-timeout", type=positive_float, default=10.0)
    parser.add_argument("--hass-refresh-seconds", type=nonnegative_float, default=0.0)
    parser.add_argument("--hass-prompt-max-tokens", type=prompt_budget, default=200)
    parser.add_argument("--hass-prompt-timeout", type=nonnegative_float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT)
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def build_info(
    model_name: str,
    languages: list[str],
    *,
    streaming: bool = False,
    endpointing: bool = False,
) -> Info:
    attribution = Attribution(
        name="ggml-org", url="https://github.com/ggml-org/whisper.cpp"
    )
    return Info(
        asr=[
            AsrProgram(
                name="whisper.cpp",
                description="Whisper.cpp transcription",
                attribution=attribution,
                installed=True,
                version=__version__,
                supports_transcript_streaming=streaming,
                requires_external_vad=not endpointing,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=attribution,
                        installed=True,
                        languages=languages,
                        version="1",
                    )
                ],
            )
        ]
    )


async def main() -> None:
    args = parse_args(get_parser())
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    if not args.uri.startswith(("tcp://", "unix://")):
        raise ValueError("Only tcp:// and unix:// are supported")
    if args.beam_size == 0:
        machine = platform.machine().lower()
        args.beam_size = 1 if "arm" in machine or "aarch" in machine else 5
    vad_settings = None
    if args.vad_filter or args.vad_endpointing is not None:
        try:
            import pysilero_vad  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install wyoming-whisper-cpp[vad] to enable VAD"
            ) from exc
        vad_settings = VadSettings(
            threshold=args.vad_threshold,
            min_speech_ms=args.vad_min_speech_ms,
            min_silence_ms=args.vad_min_silence_ms,
            pad_ms=args.vad_pad_ms,
            endpoint_seconds=args.vad_endpointing,
        )
    names = None
    if args.hass_token:
        from .hass_api import HomeAssistant
        from .name_cache import HassNameCache

        names = HassNameCache(
            HomeAssistant(args.hass_token, args.hass_api, args.hass_api_timeout),
            prefix=args.initial_prompt,
            max_tokens=args.hass_prompt_max_tokens,
            refresh_seconds=args.hass_refresh_seconds,
            wait_timeout=args.hass_prompt_timeout,
        )
    path = await asyncio.to_thread(
        resolve_model,
        args.model,
        args.data_dir,
        args.download_dir or args.data_dir[0],
        args.local_files_only,
    )
    _LOGGER.info("Loading %s", path)
    transcriber = await asyncio.to_thread(
        WhisperCppTranscriber,
        path,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        device=args.device,
        gpu_device=args.gpu_device,
        flash_attn=args.flash_attn,
        task=args.whisper_task,
    )
    model_name = path.stem.removeprefix("ggml-")
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready: %s", args.uri)
    try:
        if names is not None:
            names.schedule_refresh()
        await server.run(
            partial(
                DispatchEventHandler,
                build_info(
                    model_name,
                    transcriber.languages,
                    streaming=args.streaming,
                    endpointing=args.vad_endpointing is not None,
                ),
                transcriber,
                language=args.language,
                beam_size=args.beam_size,
                initial_prompt=args.initial_prompt,
                max_audio_seconds=args.max_audio_seconds,
                names=names,
                vad_settings=vad_settings,
                vad_filter=args.vad_filter,
                streaming=args.streaming,
                streaming_interval=args.streaming_interval,
            )
        )
    finally:
        await server.stop()
        if names is not None:
            await names.close()


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
