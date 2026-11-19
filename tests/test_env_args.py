"""Configuration compatibility with the upstream server."""

from pathlib import Path

import pytest

from wyoming_whisper_cpp.__main__ import get_parser
from wyoming_whisper_cpp.env_args import parse_args

BASE = {"WYO_WHISPER_URI": "tcp://127.0.0.1:10300", "WYO_WHISPER_DATA_DIR": "/data"}


def test_environment():
    args = parse_args(
        get_parser(),
        [],
        {**BASE, "WYO_WHISPER_CPU_THREADS": "2", "WYO_WHISPER_DEBUG": "yes"},
    )
    assert args.cpu_threads == 2
    assert args.debug
    assert args.data_dir == [Path("/data")]


def test_cli_wins_over_invalid_environment():
    args = parse_args(
        get_parser(),
        ["--cpu-threads=3", "--data-dir", "/local"],
        {**BASE, "WYO_WHISPER_CPU_THREADS": "bad"},
    )
    assert args.cpu_threads == 3
    assert args.data_dir == [Path("/local")]


def test_file_precedence(tmp_path):
    secret = tmp_path / "prompt"
    secret.write_text("Kitchen lamp\n")
    args = parse_args(
        get_parser(),
        [],
        {
            **BASE,
            "WYO_WHISPER_INITIAL_PROMPT": "other",
            "WYO_WHISPER_INITIAL_PROMPT_FILE": str(secret),
        },
    )
    assert args.initial_prompt == "Kitchen lamp"


@pytest.mark.parametrize(
    "option,value",
    [
        ("--cpu-threads", "0"),
        ("--beam-size", "-1"),
        ("--max-audio-seconds", "nan"),
        ("--max-audio-seconds", "inf"),
        ("--device", "cuda"),
        ("--streaming-interval", "0"),
        ("--streaming-interval", "nan"),
        ("--vad-endpointing", "-1"),
        ("--vad-threshold", "1"),
        ("--vad-threshold", "nan"),
        ("--vad-min-speech-ms", "0"),
        ("--hass-prompt-max-tokens", "224"),
        ("--hass-prompt-max-tokens", "0"),
        ("--hass-refresh-seconds", "-1"),
        ("--hass-prompt-timeout", "nan"),
    ],
)
def test_invalid_values(option, value):
    with pytest.raises(SystemExit):
        parse_args(get_parser(), [option, value], BASE)


def test_feature_environment_and_token_file(tmp_path):
    secret = tmp_path / "token"
    secret.write_text("test-secret\n")
    args = parse_args(
        get_parser(),
        [],
        {
            **BASE,
            "WYO_WHISPER_HASS_TOKEN_FILE": str(secret),
            "WYO_WHISPER_VAD_FILTER": "true",
            "WYO_WHISPER_VAD_ENDPOINTING": "1.2",
            "WYO_WHISPER_STREAMING": "true",
            "WYO_WHISPER_STREAMING_INTERVAL": "0.5",
        },
    )
    assert args.hass_token == "test-secret"
    assert args.vad_filter and args.streaming
    assert args.vad_endpointing == 1.2 and args.streaming_interval == 0.5


def test_unknown_options_not_silently_accepted():
    with pytest.raises(SystemExit):
        parse_args(get_parser(), ["--compute-type", "int8"], BASE)
