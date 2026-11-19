# Wyoming Whisper.cpp

A Wyoming speech-to-text server using **whisper.cpp**, organized like
[OHF-Voice/wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper):
a protocol dispatcher, a separate synchronous transcriber, a persistent model,
background inference, and familiar command-line/environment configuration.

This is an independent implementation, not an official OHF release or a complete
feature-equivalent fork. See [comparison and design](docs/comparison.md).

## Install

Python 3.10+ is required. Python 3.11 is used by the Docker image.

```sh
python3 -m venv .venv
.venv/bin/pip install '.[hass,vad]'
script/run --model base-q8_0 --language ru \
  --uri tcp://0.0.0.0:10300 --data-dir ./data
```

In Home Assistant, add the **Wyoming Protocol** integration with this machine's
address and port `10300`, then select `whisper.cpp` in the voice pipeline.
This service does not implement authentication or TLS: expose it only on a trusted
network. Automatic mDNS discovery is not implemented.

## Docker

```sh
export RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
docker compose up --build -d
```

The default image builds whisper.cpp with **Vulkan** and requests GPU offload.
Compose passes `/dev/dri` through and requires `RENDER_GID` so the non-root user
can access the host's render device. If using another render node, obtain its
GID instead. A host without `/dev/dri` cannot run the default Compose configuration;
use the explicit CPU command below.

Models persist in a named volume. The image runs as a non-root user. For bind mounts, give UID 1000 write access to the model
directory. Initial model downloads can take longer than the healthcheck startup
period. The healthcheck assumes port 10300; override it if you change `--uri`.
The image includes both optional extras. The supplied Compose configuration enables
VAD filtering, a one-second speech endpoint, and streaming; the CLI leaves these
features disabled unless explicitly requested.

Direct Vulkan build and run (hardware/driver dependent):

```sh
docker build -t wyoming-whisper-cpp:vulkan .
docker run --rm -p 10300:10300 --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -v whisper-models:/data wyoming-whisper-cpp:vulkan \
  --model base-q8_0 --language ru --device gpu
```

The Vulkan image builds pywhispercpp from source; it is not a GPU performance
guarantee. CUDA/ROCm/Metal require a suitable pywhispercpp build outside this
Dockerfile. `--device gpu` requests GPU offload, but the native library may fall
back to CPU; check whisper.cpp's startup log. `--gpu-device` selects its native
GPU index. `--flash-attn` depends on backend/model support. Source builds default
to two parallel compile jobs; override `--build-arg CMAKE_BUILD_PARALLEL_LEVEL=N`
if desired.

Explicit CPU-only build and run (no render device required):

```sh
docker build --build-arg WHISPER_CPP_VULKAN=OFF -t wyoming-whisper-cpp:cpu .
docker run --rm -p 10300:10300 -v whisper-models:/data \
  wyoming-whisper-cpp:cpu --device cpu --model base-q8_0 --language ru \
  --vad-filter --vad-endpointing 1.0 --streaming
```

`--device cpu` also disables offload in the Vulkan image. The Docker environment
selects `gpu`; a local pip installation still defaults to `cpu` because its
pywhispercpp wheel may not support a GPU backend.

## Published Vulkan image

GitHub Actions builds the Vulkan image for **linux/amd64** and publishes it to
`ghcr.io/bazuchan/wyoming-whisper-cpp` after the Python test matrix and image
smoke checks succeed. The exact locally checked image is pushed, not rebuilt.

- Push to the default branch: `:vulkan`, `:latest`, and `:vulkan-sha-<short-sha>`.
- Version tag such as `v0.2.0`: `:0.2.0-vulkan` and `:vulkan-sha-<short-sha>`.
- Pull requests and non-default branches: build/check only, no registry login or push.
- Manual runs are available through **Actions > Test and Publish > Run workflow**;
  publication uses the same default-branch/version-tag policy.

After the first successful publication:

```sh
docker pull ghcr.io/bazuchan/wyoming-whisper-cpp:vulkan
```

Use that address as `image:` in your deployment Compose file instead of `build:`.
Keep the existing render-device, group, model volume and environment settings.
No Docker Hub secret is needed: the workflow uses `GITHUB_TOKEN` with
`packages: write`. For anonymous pulls, set the resulting GHCR package visibility
to **Public** in its package settings. Private packages require an authenticated
pull. Existing packages must grant this repository Actions access. OCI labels
link the image to its source repository. The workflow derives the registry path
from the repository name, so forks publish to their own namespace.

## Configuration

Every option also reads `WYO_WHISPER_<OPTION>` (dashes become underscores), or
`WYO_WHISPER_<OPTION>_FILE`. Precedence: explicit CLI, file, environment, default.
For example `WYO_WHISPER_INITIAL_PROMPT_FILE=/run/secrets/prompt`.
Repeated `--data-dir` values use the OS path separator in the environment (`:`
on Linux). CLI `--data-dir` replaces, rather than appends to, the environment list.

| Option | Default / meaning |
| --- | --- |
| `--uri` | Required; `tcp://HOST:PORT` or `unix:///PATH` |
| `--data-dir` | Required, repeatable; model search directories |
| `--download-dir` | First data directory |
| `--model` | `auto`: ARM tiny-q8_0, otherwise base-q8_0 |
| `--local-files-only` | Never download missing models |
| `--language` | `auto`; request language overrides this default |
| `--beam-size` | `0`: ARM 1, otherwise 5; `1` greedy; `>1` beam search |
| `--cpu-threads` | 4 |
| `--initial-prompt` | Optional context, e.g. room and device names |
| `--whisper-task` | `transcribe` or `translate` (to English) |
| `--device` | `gpu` in Docker/Compose; `cpu` for local CLI |
| `--gpu-device` | 0 |
| `--flash-attn` | Disabled |
| `--max-audio-seconds` | 120 per connection |
| `--debug`, `--log-format`, `--version` | Logging and version |

Use model names from pywhispercpp, e.g. `tiny`, `base.en`, `small-q5_1`,
`large-v3-turbo-q5_0`, or an existing GGML `.bin` path. `small-int8` and
`small.int8` are compatibility aliases for `small-q8_0`, **not** CTranslate2
models. The service searches every data directory and the download directory;
missing known models download from ggerganov/whisper.cpp on Hugging Face through
an atomic temporary file. Downloads use HTTPS and check length, but do not have
pinned model checksums. Use pre-provisioned files and `--local-files-only` for
controlled offline deployment.

English-only models advertise only English (queried from the native model, not
inferred from its filename). Locale tags such as `ru-RU` normalize to `ru`;
`auto` explicitly requests detection, including when a default language is set.
The binding's language list is advertised for multilingual models.

## VAD and automatic endpointing

```sh
script/run --model base-q8_0 --language ru --data-dir ./data \
  --uri tcp://0.0.0.0:10300 --vad-filter --vad-endpointing 1.0
```

`--vad-filter` removes non-speech regions before decoding with Silero VAD v5.
The ONNX model is included in the `pysilero-vad` extra; no separate model download
is needed. Every connection has independent recurrent VAD state. Detection runs
on normalized 16 kHz audio, in 512-sample windows, outside the event loop.

`--vad-endpointing SECONDS` sends the final result after that much silence
following confirmed speech, without waiting for `audio-stop`. The connection then
closes; no second final result is sent if an audio-stop was already in flight.
Only endpointing advertises `requires_external_vad=false`. A client's
`vad_sensitivity=relaxed` multiplies the endpoint by 1.5, `aggressive` by 0.5.
Silence or short noise without confirmed speech does not trigger an endpoint;
at audio-stop it returns an empty transcript without calling Whisper. The global
`--max-audio-seconds` limit still applies, including to silence.

| Option | Default |
| --- | --- |
| `--vad-filter` | Disabled |
| `--vad-endpointing` | Disabled; seconds of trailing silence |
| `--vad-threshold` | 0.5, strictly between 0 and 1 |
| `--vad-min-speech-ms` | 250 consecutive milliseconds to confirm speech |
| `--vad-min-silence-ms` | 500 to split speech regions for filtering |
| `--vad-pad-ms` | 200 of context around filtered regions |

Filtering and endpointing can be enabled independently. Endpoint-only mode keeps
interior audio intact but suppresses all-silence utterances. Padding prevents
cutting off consonants; use more padding or disable filtering if recognition
quality suffers. Confirmed commands shorter than `--vad-min-speech-ms` are discarded,
so lower that value if very short responses matter.

## Home Assistant names

The `hass` extra fetches **conversation-exposed** entity names and aliases, plus
areas/floors, using Home Assistant's read-only websocket API. The service never
calls HA actions/services. Used areas and floors have highest priority, followed
by commonly spoken entity domains and other names. Disabled/unexposed entities
are excluded. Friendly names come from states; aliases and device/area mappings
come from the registries.

Use a long-lived HA access token with permission to read those registries. For
Docker, put it in `secrets/hass_token` (gitignored), readable by container UID
1000, and enable the provided secret overlay:

```sh
HASS_API=http://192.168.1.10:8123/api \
  docker compose -f compose.yaml -f compose.hass.yaml up --build -d
```

For a local installation:

```sh
WYO_WHISPER_HASS_TOKEN_FILE=/path/to/hass_token \
  script/run --model base-q8_0 --language ru --data-dir ./data \
  --uri tcp://0.0.0.0:10300 --hass-api http://192.168.1.10:8123/api
```

Use HTTPS outside a trusted network: an HTTP websocket sends the token without
transport encryption. The secret is not logged; server errors do not echo HA's
raw authentication messages. Transcripts themselves are logged at INFO as before.
The API URL must end in `/api` (reverse-proxy path prefixes are supported).

| Option | Default |
| --- | --- |
| `--hass-token` | Disabled; prefer `WYO_WHISPER_HASS_TOKEN_FILE` |
| `--hass-api` | `http://homeassistant.local:8123/api` |
| `--hass-api-timeout` | 10 seconds for the entire websocket exchange |
| `--hass-refresh-seconds` | 0: refresh once per utterance; positive values throttle |
| `--hass-prompt-timeout` | Wait at most 1 second for names at batch decode |
| `--hass-prompt-max-tokens` | 200; maximum allowed 223 |

A shared cache coalesces simultaneous refreshes. Refresh starts at service startup
and when audio arrives. Unreachable HA, permission errors, or timeouts retain the
last good snapshot (or `--initial-prompt` alone). Partial decoding never waits on
network I/O. Its first prompt snapshot remains fixed through the final decode;
names arriving later are available to subsequent requests. Batch mode can wait
up to the configured prompt timeout. Outstanding refreshes are cancelled on shutdown.

The configured `--initial-prompt` is kept ahead of fetched names. Because
pywhispercpp does not provide a safe high-level tokenizer API, this implementation
budgets with a conservative UTF-8 byte upper bound on BPE tokens, including for
Cyrillic. This can fit fewer names than exact token counting but prevents overflowing
the decoder prompt window. An overlong prefix is also trimmed in HA mode.
Older HA versions lacking the exposed-entity or registry commands fall back to
the previous snapshot/prefix rather than exposing all entities or failing ASR.

## Streaming

Enable `--streaming` and optionally `--streaming-interval 1.0` (minimum newly
received audio seconds between partial decodes). With VAD enabled, previews are
suppressed until speech is confirmed. Capabilities are advertised through
`supports_transcript_streaming=true`.

Wyoming 1.11 defines this event sequence:

1. `transcript-start`
2. Zero or more `transcript-chunk` events during audio reception, plus the final
   hypothesis if it changed. Each chunk **replaces** the preceding text; never
   concatenate chunk text.
3. One final `transcript`, also required for backwards compatibility.
4. `transcript-stop`, then disconnect.

There is no per-request streaming negotiation in this Wyoming version: the
server flag enables the additional events for every connection. Ordinary clients
must ignore unknown events and consume the final transcript. Without the flag,
only the usual final transcript is sent.

Whisper is not a native online recognizer. This implementation re-decodes the
accumulated utterance, emitting revisable hypotheses while audio continues to
arrive. It does not splice independent windows (which can repeat/drop words).
There is at most one partial decode in flight per client; slow decoders skip
intermediate snapshots and use the newest audio rather than growing a job queue.
Final decoding uses the complete utterance, or its VAD-filtered speech regions.
This costs extra compute, especially on long utterances and with concurrent clients;
increase the interval or disable streaming on slower machines. An in-flight
partial must finish before the final decode can acquire the model. A partial
inference failure falls back to the final decode; it is logged, not sent as a
fake final transcript.

## Protocol and concurrency

`describe` returns `info`. A request consists of optional `transcribe`, optional
`audio-start`, PCM `audio-chunk` events, then `audio-stop`. Audio is converted to
16 kHz / signed 16-bit / mono; mono/stereo PCM widths 1-4 are accepted. Exactly
one final `transcript` is returned, followed by disconnect (with a final
`transcript-stop` in streaming mode). Empty audio returns an empty transcript without inference. Invalid
frames, excessive duration, or inference errors produce a Wyoming `error`.

Each client owns its audio buffer and temporary WAV. Inference runs in a worker
thread, and a native-context lock serializes concurrent requests, including
when an asyncio caller is cancelled. This avoids mixing clients or blocking
protocol handling. Shutdown can wait for current native inference to complete.
There is no global client/queue limit: use network-level controls on busy or
untrusted deployments. VAD and partial results are isolated per client as well;
disconnect waits for in-flight native work before releasing its temporary WAV.

## Development

```sh
.venv/bin/pip install '.[dev]'
script/test
script/lint
.venv/bin/python -m build
# Optional real inference test; the normal suite does not download models:
WHISPER_CPP_TEST_MODEL="$PWD/data/ggml-base.en-q8_0.bin" script/test -m integration
```

The real-model test expects the fixture's words, and was verified with
`base.en-q8_0`. `tiny.en-q8_0` runs successfully but misrecognizes "lamp" as
"Liam" on this fixture; it does not pass that accuracy assertion.

The test suite covers configuration, cache lookup/download failures, audio
conversion, empty/error paths, client isolation, cancellation, native parameters,
VAD frame boundaries/real Silero inference, streaming backpressure and event order,
and HA websocket auth/registries/deadlines with a local simulated server. `python -m wyoming_whisper_cpp.health_check --uri tcp://localhost:10300`
checks a real Describe/Info exchange.

Not implemented from current OHF upstream: multi-engine/language routing,
native online-model decoding, and zeroconf. `--compute-type` and `--stt-library` are intentionally unsupported;
quantization belongs to the GGML model and acceleration to the native build.

## Attribution

MIT licensed. `env_args.py`, `hass_api.py`, `name_cache.py`, `vocabulary.py`, and
the HA/vocabulary tests are adapted from OHF-Voice/wyoming-faster-whisper
(Michael Hansen); its license is retained in `LICENSE.md`. Architecture and
behavior were compared with rhasspy/wyoming-whisper-cpp and
Laurent Debacker's wyoming-whisper.cpp. The test WAV originates from the latter
(MIT; see `NOTICE.md`). whisper.cpp and pywhispercpp remain separate dependencies
with their own licenses.
