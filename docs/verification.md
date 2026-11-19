# Verification performed

## Vulkan default deployment

Docker now defaults to `WHISPER_CPP_VULKAN=ON` and `WYO_WHISPER_DEVICE=gpu`.
Compose passes `/dev/dri` and requires the host render-device group via
`RENDER_GID`. Local pip/CLI defaults remain CPU.

The default Vulkan image was built successfully as `wyoming-whisper-cpp:vulkan`.
The source build required SPIRV-Headers and static whisper.cpp/GGML libraries
with position-independent code; this avoids missing `libwhisper.so` at runtime.
Native imports and real recognition of the speech fixture passed. With no
`/dev/dri` available here, Vulkan reported no devices and recognition used the
CPU fallback. Hardware GPU offload remains untested. Compose validation passed
with an explicitly supplied render GID. CI now checks native imports, not just
`--help`, to catch missing runtime libraries.

## Version 0.2.0: VAD, HA vocabulary, streaming

Local environment: Linux x86-64, Docker Python 3.11, pywhispercpp 1.5.1,
Wyoming 1.11.0, pysilero-vad 2.1.1, ONNX Runtime 1.32.0, aiohttp 3.14.5.

- Full suite with `WHISPER_CPP_TEST_MODEL=/work/data/ggml-base.en-q8_0.bin`:
  **120 passed**. Without the GGML model the three native integration cases skip;
  the normal suite still runs real Silero inference using its bundled ONNX model.
- Real TCP streaming: client sends only the first two seconds of the WAV, waits
  for TranscriptStart/TranscriptChunk, and then continues sending audio. This
  verifies partial results arrive before audio-stop, not merely at final decode.
- Real VAD endpoint: client sends the remaining audio plus silence without
  audio-stop; server returns the correct final transcript followed by
  TranscriptStop and disconnects. The plain batch TCP path and SIGTERM test also pass.
- VAD unit tests cover arbitrary chunk boundaries, incomplete final windows,
  short-noise rejection, silence, padding/overlap merging, and endpoints occurring
  inside a large chunk (the tail is not transcribed).
- Streaming tests cover replacement/correction semantics, one final transcript,
  reader responsiveness during slow decoding, coalescing of queued audio, cleanup,
  partial-failure fallback, frozen prompt snapshots, and silence suppression.
- HA is tested against a local simulated websocket server, including auth,
  exposed entities, aliases, device/area/floor relationships, disabled entities,
  malformed response IDs, hung exchanges and sanitized error messages.
- Cache tests cover refresh coalescing/throttling, stale-data fallback, bounded
  waiting without cancelling the shared refresh, shutdown cleanup, and conservative
  UTF-8/BPE budgeting including Cyrillic and overlong prefixes.
- Ruff checks/formatting and wheel/sdist builds pass. Docker Compose configuration
  validates with the optional HA secret overlay.
- CPU Docker image rebuilt as `wyoming-whisper-cpp:local` (0.2.0), with both extras
  installed. Its non-root process ran real Silero endpoint detection and native
  inference: "Turn on the living room lamp."

Not tested: a live user Home Assistant instance (no credentials were supplied),
GPU acceleration, multilingual accuracy/latency benchmarks, or the full CI Python
matrix. Streaming uses repeated batch decoding, not a native online model, and
increases computation. The only test warning is Wyoming's Python 3.11 audioop
deprecation warning.

## Version 0.1.0 baseline

Local environment: Linux x86-64, Docker `python:3.11-slim`, Python 3.11,
pywhispercpp 1.5.1, Wyoming 1.11.0. Host Python 3.14 lacked ensurepip, so
validation used a container rather than changing the host Python installation.

- Ruff checks and formatting: passed.
- Full pytest suite with `WHISPER_CPP_TEST_MODEL` pointing to
  `ggml-base.en-q8_0.bin`: **31 passed**. The only warning is Python 3.11's
  deprecation warning for Wyoming's audioop dependency.
- Without a provisioned model, the two integration tests are intentionally
  skipped; unit tests do not download weights.
- Real native CPU inference on the bundled speech fixture: passed with
  `base.en-q8_0`, including the words "living room" and "lamp".
- TCP subprocess test: CLI startup, offline model loading, Describe/Info,
  chunked WAV transcription, connection close, another healthcheck, and
  graceful SIGTERM exit all passed.
- Python wheel and sdist builds: passed.
- Docker CPU image built as `wyoming-whisper-cpp:local`; its non-root server
  loaded a read-only bind-mounted model and answered Describe/Info.
- Docker Compose configuration validation: passed.

Observed limitations:

- `tiny.en-q8_0` produced "Turn on the living room, Liam." for the same fixture,
  so its accuracy assertion failed. This is recorded rather than weakening the
  test. The service/inference path itself ran successfully.
- Downloaded model files inherit private temporary-file permissions. A model
  downloaded as root in a test container needed ownership/read permissions fixed
  before the non-root image could open it. The transcriber now checks readability
  and the GGML magic before invoking the native loader; use correct ownership on
  bind mounts. Downloads made by the normal non-root service remain readable by it.
- Vulkan source-build recipe and GPU inference were not tested. No CPU/GPU
  performance comparison or Russian accuracy benchmark was performed.
- The GitHub Actions Python-version matrix is provided, but only Python 3.11
  was executed locally.
