# Source comparison and design decisions

Inspected Git HEAD snapshots (not release-version assumptions):

| Repository | Commit |
| --- | --- |
| albe873/wyoming-whisper-cpp | `4dcb7903e59e4b8567268ce64988b5c30bc9e623` |
| rhasspy/wyoming-whisper-cpp | `476b0e631392034a94196eb578b3d0a60164af53` |
| debackerl/wyoming-whisper.cpp | `0959e732897a0de9f7e1e6a73f1cd3d28f71bae2` |
| OHF-Voice/wyoming-faster-whisper | `5f407397a5d6976ee6487c4cd900d59171b9d82e` |

These are static implementation comparisons, not latency/accuracy benchmarks.
GPU suitability depends on hardware, driver, model and native build.

## Comparison

| Area | rhasspy | albe873 | debackerl | OHF upstream |
| --- | --- | --- | --- | --- |
| Inference interface | Modified bundled whisper.cpp `main`, JSON + WAV over stdin | Custom `simple-server`, float PCM + NaN delimiter | In-process pywhispercpp | Transcriber abstraction; faster-whisper and other engines |
| Model lifetime | Persistent subprocess | Persistent subprocess | Persistent Model | ModelLoader/cache and engine dispatch |
| Audio | Wyoming converter to 16k mono WAV | Wyoming converter, numpy float PCM | numpy and soxr, buffered until stop | Wyoming converter, temporary WAV or streaming session |
| Concurrency | Async subprocess lock | Shared process; supplied lock is not used by handler | Async lock, synchronous inference inside event loop | `asyncio.to_thread` inference |
| Streaming | Batch final transcript | Incremental text assembled from stdout | Batch | Batch or engine-specific streaming |
| Model location | Repeatable data-dir lookup, download helper | Similar model/download conventions | Model loaded from download-dir; data-dir search not actually implemented | Model loader, cache and offline mode |
| GPU deployment | Custom native compilation | Vulkan/Mesa and ROCm Dockerfiles | Binding build determines available accelerators | CTranslate2 CPU/CUDA, additional engine support |
| CLI/API scope | Small Wyoming ASR service | GPU/stream-oriented customization | Closest small Python binding alternative | Env/secrets, language dispatch, VAD, HA vocabulary, healthcheck, discovery |

## Specific risks found in the snapshots

- **albe873 `handler.py`:** different clients write to the same stdin without
  using `model_proc_lock`; each can create its own stdout reader. `Transcribe`
  does not update the inference language. A reader task is started only after
  events that fall through the early returns. Multiple `Transcript` events are
  sent as cumulative partial text rather than a clear batch-final contract.
- **albe873 `simple-server.cpp`:** `markerFound` is initially uninitialized;
  reads assume float-aligned chunks and only look for NaN at the end of the read.
  These are reasons not to reuse this framing protocol. GPU build ideas remain
  useful. No explicit license file was found, so no code was copied.
- **rhasspy `handler.py`:** EOF is treated like an empty output line, so a crashed
  child can leave the loop waiting for `<|endoftext|>` indefinitely. Repeated
  immutable `bytes` concatenation increases copying. Its custom `main` protocol
  is not the stock current whisper.cpp CLI/server API.
- **debackerl `handler.py`:** `Model.transcribe()` runs synchronously inside the
  asyncio handler; an `asyncio.Lock` does not prevent blocking other clients.
  Widths other than two bytes are interpreted as int8, and empty stop asserts.
- **debackerl `__main__.py`:** `params_sampling_strategy` receives beam size,
  although it selects greedy vs beam, not the beam width. `--beam-size 0` is
  documented as greedy but rewritten to automatic 1/5. Repeated data-dir is
  exposed but not searched. Discovery advertises all languages for `.en` models.

## Implemented synthesis

1. Use debackerl's binding approach instead of maintaining a whisper.cpp fork
   or a private subprocess wire protocol. Require pywhispercpp >=1.5.1 for its
   explicit context-parameter API; the Docker image pins 1.5.1.
2. Follow current OHF's separation into `dispatch_handler.py`, a synchronous
   `WhisperCppTranscriber.transcribe(wav_path, language, beam_size, initial_prompt)`,
   model resolution, and a small CLI. Keep the model resident and use worker
   threads. A threading lock protects the native context even after async
   cancellation; the awaiting handler preserves the WAV until the worker finishes.
3. Preserve rhasspy's explicit search across data directories and English-only
   discovery, with atomic downloads and offline mode. Query multilingual status
   from the loaded native model rather than trusting a file's name.
4. Adapt OHF environment/secrets parsing with CLI-over-environment precedence,
   including ignoring an invalid environment value when the CLI overrides it.
   Preserve the core option names and beam-size auto convention.
5. Keep GPU acceleration a build concern, inspired by albe873: an optional Vulkan
   Docker build, not an extra custom inference server. No unverified speed claims.

## Deliberate boundaries

This is **upstream-style**, not a full source-level fork of current OHF. The
current upstream is already a multi-engine server with substantial unrelated
routing, VAD, vocabulary and streaming features. Copying all of that would add
large dependencies and obscure the single whisper.cpp backend.

The service does not claim compatibility for CTranslate2 compute types,
Hugging Face faster-whisper model directories, engine routing or mDNS discovery.
The native library may fall back to CPU even when GPU is requested. CPU correctness
is tested separately from hardware-specific acceleration.

## Features added in 0.2

- Incremental Silero VAD v5 scoring on normalized audio, speech-region filtering
  and optional silence endpointing. Unlike the inspected upstream endpointing
  adapter, this uses the actual pysilero-vad 2.x `process_array` API. Client VAD
  state and speech spans are independent; empty speech bypasses Whisper.
- HA websocket client, background name cache, and priority-based prompt building
  adapted directly from the inspected OHF snapshot (MIT). Changes include an
  overall exchange deadline, response-ID validation, bounded websocket messages,
  sanitized errors, shutdown cleanup, and a conservative multilingual token budget.
- Wyoming transcript streaming via full-utterance re-decoding while receiving
  audio. Chunks are revisable replacements, with a final Transcript and
  TranscriptStop as required by Wyoming 1.11. This is not a native online model
  like upstream's sherpa backend. One in-flight partial per client coalesces
  intermediate input; it shares the serialized native context with final decoding.

All three features are opt-in CLI options. They are described in README.md,
including costs, fallback behavior and compatibility boundaries.
