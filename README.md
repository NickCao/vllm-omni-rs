# vllm-omni-rs

A pure-Rust, ZMQ-native HTTP frontend for [vllm-omni](https://github.com/vllm-project/vllm-omni)'s
Qwen3-TTS model. Rust replaces the Python `Orchestrator` entirely: it spawns
vllm-omni's stage engines in `--headless` mode and talks to their ZMQ engine
cores directly over the same msgpack wire protocol vLLM's own Rust frontend
uses. There is zero Python in the request path -- Python only runs inside the
headless stage subprocesses to do model inference.

```
speech-01234... HTTP request
      |
      v
+---------------------------+        ZMQ (msgpack)        +--------------------------+
|      vllm-omni-rs          |<-------------------------->| headless stage 0 (talker)|
|  axum HTTP server          |                             +--------------------------+
|  OmniMasterServer (ZMQ)     |        ZMQ (msgpack)        +--------------------------+
|  EngineCoreClient x2        |<-------------------------->| headless stage 1(code2wav)|
|  WAV/PCM encoder            |                             +--------------------------+
+---------------------------+                                        |
                                                              /dev/shm (SharedMemoryConnector,
                                                               async_chunk, Python-side only)
```

## Why

vllm-omni's normal `vllm serve --omni` path runs an in-process Python
`Orchestrator` that owns every stage and drives requests through them one
prompt at a time. That's a lot of Python sitting directly in the request
path. Upstream vLLM already ships a
[Rust HTTP frontend](https://github.com/vllm-project/vllm/tree/main/rust)
that talks to a single `EngineCoreProc` over ZMQ; this project extends that
idea to a multi-stage omni pipeline: Rust becomes the orchestrator, each
stage runs headless (inference only, no orchestration logic), and Rust
drives them the same way upstream drives a single engine core.

The one thing this does *not* reimplement is the codec-token-to-audio
bridge between stages. vllm-omni's Python workers already do that via
`async_chunk` mode (`SharedMemoryConnector` over `/dev/shm`); Rust just
turns that mode on and lets it run. There is no non-async-chunk fallback
and no Rust-side reimplementation of that bridge -- see
[Required vllm-omni patch](#required-vllm-omni-patch) below for the one
thing that has to change to make headless mode support it.

## Architecture

### Components

| File | Role |
|---|---|
| `src/main.rs` | CLI parsing, startup sequencing, graceful shutdown |
| `src/introspect.rs` | One-time startup calls into Python: tokenizer extraction, per-stage sampling defaults |
| `src/stages.rs` | Spawns headless Python stage processes, kills their process groups on shutdown |
| `src/master.rs` | `OmniMasterServer`: accepts stage registrations over ZMQ, allocates ports, drives the vLLM engine-core handshake for each stage |
| `src/routing.rs` | `TtsRouter`: submits a request to both stages concurrently, collects and concatenates the audio |
| `src/server.rs` | axum `Router` wiring |
| `src/routes/speech.rs` | `POST /v1/audio/speech`: builds the wire request, decodes the wire response |
| `src/routes/models.rs`, `src/routes/health.rs` | `GET /v1/models`, `GET /health` |
| `vendor/engine-core-client` | vLLM's Rust ZMQ engine-core client, vendored from vLLM tag `v0.25.0`, extended with omni-specific wire fields (see below) |
| `vendor/metrics` | Metrics types pulled in transitively by the vendored client |
| `patches/` | Patches required against vllm-omni itself (not this repo) |

### Startup sequence

1. `main.rs` allocates a port for the master's own ZMQ registration socket
   (`vllm_managed_engine::allocate_handshake_port`).
2. `stages.rs` spawns two headless Python processes:
   ```
   python3 -m vllm_omni.entrypoints.cli.main serve <model> --omni --headless \
     --stage-id 0 --omni-master-address <host> --omni-master-port <port> --async-chunk
   python3 -m vllm_omni.entrypoints.cli.main serve <model> --omni --headless \
     --stage-id 1 --omni-master-address <host> --omni-master-port <port> --async-chunk
   ```
   Stage 0 is the Talker (autoregressive codec generation), stage 1 is
   Code2Wav (the vocoder). Each process is put in its own process group so
   shutdown can `SIGTERM` the whole group instead of leaking children.
3. `master.rs` runs a ZMQ `ROUTER` socket (`run_registration`) and waits for
   each headless stage to send a msgpack `{stage_id, replica_id}`
   registration. For each one it allocates three fresh ports (handshake,
   input, output) and replies with their addresses -- this is the same
   registration contract vllm-omni's own `OmniMasterServer` implements, just
   reimplemented in Rust.
4. Once both stages have registered, `master.rs` runs vLLM's standard
   engine-core HELLO/INIT/READY handshake (`EngineCoreClient::connect` in
   `HandshakeOwner` mode) against each stage's handshake address. This is
   the vendored, unmodified upstream client -- vllm-omni-rs doesn't
   reimplement the handshake itself.
5. `routing.rs` builds a `TtsRouter` from the two connected clients and
   loads a tokenizer (see [Tokenizer](#tokenizer) below).
6. The axum HTTP server starts.

### Request lifecycle (`POST /v1/audio/speech`)

1. Build `additional_information` as vllm-omni's
   `AdditionalInformationPayload` wire struct: a msgpack map
   `{"entries": {"text": {...}, "task_type": {...}, "language": {...},
   "speaker": {...}, "instruct"?: {...}}}`, each entry carrying its string
   value in `list_data`.
2. Estimate the talker's prompt length by reimplementing
   `Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information`
   in Rust, using a native tokenizer (see [Tokenizer](#tokenizer)) instead
   of round-tripping into Python. Build a placeholder prompt of that many
   token ids.
3. Submit **every stage in the pipeline concurrently**, matching what
   vllm-omni's Python orchestrator does in `_prewarm_async_chunk_stages`
   for N stages, not just 2:
   - **Root stages** (no upstream source, e.g. the talker) get the real
     placeholder prompt and `additional_information`.
   - **Downstream stages** (fed by an earlier stage via the connector,
     e.g. code2wav) get a 1-token placeholder prompt, the *same*
     `additional_information`, and `resumable=true` (marks the request as
     a streaming input the connector will keep extending).
   - All requests for one HTTP call share the same
     `request_id`/`external_req_id` -- that's what lets the connector
     match each stage's output to the next stage's input.
   - Which stage is root vs downstream isn't hardcoded: `TtsRouter` reads
     each stage's `engine_input_source` from the introspected pipeline
     topology (empty = root). Every async_chunk pipeline checked so far is
     a linear chain (at most one upstream source per stage) -- Qwen3-TTS's
     2-stage talker/code2wav, and also Qwen3-Omni's 3-stage
     thinker -> talker -> code2wav.
   - Per-stage sampling params (`temperature`, `top_p`, `top_k`,
     `repetition_penalty`, `max_tokens`, `min_tokens`, `stop_token_ids`,
     `detokenize`) are *not* hardcoded either. `introspect.rs` shells out
     to Python once at startup and calls the exact same
     `load_and_resolve_stage_configs` function `run_headless()` itself uses,
     which merges the deploy YAML's `default_sampling_params` with the
     pipeline topology's `sampling_constraints` (e.g. the talker's
     `stop_token_ids: [2150]`). `TtsRouter` builds each stage's
     `EngineCoreSamplingParams` from that at startup; the only thing it
     adds on top is `output_kind` (`DELTA` for the audio-output stage,
     `FINAL_ONLY` for every other stage), since that's about how *this
     frontend* consumes the stream, not a model property.
4. From here on, **Rust does not move any codec data**. Each headless
   stage's own worker process runs vllm-omni's `OmniChunkTransferAdapter` /
   `SharedMemoryConnector`, which writes a stage's codec chunks to
   `/dev/shm` and the next stage's scheduler polls them out as they arrive.
   This is exactly the same mechanism the orchestrated Python path uses.
5. Rust drains every non-audio stage's output stream to completion, in
   stage-id order (only used for token-count logging).
6. Rust drains the audio stage's output stream last, collecting every
   `multimodal_output` chunk. It uses `output_kind=DELTA`, so **each
   chunk is a new slice of audio, not the cumulative buffer** -- these are
   concatenated in order, mirroring the Python frontend's `torch.cat` over
   all streamed audio deltas. (Overwriting instead of accumulating here
   was the root cause of a truncated-audio bug during development --
   only the last delta chunk survived.)
7. Each chunk's PCM tensor is decoded into `f32` samples, converted to
   16-bit PCM, and encoded as WAV (or returned raw for
   `response_format: "pcm"`).

### Wire protocol notes

- Everything on the wire is msgpack, matching vLLM's `EngineCoreRequest` /
  `EngineCoreOutput` structs (`vendor/engine-core-client/src/protocol/`).
  `additional_information` and `multimodal_output` are untyped
  `OpaqueValue` (`= rmpv::Value`) fields -- vllm-omni decides their shape,
  not vLLM core.
- `rmpv::ext`'s serializer encodes plain `#[derive(Serialize)]` structs as
  msgpack **arrays**, matching vLLM's `array_like=True` convention used for
  most core structs. `AdditionalInformationPayload` is a plain Python
  dict, though, so it must round-trip as a msgpack **map**. `routes/speech.rs`
  builds it through `serde_json::Value`/`json!` instead of a struct, since
  `Value`'s `Serialize` impl dispatches through `serialize_map` for objects.
  (Deserializing is unaffected by this -- `rmpv::ext`'s deserializer is
  fully value-driven, so decoding `multimodal_output` into typed structs
  works normally.)
- `EngineCoreSamplingParams` is a plain msgpack **map** on the wire despite
  looking array-like elsewhere in vLLM -- Python's `SamplingParams` actually
  serializes via `msgspec` field names, not position.
- Tensors use vLLM's `(dtype, shape, data)` wire tuple
  (`vendor/engine-core-client/src/protocol/tensor.rs::WireTensor`). `data`
  can be a msgpack extension type (inline bytes) or an integer aux-frame
  index; `EngineCoreClient` resolves aux-frame references to inline
  `Binary` before handing output to callers, so `WireTensor`'s deserializer
  accepts both `Ext` and `Binary`.

### Tokenizer

Qwen3-TTS's prompt-length math needs a tokenizer, but the HF repo doesn't
ship a `tokenizer.json` (it builds one from `vocab.json` + `merges.txt` at
load time). `--tokenizer-path` lets you point at one directly; if omitted,
`main.rs` shells out to `python3 -c '...AutoTokenizer...save(...)'` once to
extract and cache one under `/tmp`. This is the only Python that ever runs
outside a headless stage subprocess, and only at startup, never per-request.

### Required vllm-omni patch

`run_headless()` in `vllm_omni/entrypoints/cli/serve.py` hardcodes
`async_chunk=False` when building the stage connector spec -- headless mode
was never wired up to respect the (pre-existing) `--async-chunk` flag the
same way the orchestrated path does. Without this patch, the two stages
never exchange codec data and stage 1 produces empty or garbage audio.

Apply it from your `vllm-omni` checkout:

```bash
git apply /path/to/vllm-omni-rs/patches/0001-headless-respect-async-chunk.patch
```

It's a 4-line change: `run_headless` was calling
`get_stage_connector_spec(..., async_chunk=False)` unconditionally, which
makes that function return `{}` and skip building a real connector spec
entirely. The patch reads the (pre-existing) `--async-chunk` CLI flag
instead of hardcoding `False`. That's the only thing that needed fixing --
`build_engine_args_dict` already threads a non-empty connector spec through
correctly once it receives one.

## Prerequisites

- Rust (see `rust-toolchain.toml` -- currently `stable`)
- A working [vllm-omni](https://github.com/vllm-project/vllm-omni) Python
  environment (`vllm_omni` importable by `python3`), **with the patch above
  applied**
- `transformers` installed in that same environment (used for the one-time
  tokenizer extraction, unless you pass `--tokenizer-path`)
- A GPU that can run Qwen3-TTS (this has only been tested against
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
- `libzmq` available for the `zmq` crate to link against

## Building

```bash
git clone https://github.com/NickCao/vllm-omni-rs
cd vllm-omni-rs
cargo build --release
```

`vendor/engine-core-client` and `vendor/metrics` build as local path
dependencies; `vllm-managed-engine` and `vllm-tokenizer` are pulled from
vLLM's `rust/` directory at tag `v0.25.0`. If your vllm-omni checkout runs
a different vLLM version, the wire structs in `vendor/engine-core-client`
may need re-syncing -- see the comments in
`vendor/engine-core-client/src/protocol/*.rs` for what's been changed from
upstream and why.

## Running

```bash
./target/release/vllm-omni-rs Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --port 8091 \
  --tokenizer-path /path/to/tokenizer.json   # optional, auto-extracted if omitted
```

CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `<model>` (positional) | -- | Model name or path, passed through to both headless stages |
| `--host` | `0.0.0.0` | HTTP server bind host |
| `--port` | `8000` | HTTP server bind port |
| `--master-host` | `127.0.0.1` | Bind host for the ZMQ registration/handshake sockets |
| `--handshake-timeout` | `300` | Seconds to wait for each stage's engine-core handshake |
| `--tokenizer-path` | (auto-extract) | Path to a `tokenizer.json`; see [Tokenizer](#tokenizer) |
| `--stage-args` | (none) | Extra args forwarded verbatim to both headless stage processes, e.g. `--stage-args --gpu-memory-utilization 0.85` |

On startup you should see both stages register and complete their
handshake, then `Listening on <host>:<port>`.

## Testing it

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world, this is a test.", "voice": "Vivian"}' \
  -o out.wav
```

Request fields (OpenAI `/v1/audio/speech`-compatible):

| Field | Default | Notes |
|---|---|---|
| `input` | required | Text to synthesize |
| `voice` | `Vivian` | Speaker name |
| `instructions` | none | Optional style/instruction text |
| `response_format` | `wav` | `wav` or `pcm` (raw 16-bit PCM, no header) |
| `model`, `stream_format` | -- | Accepted for API compatibility, unused |
| `stream` | `false` | Must be `false` -- streaming responses aren't implemented; `true` returns `400` |

Any other top-level field (e.g. `task_type`, `language`) is passed through
into `additional_information` verbatim.

Other endpoints:

- `GET /health` -- liveness check
- `GET /v1/models` -- lists the one model this instance was started with

To sanity-check output end to end, transcribe the result and compare it
against the input text (this is how the truncation and encoding bugs
during development were actually caught -- a byte count or HTTP 200 alone
doesn't tell you the audio is *correct*):

```bash
python3 -c "
import whisper, numpy as np, wave
with wave.open('out.wav', 'rb') as wf:
    sr = wf.getframerate(); n = wf.getnframes()
    audio = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
model = whisper.load_model('base')
print(model.transcribe(audio, fp16=False)['text'])
"
```

## Known limitations

- `TtsRouter` is topology-driven, not hardcoded to 2 stages: it introspects
  stage count, roles (root vs connector-fed), and per-stage sampling
  params from vllm-omni's own `load_and_resolve_stage_configs` at startup
  (see [Request lifecycle](#request-lifecycle-post-v1audiospeech)). This
  covers every async_chunk pipeline checked so far, including 2-stage
  talker/vocoder models (CosyVoice3, Higgs-Audio v2/v3, GLM-TTS,
  Fish-Speech) and Qwen3-Omni's 3-stage thinker -> talker -> code2wav
  chain -- verified end to end only against Qwen3-TTS, though; the 3+-stage
  path is exercised by the topology data, not a live test. What's still
  Qwen3-TTS-specific and would need porting to run a different model:
  - `additional_information`'s schema (`speech.rs`) --
    `{text, task_type, language, speaker, instruct}` is
    `Qwen3TTSPromptEmbedsBuilder`'s input contract specifically, and there's
    no config to introspect for this, only procedural Python to reimplement.
  - `TtsRouter::estimate_prompt_len` (`routing.rs`) -- reimplements
    Qwen3-TTS's exact chat-template token-counting formula; other models
    compute this differently.
  - Only linear chains are supported (each stage has at most one upstream
    source) -- every pipeline checked so far is linear, but a genuinely
    branching DAG (a stage fed by more than one upstream) isn't handled.
- `stream: true` is rejected outright -- responses are always fully
  buffered before being returned.
- Requires the vllm-omni patch above; there is intentionally no fallback
  to non-`async_chunk` mode.
- The vendored client's own test suite
  (`vendor/engine-core-client/src/tests/`) currently fails to compile --
  it predates the `EngineCoreSamplingParams` map-format rewrite and hasn't
  been updated to match. `cargo build`/`cargo check` on the actual binary
  are unaffected.
