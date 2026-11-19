# Candle migration proposal (branch `candle`)

Status: **implemented and measured** — see [candle-benchmark-report.md](candle-benchmark-report.md)
for results and the final recommendation. This file is the original proposal; the corrections
below were made after the work (marked **Update**).

## 1. What the candle checkout is

`/home/meme/Documentos/GitHub/candle`, branch `feat/audio-runtime`
(fork `nglmercer/candle`, 293 commits ahead of upstream `main`).

- **Vulkan backend**: `candle-core/src/vulkan_backend`, `candle-vulkan-kernels`
  (Slang shaders compiled to SPIR-V by `build.rs`), built on a vendored
  `vulkano`/`ash` fork (~635k lines vendored). Kernel coverage relevant to us:
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K–Q8K GEMM, conv, conv_transpose1d, rms_norm, rope,
  softmax (no dedicated attention/flash shader found in the shader list).
  `perf/vulkan-experiments` holds GEMV/Q4_K/Q6_K
  experiments that are **not** on this branch.
- **`candle-audio`**: model-independent runtime. `TtsModel` trait
  (`synthesize(request, cancel, sink)`), `ModelLoader`/`ModelRegistry`,
  cancellation, event sink, PCM handling, a CLI with RTF/latency/RSS reports
  (`bench_results/audio-runtime/`). Registered families: whisper, parler_tts,
  mimi, lfm3_audio, encodec, dac, snac, csm, metavoice, musicgen, voxtral, gemma4.
- **Not present**: Qwen3-TTS, Kyutai TTS (no Moshi LM, only the `mimi` codec),
  Supertonic. Each is new work.

## 2. Feasibility per target model

| Model | What candle gives | What is missing | Verdict |
|---|---|---|---|
| **Supertonic 3** | `candle-onnx` interpreter | 3 ops (below) | Feasible, smallest effort |
| **Qwen3-TTS 1.7B** | `quantized_qwen3`, GGUF Q5_K/Q8_0 on Vulkan, conv/conv_transpose1d | Talker→code-predictor loop, 12 Hz speech-tokenizer decoder/encoder, speaker encoder | Feasible, large effort; seed exists in git history |
| **Kyutai TTS** | `mimi` codec | Delayed-streams LM + depformer, text-stream scheduling, weights (not local) | Feasible, largest effort, most uncertain |

### Supertonic (measured)
Op census of the four graphs against `candle-onnx/src/eval.rs`
(probe kept out of both repos):

| Graph | Nodes | Missing in candle-onnx |
|---|---:|---|
| duration_predictor | 1086 | `LayerNormalization` |
| text_encoder | 1934 | `LayerNormalization` |
| vector_estimator | 1004 | `LayerNormalization`, `Reciprocal`, `Softplus` |
| vocoder | 401 | `LayerNormalization` |

No `ConvTranspose`, `LSTM` or control flow. The three ops are trivial.
**Update:** the census only compared op *names*. Running the graphs also needed `Pad` modes
`constant`/`edge`, integer `Pow`, broadcast `PRelu`, and a depthwise-conv fast path in
`candle-core` (the grouped-conv path was ~30× slower than ORT). SonicBoom's
Supertonic path touches `ort` in only four `.run()` calls
(`src/tts/inference.rs:279,301,348,374`) plus session construction in
`src/tts/model.rs`, so a backend trait is a small change.
Risk: candle-onnx is an interpreter and these graphs contain many shape-plumbing
nodes (Shape/Gather/Unsqueeze/Concat/Cast, several hundred per graph, not yet
timed); per-node overhead on a 2-core CPU may
dominate. If so, the fallback is a hand-written native port of the four
networks, which the candle-audio loader model supports.

### Qwen3-TTS (unverified details)
SonicBoom already shipped a Candle Qwen3-TTS provider (0.6B) and removed it in
`98d59c8`; the code is recoverable with `git show 98d59c8^:vendor/qwen3-tts`
and is the natural seed. Its recorded CPU RTF was ~240–320 (see
`benchmarks/providers.md`), which is why the project moved to GGUF + ORT
(RTF ~2.3–2.8 on this same Vulkan GPU, `benchmarks/results/2026-10-15-qwen-gguf`).
The candle proposal only makes sense if the Vulkan backend closes that gap.
Current assets use Q5_K talker, Q8_0 predictor and int8 ONNX codec/speaker
models; the int8 ONNX parts need either candle-onnx (QLinear* ops are **not**
in its op list) or an f16/f32 candle port of the codec.

### Kyutai TTS (unverified details)
`kyutai/tts-1.6b-en_fr`, built on Mimi at 12.5 Hz.
**Update:** Kyutai's own Rust server runs this checkpoint through *Python* (`type = "Py"`), and
`moshi-core`'s Rust LM targets an older layout (no depformer weight-sharing schedule, no demuxed
text stream). Only the Mimi codec and generic transformer pieces were reusable; the LM, state
machine and generation loop were ported from the Python reference (`vendor/kyutai-tts`). Weights are not downloaded
locally; budget a multi-GB download. Streaming text input is a different
contract from `/api/tts`'s one-shot `text/plain`, so the provider API needs a
streaming-capable path to show the realtime advantage.

## 3. Proposed architecture

Do **not** delete ORT/llama.cpp first. Add a backend seam and run candle
side by side until it wins on measurements.

1. Feature `candle` (off by default) with a path/git dependency on the fork
   pinned by revision. `candle-audio` is used for the runtime types and bench
   conventions; model code lives in a new `src/tts/providers/candle/` module.
2. Providers `supertonic-candle`, `qwen3-candle`, `kyutai-candle` implementing the
   existing `TtsProvider` trait (`src/tts/providers/mod.rs`); the existing
   registry/pool/worker/queue/auth/API code is untouched.
3. Device selection gains `vulkan` (`DevicePreference`, `EffectiveDevice`);
   CPU fallback preserved.
4. Only after Phase 3 gates pass: make candle the default, then remove
   `ort`, `native/qwen-llama` and their feature flags in a separate branch.

### Phases and gates
| Phase | Work | Gate to continue |
|---|---|---|
| 0 | Install `slangc`, build fork with `--features vulkan`, run `candle-audio` CLI smoke on BC-250 | Vulkan device is selected and a known model (e.g. whisper-tiny) matches CPU output |
| 1 | Supertonic on candle-onnx (+3 ops), same wav as ORT within tolerance | RTF measured on CPU and Vulkan; compare with ORT baseline |
| 2 | Qwen3-TTS port (talker/predictor/codec) | Audible, matches GGUF output envelope; RTF vs 2.3–2.8 |
| 3 | Kyutai TTS port | RTF + time-to-first-audio |
| 4 | Decide: replace, keep side by side, or abandon | Written comparison table |

## 4. Benchmark protocol

Report per model, backend and device, with identical texts (short ≈2 s,
medium ≈10 s, long ≈30 s; EN and one non-EN):

- **RTF** = synthesis wall time / audio duration (lower is better; < 1 is
  faster than realtime). Report cold (first call) and warm (median of ≥5).
- **TTFA**: time to first audio chunk (matters for streaming models).
- Load time, peak RSS (`VmHWM`), and VRAM/GTT where readable.
- Output sanity: finite PCM, audible, spectral/envelope check, and a
  similarity comparison against the current provider's output for the same text.
- Raw JSONL stored under `benchmarks/results/<date>-candle/`, same shape as the
  existing results so they compare directly.

## 5. Risks and open items

- **Blocker**: `slangc` is not installed (`shader-slang` is in the Arch repos);
  the Vulkan kernels cannot compile without it. Vulkan is not testable until then.
- **Memory**: 14 GB total, ~5 GB free here. Qwen 1.7B plus Kyutai 1.6B cannot be
  resident together with Supertonic; benchmarks must run one model at a time.
- **Dependency weight**: the fork vendors vulkano/ash; build time on 2 cores
  will be long and the fork is a maintenance commitment vs. upstream candle.
- **Prior art**: the previous candle Qwen backend was removed for speed; the
  migration must beat a measured baseline, not an assumption.
- **Docker/Ubuntu 24.04** and ROCm/Metal targets in `CLAUDE.md`: candle covers
  Vulkan/Metal/CUDA but not ROCm directly; AMD on Linux would use Vulkan.
