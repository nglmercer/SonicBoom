# SonicBoom Qwen maintenance patch

Source: https://github.com/TrevorS/qwen3-tts-rs at
`711ceee07cad92673f86de8997bdf54c30caa49f` (the previously reviewed pin).
The upstream package declares MIT licensing in Cargo.toml. Original source
notices are retained. `upstream-source.sha256.json` records the original bytes
for review; it is provenance, not a replacement for model-file SHA-256 pins.

This vendored dependency keeps Rust/Candle inference and the upstream sampling,
EOS, acoustic prediction, tokenizer, and ICL reference-context decoding.
SonicBoom's patch adds scoped, thread-local cooperative control between native
operations, checkpoints inside generation/acoustic prediction and preparation,
stage timings, a bounded frame-vector reservation, and prompt-aware KV capacity.
With a control hook installed, max-frame termination returns an explicit error
instead of silently delivering truncated audio. Calls without a hook preserve
upstream max-frame behavior. No audio streaming API for cloned voices is claimed.

Control hooks are scoped with an unwind-safe restoration guard. Hooks execute
on the model-owning thread. A native kernel cannot be interrupted while it runs;
cancellation takes effect at the next checkpoint. Neither thread termination
nor `spawn_blocking` abort is used to stop computation.

When upgrading, compare these files against this exact upstream revision,
reapply only the maintenance changes, update provenance and the pin constant,
and run SonicBoom's tests and gated real-model cancellation benchmark.
