import { computed, getCurrentScope, onScopeDispose, ref, shallowRef, watch } from 'vue';
import { ApiError } from '../services/http';
import { monitorJob } from '../services/jobs';
import { t, msg, uiError, errorMessage, type Message } from '../i18n';
import type { Job, NormalizationMode, QualityReport, ReadingStyle } from '../services/contracts';
import type { Studio } from './useStudio';
/**
 * Everything a generation needs, frozen at submission. `normalization` and `locale`
 * are absent for servers without pronunciation support; retry, regenerate and resume
 * reuse this snapshot, never the live form.
 */
export type GenerationInput = Readonly<{
  text: string;
  model: string;
  voice: string;
  language: string;
  cloned: boolean;
  /** Set for named-voice engines other than Supertonic (Kyutai TTS). */
  provider?: string;
  normalization?: NormalizationMode;
  locale?: string;
  reading_style?: ReadingStyle;
}>;
type ActiveGeneration = Readonly<{ jobId: string; input: GenerationInput; requestEpoch: number; preset: string }>;
/** One completed generation, retained for side-by-side comparison. */
export interface GenerationResult {
  jobId: string;
  url: string;
  filename: string;
  durationSecs: number;
  text: string;
  language: string;
  preset: string;
  seed?: string;
  quality?: QualityReport;
  model: string;
  voice: string;
  /** The generated audio; the waveform player renders it directly. */
  blob: Blob;
}
/** How many alternatives are kept before the oldest URL is released. */
export const MAX_RESULTS = 4;
export function useSynthesis(studio: Studio) {
  const pending = ref(false);
  const cancelling = ref(false);
  const active = shallowRef<ActiveGeneration | null>(null);
  const jobId = computed(() => active.value?.jobId ?? null);
  const snapshot = shallowRef<Job | null>(null);
  const results = shallowRef<GenerationResult[]>([]);
  const audioUrl = ref('');
  const audioFilename = ref('');
  const audioBlob = ref<Blob | null>(null);
  const error = shallowRef<Message | null>(null);
  const complete = ref(false);
  let lastInput: GenerationInput | null = null;
  let control: AbortController | null = null;
  let disposed = false;
  let epoch = 0;
  const canGenerate = computed(() => studio.selectedModel.value.ready && !!studio.voice.value);
  /** Sampling overrides apply only to the Qwen3-TTS provider. */
  const samplingFor = (model: string) =>
    studio.selectedModel.value.provider === 'qwen3-gguf' && studio.selectedModel.value.id === model
      ? studio.samplingParams.value
      : undefined;
  function release(result: GenerationResult) {
    URL.revokeObjectURL(result.url);
  }
  function clearAudio() {
    for (const result of results.value) release(result);
    results.value = [];
    audioUrl.value = '';
    audioFilename.value = '';
    audioBlob.value = null;
  }
  // Detach the current-player pointer without discarding the comparison list:
  // a new attempt must never present stale audio as its own result.
  function resetCurrent() {
    audioUrl.value = '';
    audioFilename.value = '';
    audioBlob.value = null;
  }
  function publish(result: GenerationResult) {
    const next = [result, ...results.value].slice(0, MAX_RESULTS);
    const evicted = results.value.filter((previous) => !next.includes(previous));
    results.value = next;
    audioUrl.value = result.url;
    audioFilename.value = result.filename;
    audioBlob.value = result.blob;
    for (const previous of evicted) release(previous);
  }
  function removeResult(index: number) {
    const target = results.value[index];
    if (!target) return;
    release(target);
    const next = results.value.filter((_, i) => i !== index);
    results.value = next;
    // Never leave the player pointing at a released object URL.
    audioUrl.value = next[0]?.url ?? '';
    audioFilename.value = next[0]?.filename ?? '';
    audioBlob.value = next[0]?.blob ?? null;
  }
  async function monitor(generation: ActiveGeneration, signal: AbortSignal) {
    const current = () => !disposed && generation.requestEpoch === epoch;
    try {
      const result = await monitorJob(
        studio.api,
        generation.jobId,
        (job) => {
          if (current()) snapshot.value = job;
        },
        signal,
      );
      if (!current()) return;
      if (result.stage !== 'completed') {
        active.value = null;
        throw uiError(
          result.error ||
            (result.stage === 'failed' ? 'Inference failed.' : 'Generation cancelled or its deadline expired.'),
        );
      }
      const audio = await studio.api.jobAudio(generation.jobId, signal);
      if (!current()) return;
      publish({
        jobId: generation.jobId,
        blob: audio,
        url: URL.createObjectURL(audio),
        filename: `sonicboom-${generation.jobId}.wav`,
        durationSecs: result.audio_secs ?? 0,
        text: generation.input.text,
        language: generation.input.language,
        preset: generation.preset,
        seed: result.context?.seed,
        quality: result.quality,
        model: generation.input.model,
        voice: generation.input.voice,
      });
      active.value = null;
      complete.value = true;
    } catch (failure) {
      if (!current()) return;
      if (failure instanceof ApiError && failure.status === 404) active.value = null;
      error.value = errorMessage(failure);
    } finally {
      if (current()) {
        pending.value = false;
        cancelling.value = false;
      }
    }
  }
  async function run(input: GenerationInput) {
    const limits = studio.info.value?.synthesis;
    if (
      limits &&
      ([...input.text].length > limits.max_text_chars ||
        new TextEncoder().encode(input.text).length > limits.max_body_bytes)
    ) {
      error.value = msg('Text exceeds the server limit of {chars} characters or {bytes} bytes.', {
        chars: limits.max_text_chars,
        bytes: limits.max_body_bytes,
      });
      return;
    }
    if (disposed || pending.value) return;
    // An unresolved job is resumed only by the explicit resume action.
    if (active.value) {
      error.value = msg('Use Resume monitoring or Cancel generation.');
      return;
    }
    if (!input.text) {
      error.value = msg('Please enter some text.');
      return;
    }
    if (studio.info.value?.auth.required && !studio.token.value.trim()) {
      error.value = msg('Please enter your API token.');
      return;
    }
    if (!canGenerate.value) {
      error.value = msg('Create a voice profile for this model first.');
      return;
    }
    const requestEpoch = ++epoch;
    error.value = null;
    snapshot.value = null;
    complete.value = false;
    resetCurrent();
    pending.value = true;
    control = new AbortController();
    const signal = control.signal;
    try {
      const preset = studio.sampling.value.preset;
      const job = await studio.api.createJob(
        input.text,
        input.language,
        input.voice,
        input.cloned,
        samplingFor(input.model),
        signal,
        { normalization: input.normalization, locale: input.locale, reading_style: input.reading_style },
        input.provider,
      );
      if (disposed || requestEpoch !== epoch) return;
      const generation = Object.freeze({ jobId: job.id, input, requestEpoch, preset });
      active.value = generation;
      await monitor(generation, signal);
    } catch (failure) {
      if (!disposed && requestEpoch === epoch) {
        error.value = errorMessage(failure);
        pending.value = false;
      }
    }
  }
  async function generate(preview = false) {
    const input: GenerationInput = Object.freeze({
      text: preview ? t('Hi there. This is a voice preview.') : studio.text.value.trim(),
      model: studio.selectedModel.value.id,
      voice: studio.voice.value,
      language: studio.language.value,
      cloned: studio.selectedModel.value.cloned,
      provider: studio.selectedModel.value.namedVoices?.length ? studio.selectedModel.value.provider : undefined,
      normalization: studio.pronunciation?.value?.normalization,
      locale: studio.pronunciation?.value?.locale,
      reading_style: studio.pronunciation?.value?.reading_style,
    });
    lastInput = input;
    await run(input);
  }
  /** Re-run the most recent request; the earlier result stays for comparison. */
  async function regenerate() {
    if (!lastInput) return;
    await run(lastInput);
  }
  async function resume() {
    if (disposed || pending.value || !active.value) return;
    error.value = null;
    pending.value = true;
    control = new AbortController();
    await monitor(active.value, control.signal);
  }
  async function cancel() {
    const generation = active.value;
    if (!generation || cancelling.value || disposed) return;
    cancelling.value = true;
    try {
      await studio.api.cancelJob(generation.jobId);
      if (disposed || generation.requestEpoch !== epoch || active.value?.jobId !== generation.jobId) return;
      // The acknowledgement is not terminal. Reconcile server state if disconnected.
      if (!pending.value) await resume();
    } catch (failure) {
      if (!disposed && generation.requestEpoch === epoch) {
        error.value = errorMessage(failure);
        cancelling.value = false;
      }
    }
  }
  const stopTokenWatch = watch(
    [() => studio.scopeEpoch?.value, studio.token],
    () => {
      ++epoch;
      control?.abort();
      active.value = null;
      pending.value = false;
      cancelling.value = false;
      snapshot.value = null;
      complete.value = false;
      lastInput = null;
      clearAudio();
      error.value = null;
    },
    { flush: 'sync' },
  );
  function dispose() {
    stopTokenWatch();
    disposed = true;
    ++epoch;
    control?.abort();
    clearAudio();
  }
  if (getCurrentScope()) onScopeDispose(dispose);
  return {
    pending,
    cancelling,
    jobId,
    active,
    snapshot,
    results,
    audioUrl,
    audioFilename,
    audioBlob,
    error,
    complete,
    canGenerate,
    generate,
    regenerate,
    removeResult,
    resume,
    cancel,
    dispose,
  };
}
