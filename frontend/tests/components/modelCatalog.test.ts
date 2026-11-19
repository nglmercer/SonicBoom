import { ref } from 'vue';
import { describe, expect, it } from 'vitest';
import { useModelCatalog } from '../../src/composables/useModelCatalog';
import type { Capabilities, ProviderStatus } from '../../src/services/contracts';

const cap = (over: Partial<Capabilities> & { provider_id: string; models: string[] }): Capabilities => ({
  languages: ['en'],
  devices: ['cpu'],
  clone_modes: [],
  requires_reference_audio: false,
  requires_reference_transcript: false,
  ...over,
});
const ready = (provider_id: string, active_model: string): ProviderStatus =>
  ({ provider_id, state: 'ready', enabled: true, active_model }) as ProviderStatus;

function setup(modelId: string) {
  const model = ref(modelId);
  const voice = ref('');
  const language = ref('en');
  const catalog = useModelCatalog({
    model,
    voice,
    language,
    voices: ref([]),
    builtinVoices: ref([{ id: 'M1', name: 'M1' }]),
    providers: ref([ready('kyutai-candle', 'kyutai-tts-1.6b-en-fr')]),
    capabilities: ref([
      cap({ provider_id: 'supertonic', models: ['supertonic-3'] }),
      cap({
        provider_id: 'kyutai-candle',
        models: ['kyutai-tts-1.6b-en-fr'],
        languages: ['en', 'fr'],
        voices: ['calm', 'other'],
      }),
      cap({ provider_id: 'qwen3-candle', models: ['qwen3-tts-candle'], requires_reference_audio: true, clone_modes: ['x_vector'] }),
    ]),
    info: ref(null),
    modelReady: true,
  });
  return { ...catalog, voice };
}

describe('model catalog with named-voice engines', () => {
  it('lists Kyutai next to Supertonic and the cloning engines', () => {
    const { models } = setup('supertonic-3');
    expect(models.value.map((m) => m.id)).toEqual(['supertonic-3', 'qwen3-tts-candle', 'kyutai-tts-1.6b-en-fr']);
    const kyutai = models.value.find((m) => m.id === 'kyutai-tts-1.6b-en-fr');
    expect(kyutai).toMatchObject({ provider: 'kyutai-candle', cloned: false, ready: true, languages: ['en', 'fr'] });
  });

  it('offers the engine voices, not the Supertonic styles, for a named-voice model', () => {
    const { availableVoices } = setup('kyutai-tts-1.6b-en-fr');
    expect(availableVoices.value.map((v) => v.id)).toEqual(['calm', 'other']);
    expect(availableVoices.value.every((v) => !v.cloned)).toBe(true);
  });

  it('keeps the built-in Supertonic styles for Supertonic', () => {
    const { availableVoices } = setup('supertonic-3');
    expect(availableVoices.value.map((v) => v.id)).toEqual(['M1']);
  });
});
