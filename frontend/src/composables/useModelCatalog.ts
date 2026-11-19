import { computed, watch, type Ref } from 'vue';
import type { BuiltinVoice, Capabilities, ProviderStatus, ServerInfo, Voice } from '../services/contracts';
import { modelLanguages } from '../services/providerLanguages';
import { isModelReady } from '../services/providerReadiness';

type ModelCatalogInput = {
  model: Ref<string>;
  voice: Ref<string>;
  language: Ref<string>;
  voices: Ref<Voice[]>;
  builtinVoices: Ref<BuiltinVoice[]>;
  providers: Ref<ProviderStatus[]>;
  capabilities: Ref<Capabilities[]>;
  info: Ref<ServerInfo | null>;
  modelReady: boolean;
};

/**
 * Builds the model list from provider capabilities and keeps the selected
 * voice/language consistent with what the active model actually supports.
 */
export function useModelCatalog(input: ModelCatalogInput) {
  const { model, voice, language, voices, builtinVoices, providers, capabilities, info, modelReady } = input;

  const models = computed(() => [
    {
      id: 'supertonic-3',
      provider: 'supertonic',
      cloned: false,
      languages: capabilities.value.find((cap) => cap.provider_id === 'supertonic')?.languages || ['en'],
      ready: info.value ? info.value.model.status === 'ready' : modelReady,
      namedVoices: [] as string[],
    },
    ...capabilities.value
      .filter((cap) => cap.requires_reference_audio)
      .flatMap((cap) =>
        cap.models.map((id) => ({
          id,
          provider: cap.provider_id,
          cloned: true,
          languages: modelLanguages(cap, id),
          ready: isModelReady(providers.value, cap.provider_id, id),
          namedVoices: [] as string[],
        })),
      ),
    // Providers with built-in named voices and no voice profile (Kyutai TTS).
    ...capabilities.value
      .filter((cap) => !cap.requires_reference_audio && cap.provider_id !== 'supertonic' && cap.voices?.length)
      .flatMap((cap) =>
        cap.models.map((id) => ({
          id,
          provider: cap.provider_id,
          cloned: false,
          languages: modelLanguages(cap, id),
          ready: isModelReady(providers.value, cap.provider_id, id),
          namedVoices: cap.voices ?? [],
        })),
      ),
  ]);

  const selectedModel = computed(() => models.value.find((item) => item.id === model.value) || models.value[0]);

  function profileCompatible(profile: Voice) {
    const provider = providers.value.find((item) => item.provider_id === profile.provider);
    const cap = capabilities.value.find((item) => item.provider_id === profile.provider);
    return (
      !(provider?.model_revision && profile.model_revision && provider.model_revision !== profile.model_revision) &&
      (!cap ||
        (cap.models.includes(profile.model) &&
          cap.clone_modes.includes(profile.clone_mode) &&
          cap.languages.includes(profile.language)))
    );
  }

  const availableVoices = computed(() =>
    selectedModel.value.cloned
      ? voices.value
          .filter((item) => item.model === model.value && profileCompatible(item))
          .map((item) => ({
            id: item.id,
            name: item.name,
            detail: `${item.provider} · ${item.language}`,
            cloned: true,
          }))
      : selectedModel.value.namedVoices.length
        ? selectedModel.value.namedVoices.map((name) => ({ id: name, name, detail: 'Built-in voice', cloned: false }))
        : builtinVoices.value.map((item) => ({ ...item, detail: 'Built-in voice', cloned: false })),
  );

  watch(availableVoices, (value) => {
    if (!value.some((item) => item.id === voice.value)) voice.value = value[0]?.id || '';
  });
  watch(voice, (value) => {
    const profile = voices.value.find((item) => item.id === value);
    if (profile) language.value = profile.language;
  });
  watch(selectedModel, (value) => {
    if (value.languages.length && !value.languages.includes(language.value)) language.value = value.languages[0];
  });

  return { models, selectedModel, availableVoices, profileCompatible };
}
