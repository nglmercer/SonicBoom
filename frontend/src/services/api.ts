import type { HttpClient } from './http';
import { jsonBody } from './http';
import * as responses from './responseSchemas';
import { voiceCreateResultSchema } from '../features/cloning/cloneSchemas';
import { parseConstraints } from '../features/cloning/constraints';
import { pronunciationQuery, previewRequestBody } from '../features/speech/pronunciation';
import type {
  ServerInfo,
  ProvidersResponse,
  BuiltinVoice,
  Voice,
  ConfigSnapshot,
  ConfigStatus,
  Schema,
  SettingSchema,
  JsonObject,
  AudioDevice,
  Job,
  CloneInput,
  SpeechPronunciation,
  NormalizeRequest,
  SetupStatus,
  SetupInput,
  SetupResult,
} from './contracts';
const id = encodeURIComponent;
/** Optional Qwen3-TTS sampling/quality query parameters. Empty values are omitted. */
export type SamplingQuery = Record<string, string | number | undefined | null>;
export function createApi(http: HttpClient) {
  return {
    info: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.infoSchema, await http.json<unknown>('/api/info', { signal })),
    providers: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.providersSchema, await http.json<unknown>('/api/providers', { signal })),
    builtinVoices: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.builtinListSchema, await http.json<unknown>('/v1/voices', { signal })).data,
    voices: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.voiceListSchema, await http.json<unknown>('/api/voices', { signal })).data,
    voiceConstraints: async (signal?: AbortSignal) =>
      parseConstraints(await http.json<unknown>('/api/voices/constraints', { signal })),
    createVoice: async (input: CloneInput, file: File, signal?: AbortSignal) => {
      const body = new FormData();
      body.append('metadata', new Blob([JSON.stringify(input)], { type: 'application/json' }));
      body.append('file', file);
      return responses.parseResponse(
        voiceCreateResultSchema,
        await http.json<unknown>('/api/voices/clones', { method: 'POST', body, signal }),
      );
    },
    deleteVoice: (voice: string) => http.response(`/api/voices/${id(voice)}`, { method: 'DELETE' }),
    previewVoice: (voice: string, text: string) =>
      http.blob(`/api/voices/${id(voice)}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: text,
      }),
    config: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.configSchema, await http.json<unknown>('/api/config', { signal })),
    configStatus: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.statusSchema, await http.json<unknown>('/api/config/status', { signal })),
    schema: async (signal?: AbortSignal): Promise<Schema> => {
      const value = responses.parseResponse(
        responses.schemaResponse,
        await http.json<unknown>('/api/config/schema', { signal }),
      );
      return Array.isArray(value.settings)
        ? Object.fromEntries(value.settings.map((setting) => [setting.path, setting]))
        : (value as Schema);
    },
    patchConfig: (changes: JsonObject, revision?: number) =>
      http.json<unknown>('/api/config', {
        method: 'PATCH',
        ...jsonBody({ changes, ...(revision === undefined ? {} : { expected_revision: revision }) }),
      }),
    configAction: (action: 'reload' | 'validate' | 'open') =>
      http.response(`/api/config/${action}`, { method: 'POST' }),
    audioDevices: async (signal?: AbortSignal) =>
      responses.parseResponse(responses.devicesSchema, await http.json<unknown>('/api/audio/devices', { signal }))
        .devices,
    audioOutput: async () =>
      responses.parseResponse(responses.audioOutputSchema, await http.json<unknown>('/api/audio/output')),
    testAudio: (text: string) =>
      http.response('/api/tts/play', { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: text }),
    createJob: (
      text: string,
      language: string,
      voice: string,
      cloned: boolean,
      sampling?: SamplingQuery,
      signal?: AbortSignal,
      pronunciation?: SpeechPronunciation,
      provider?: string,
    ) => {
      const params = new URLSearchParams({ lang: language, format: 'wav', [cloned ? 'profile' : 'voice']: voice });
      // Named-voice engines other than Supertonic (Kyutai TTS) are routed by provider.
      if (provider) params.set('provider', provider);
      if (sampling)
        for (const [key, value] of Object.entries(sampling)) {
          if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
        }
      // Added last and only when present: legacy callers produce the same URL as before.
      for (const [key, value] of Object.entries(pronunciationQuery(pronunciation))) params.set(key, value);
      return http
        .json<unknown>(`/api/tts/jobs?${params}`, {
          method: 'POST',
          headers: { 'Content-Type': 'text/plain' },
          body: text,
          signal,
        })
        .then((value) => responses.parseResponse(responses.jobCreatedSchema, value));
    },
    /** "What will be spoken?" preview. Pass a signal so a stale request can be cancelled. */
    previewNormalization: async (request: NormalizeRequest, signal?: AbortSignal) =>
      responses.parseResponse(
        responses.normalizePreviewSchema,
        await http.json<unknown>('/api/tts/normalize', {
          method: 'POST',
          ...jsonBody(previewRequestBody(request)),
          signal,
        }),
      ),
    job: async (job: string, signal?: AbortSignal) =>
      responses.parseResponse(responses.jobSchema, await http.json<unknown>(`/api/tts/jobs/${id(job)}`, { signal })),
    jobEvents: (job: string, signal?: AbortSignal) => http.response(`/api/tts/jobs/${id(job)}/events`, { signal }),
    jobAudio: (job: string, signal?: AbortSignal) => http.blob(`/api/tts/jobs/${id(job)}/audio`, { signal }),
    cancelJob: (job: string) => http.response(`/api/tts/jobs/${id(job)}/cancel`, { method: 'POST' }),
    setupStatus: async () =>
      responses.parseResponse(responses.setupStatusSchema, await http.json<unknown>('/setup/status')),
    setupDevices: async () =>
      responses.parseResponse(responses.devicesSchema, await http.json<unknown>('/setup/audio-devices')).devices,
    completeSetup: async (input: SetupInput) =>
      responses.parseResponse(
        responses.setupResultSchema,
        await http.json<unknown>('/setup/complete', { method: 'POST', ...jsonBody(input) }),
      ),
  };
}
export type Api = ReturnType<typeof createApi>;
