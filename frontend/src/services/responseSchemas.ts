import { z } from './schema';
import { ApiError } from './http';
import type {
  ServerInfo,
  NormalizationInfo,
  NormalizePreview,
  ProvidersResponse,
  ConfigSnapshot,
  ConfigStatus,
  SettingSchema,
  Job,
  QualityReport,
  Json,
} from './contracts';
import { voiceSummarySchema } from '../features/cloning/cloneSchemas';
const number = z.number().finite();
const optionalNumber = number.nullish().transform((value) => value ?? undefined);
const optionalString = z
  .string()
  .nullish()
  .transform((value) => value ?? undefined);
const json: z.ZodType<Json> = z.lazy(() =>
  z.union([z.null(), z.boolean(), number, z.string(), z.array(json), z.record(z.string(), json)]),
);
const numericBounds = z
  .object({ min: number, max: number, integer: z.boolean().optional(), exclusive_min: z.boolean().optional() })
  .refine((value) => value.min <= value.max);
const coverageLevel = z.enum(['full', 'partial', 'none']);
const normalizationMode = z.enum(['auto', 'off']);
const readingStyle = z.enum(['precise', 'natural', 'digits']);
const byteOffset = number.int().nonnegative();
/** The `normalization` block of GET /api/info. Locale lists come only from the server. */
export const normalizationInfoSchema: z.ZodType<NormalizationInfo> = z.object({
  ruleset_version: z.string().min(1),
  default_mode: normalizationMode,
  default_locale: z.string().optional(),
  default_reading_style: readingStyle.optional(),
  enabled: z.boolean().optional(),
  languages: z.array(
    z.object({
      language: z.string().min(1),
      locales: z.array(z.string().min(1)),
      default_locale: z.string(),
      rules: coverageLevel,
      features: z.record(z.string(), coverageLevel).default({}),
    }),
  ),
});
/** Response of POST /api/tts/normalize. */
export const normalizePreviewSchema: z.ZodType<NormalizePreview> = z.object({
  original: z.string(),
  spoken: z.string(),
  language: z.string().min(1),
  locale: z.string().min(1),
  mode: normalizationMode,
  reading_style: readingStyle,
  normalizer_version: z.string().min(1),
  changes: z.array(
    z
      .object({
        kind: z.string().min(1),
        before: z.string(),
        after: z.string(),
        start: byteOffset,
        end: byteOffset,
        confidence: z.string(),
      })
      .refine((value) => value.end >= value.start),
  ),
  warnings: z.array(
    z
      .object({ code: z.string().min(1), before: z.string(), start: byteOffset, end: byteOffset })
      .refine((value) => value.end >= value.start),
  ),
  coverage: z.record(z.string(), coverageLevel).default({}),
});
export const infoSchema: z.ZodType<ServerInfo> = z.object({
  version: z.string(),
  mode: z.string(),
  setup_complete: z.boolean(),
  auth: z.object({ mode: z.string(), required: z.boolean() }),
  features: z.object({ playback: z.boolean(), audio_devices: z.boolean(), hot_reload: z.boolean() }),
  config_revision: number.int().nonnegative(),
  model: z.object({ status: z.string() }),
  synthesis: z
    .object({
      max_text_chars: number.int().positive(),
      max_body_bytes: number.int().positive(),
      sampling: z.object({
        provider: z.string(),
        seed_max: z.string().regex(/^\d{1,20}$/),
        temperature: numericBounds,
        top_k: numericBounds,
        top_p: numericBounds,
        repetition_penalty: numericBounds,
      }),
    })
    .optional(),
  // A malformed or absent block means "older server": the pronunciation controls stay hidden.
  normalization: normalizationInfoSchema
    .nullish()
    .catch(undefined)
    .transform((value) => value ?? undefined),
});
export const providersSchema: z.ZodType<ProvidersResponse> = z.object({
  providers: z.array(
    z.object({
      model_revision: optionalString,
      provider_id: z.string(),
      state: z.string(),
      enabled: z.boolean(),
      effective_device: optionalString,
      active_model: optionalString,
      detail: optionalString,
      backend_diagnostics: z
        .object({
          compiled_backends: z.array(z.string()),
          available_runtime_backends: z.array(z.string()),
          requested_device: z.string(),
          effective_device: optionalString,
          model_loading_status: z.string(),
          status: z.string(),
          vulkan_gpu_name: optionalString,
          vulkan_initialization_error: optionalString,
          cpu_fallback_reason: optionalString,
        })
        .nullish()
        .transform((value) => value ?? undefined),
    }),
  ),
  capabilities: z
    .array(
      z
        .object({
          provider_id: z.string(),
          models: z.array(z.string()),
          languages: z.array(z.string()),
          devices: z.array(z.string()),
          clone_modes: z.array(z.string()),
          requires_reference_audio: z.boolean(),
          requires_reference_transcript: z.boolean(),
          voices: z.array(z.string()).optional(),
          sample_rate: optionalNumber,
        })
        .nullable(),
    )
    .transform((value) => value.filter((item) => item !== null)),
  diagnostics: z.unknown().optional(),
});
export const voiceListSchema = z.object({ data: z.array(voiceSummarySchema) });
export const builtinListSchema = z.object({ data: z.array(z.object({ id: z.string(), name: z.string() })) });
export const configSchema: z.ZodType<ConfigSnapshot> = z.object({
  revision: number.int().nonnegative(),
  config: z.record(z.string(), json),
});
export const statusSchema: z.ZodType<ConfigStatus> = z.object({
  pending_restart: z.boolean().optional(),
  valid: z.boolean().optional(),
  revision: number.int().nonnegative(),
  config_path: z.string(),
  watcher: optionalString,
  last_reload: optionalString,
  last_error: optionalString,
  pending_applies: z.array(z.string()).optional(),
  apply_failures: z.array(z.object({ path: z.string(), error: z.string(), active: json })).optional(),
});
export const settingSchema: z.ZodType<SettingSchema> = z.object({
  path: z.string(),
  description: z.string(),
  type: z.enum(['string', 'integer', 'number', 'boolean', 'array', 'object']),
  apply: z.enum(['hot', 'restart_server', 'restart_subsystem', 'reload_model', 'restart_process']),
  min: number.optional(),
  max: number.optional(),
  allowed_values: z.array(z.string()).optional(),
  nullable: z.boolean(),
  secret: z.boolean(),
  allowed_values_source: z.string().optional(),
});
export const schemaResponse = z.union([
  z.object({ settings: z.array(settingSchema) }),
  z.record(z.string(), settingSchema),
]);
export const devicesSchema = z.object({ devices: z.array(z.object({ id: z.string(), name: z.string() })) });
export const audioOutputSchema = z.object({ device: z.string(), available: z.boolean() });
const qualitySchema: z.ZodType<QualityReport> = z.object({
  quality_status: z.enum(['ok', 'warning', 'failed']),
  retries: number.int().nonnegative(),
  mode: z.string(),
  issues: z.array(
    z.object({
      kind: z.string(),
      severity: z.enum(['info', 'warning', 'error']),
      detail: z.string(),
      confidence: number.nullish(),
    }),
  ),
  audio: z
    .object({
      sample_count: number,
      sample_rate: number,
      duration_secs: number,
      non_finite_samples: number,
      peak: number,
      rms: number,
      dc_offset: number,
      clipped_samples: number,
      clipping_fraction: number,
      near_silent_fraction: number,
      max_discontinuity: number,
      impulsive_samples: number,
      chunk_boundary_peaks: number,
    })
    .nullish()
    .transform((value) => value ?? undefined),
  echo: z
    .object({
      detected: z.boolean(),
      asr_required: z.boolean(),
      matched_words: number,
      candidate_words: number,
      at_start: z.boolean(),
      confidence: number.nullish(),
      evidence: z.array(z.string()),
    })
    .nullish()
    .transform((value) => value ?? undefined),
});
export const jobSchema: z.ZodType<Job> = z.object({
  id: z.string().optional(),
  stage: z.string().min(1),
  elapsed_secs: optionalNumber,
  audio_secs: optionalNumber,
  rtf: optionalNumber,
  frames: optionalNumber,
  fragments_total: optionalNumber,
  fragments_done: optionalNumber,
  cancellation_requested: z.boolean().optional(),
  deadline_exceeded: z.boolean().optional(),
  error: optionalString,
  quality: qualitySchema.nullish().transform((value) => value ?? undefined),
  context: z
    .object({
      provider: z.string(),
      model: z.string(),
      voice: z.string(),
      effective_device: z.string(),
      seed: optionalString,
    })
    .nullish()
    .transform((value) => value ?? undefined),
});
export const jobCreatedSchema = z.object({ id: z.string().min(1) });
export const setupStatusSchema = z.object({
  constraints: z
    .object({
      version: z.literal(1),
      password_min_chars: number.int().positive(),
      rejected_passwords: z.array(z.string()),
      max_inference_steps: number.int().positive(),
      auth_modes: z.array(z.string()),
      allow_insecure_remote: z.boolean(),
    })
    .optional(),
  model: z.object({ cache_dir: z.string() }),
  server: z.object({ port: number.int().positive() }),
});
export const setupResultSchema = z.object({
  api_url: z.string().url(),
  audio_device: z.string(),
  auth: z.string(),
  api_token: optionalString,
});
export function parseResponse<T>(schema: z.ZodType<T>, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success)
    throw new ApiError(0, 'The server response does not match the expected format.', { kind: 'invalid_response' });
  return result.data;
}
