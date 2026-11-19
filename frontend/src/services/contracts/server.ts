import type { SynthesisConstraints } from './speech';
import type { NormalizationInfo } from './pronunciation';

export interface ServerInfo {
  synthesis?: SynthesisConstraints;
  normalization?: NormalizationInfo;
  version: string;
  mode: string;
  setup_complete: boolean;
  auth: { mode: string; required: boolean };
  features: { playback: boolean; audio_devices: boolean; hot_reload: boolean };
  config_revision: number;
  model: { status: string };
}
export interface BackendDiagnostics {
  compiled_backends: string[];
  available_runtime_backends: string[];
  vulkan_gpu_name?: string | null;
  requested_device: string;
  effective_device?: string | null;
  model_loading_status: string;
  vulkan_initialization_error?: string | null;
  cpu_fallback_reason?: string | null;
  status: string;
}
export interface ProviderStatus {
  model_revision?: string;
  backend_diagnostics?: BackendDiagnostics;
  provider_id: string;
  state: string;
  enabled: boolean;
  effective_device?: string | null;
  active_model?: string | null;
  detail?: string | null;
}
export interface Capabilities {
  provider_id: string;
  models: string[];
  languages: string[];
  devices: string[];
  clone_modes: string[];
  requires_reference_audio: boolean;
  requires_reference_transcript: boolean;
  /** Built-in named voices selectable without a voice profile (Kyutai TTS). */
  voices?: string[];
  sample_rate?: number;
}
export interface ProvidersResponse {
  providers: ProviderStatus[];
  capabilities: Capabilities[];
  diagnostics?: unknown;
}

export interface AudioDevice {
  id: string;
  name: string;
}
