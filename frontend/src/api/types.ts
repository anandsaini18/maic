/* Shared types used across hooks and components. */

export interface Model {
  id: string;
  name: string;
  size_gb: number;
  feasible: boolean;
  downloaded: boolean;
  disk_gb: number | null;
  active: boolean;
  requires_token: boolean;
  downloading: boolean;
  download_error: string | null;
  downloads: number;
}

export interface ModelsStatus {
  available_ram_gb: number;
  active_model: string | null;
  models: Model[];
}

export interface GenerationSettings {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  maxTokens: number;
  stream: boolean;
  systemPrompt: boolean;
}

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Rendered HTML (set after streaming finishes). */
  html?: string;
  /** Wall-clock seconds for assistant responses. */
  elapsed?: string;
  /** True while still receiving SSE tokens. */
  streaming?: boolean;
}

export interface TpmPoint {
  ts: number;
  tokPerSec: number;
}
