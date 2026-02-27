import type { ModelsStatus } from "./types";

const API = "";

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const res = await fetch(`${API}/v1/models/status`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function downloadModel(modelId: string): Promise<void> {
  const res = await fetch(`${API}/v1/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
}

export async function loadModel(modelId: string): Promise<void> {
  const res = await fetch(`${API}/v1/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
}

export async function deleteModel(modelId: string): Promise<void> {
  const res = await fetch(`${API}/v1/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
}

/**
 * Send a chat completion request. Returns the raw Response so callers can
 * choose between streaming (SSE) and blocking (JSON) consumption.
 */
export async function chatCompletion(body: {
  model: string;
  messages: { role: string; content: string }[];
  stream: boolean;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  max_tokens: number;
}): Promise<Response> {
  const res = await fetch(`${API}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    if (res.status === 503)
      throw new Error("Model is still loading. Please wait and try again.");
    const err = await res.json().catch(() => ({}));
    throw new Error(
      err.detail?.error?.message || err.detail || `HTTP ${res.status}`,
    );
  }
  return res;
}
