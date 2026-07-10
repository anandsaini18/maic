import { updateSettings } from "../../api/client";
import type { GenerationSettings as Settings } from "../../api/types";
import SliderGroup from "./SliderGroup";
import ToggleRow from "./ToggleRow";

interface Props {
  settings: Settings;
  onChange: (s: Settings) => void;
}

export default function GenerationSettings({ settings, onChange }: Props) {
  const set = <K extends keyof Settings>(key: K, val: Settings[K]) =>
    onChange({ ...settings, [key]: val });

  const setKvSetting = (key: "maxKvSize" | "kvBits", val: number) => {
    onChange({ ...settings, [key]: val });
    const payload =
      key === "maxKvSize"
        ? { max_kv_size: val }
        : { kv_bits: val };
    updateSettings(payload).catch(() => {});
  };

  return (
    <>
      {/* ── Sampling ─────────────────────────────────────── */}
      <div className="p-5 border-b border-warm-border">
        <div className="font-display text-[11px] font-medium uppercase tracking-[1.5px] text-warm-muted mb-4">
          Sampling
        </div>

        <SliderGroup
          label="Temperature"
          description="Controls randomness. Lower is more focused, higher is more creative."
          value={settings.temperature}
          min={0}
          max={2}
          step={0.1}
          onChange={(v) => set("temperature", v)}
        />
        <SliderGroup
          label="Top-P"
          description="Nucleus sampling. Considers tokens within this cumulative probability."
          value={settings.topP}
          min={0}
          max={1}
          step={0.05}
          onChange={(v) => set("topP", v)}
        />
        <SliderGroup
          label="Top-K"
          description="Restricts to the K most likely tokens. 0 disables."
          value={settings.topK}
          min={0}
          max={200}
          step={1}
          onChange={(v) => set("topK", v)}
        />
        <SliderGroup
          label="Min-P"
          description="Filters tokens below this fraction of the top token's probability."
          value={settings.minP}
          min={0}
          max={0.5}
          step={0.01}
          onChange={(v) => set("minP", v)}
        />
        <SliderGroup
          label="Repetition Penalty"
          description="Penalizes repeated tokens. Higher values reduce loops in small models."
          value={settings.repetitionPenalty}
          min={1.0}
          max={2.0}
          step={0.05}
          onChange={(v) => set("repetitionPenalty", v)}
        />
      </div>

      {/* ── Output ───────────────────────────────────────── */}
      <div className="p-5 border-b border-warm-border">
        <div className="font-display text-[11px] font-medium uppercase tracking-[1.5px] text-warm-muted mb-4">
          Output
        </div>
        <SliderGroup
          label="Max Tokens"
          description="Maximum number of tokens to generate in the response."
          value={settings.maxTokens}
          min={32}
          max={4096}
          step={32}
          onChange={(v) => set("maxTokens", v)}
        />
        <ToggleRow
          label="Stream responses"
          description="Display tokens as they are generated."
          value={settings.stream}
          onChange={(v) => set("stream", v)}
        />
      </div>

      {/* ── KV Cache ─────────────────────────────────────── */}
      <div className="p-5 border-b border-warm-border">
        <div className="font-display text-[11px] font-medium uppercase tracking-[1.5px] text-warm-muted mb-4">
          KV Cache
        </div>
        <SliderGroup
          label="Max KV Size"
          description="Cap the KV cache to this many entries (rotating). 0 = unlimited."
          value={settings.maxKvSize}
          min={0}
          max={16384}
          step={512}
          onChange={(v) => setKvSetting("maxKvSize", v)}
        />
        <SliderGroup
          label="KV Cache Bits"
          description="Quantize the KV cache to fewer bits for lower memory usage. 0 = full precision."
          value={settings.kvBits}
          min={0}
          max={8}
          step={4}
          onChange={(v) => setKvSetting("kvBits", v)}
        />
      </div>

      {/* ── Batching ──────────────────────────────────────── */}
      <div className="p-5 border-b border-warm-border">
        <div className="font-display text-[11px] font-medium uppercase tracking-[1.5px] text-warm-muted mb-4">
          Batching
        </div>
        <ToggleRow
          label="Continuous batching"
          description="Serve multiple requests concurrently via BatchGenerator."
          value={settings.batchMode}
          onChange={(v) => {
            onChange({ ...settings, batchMode: v });
            updateSettings({ batch_mode: v }).catch(() => {});
          }}
        />
      </div>

      {/* ── System ───────────────────────────────────────── */}
      <div className="p-5 border-b border-warm-border">
        <div className="font-display text-[11px] font-medium uppercase tracking-[1.5px] text-warm-muted mb-4">
          System
        </div>
        <ToggleRow
          label="System prompt"
          description="Prepend a system message to guide model behavior."
          value={settings.systemPrompt}
          onChange={(v) => set("systemPrompt", v)}
        />
      </div>
    </>
  );
}
