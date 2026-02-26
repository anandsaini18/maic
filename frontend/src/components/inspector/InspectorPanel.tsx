import type { GenerationSettings as Settings, TpmPoint } from "../../api/types";
import type { Model } from "../../api/types";
import GenerationSettings from "./GenerationSettings";
import ModelSelector from "./ModelSelector";
import StatusBar from "./StatusBar";
import TpmChart from "./TpmChart";

interface Props {
  models: Model[];
  activeModel: string | null;
  ramGb: number;
  lastInference: string;
  settings: Settings;
  onSettingsChange: (s: Settings) => void;
  onLoad: (id: string) => void;
  onDownload: (id: string) => void;
  onDelete: (id: string) => void;
  isOpen: boolean;
  onToggle: () => void;
  tpmHistory: TpmPoint[];
}

export default function InspectorPanel({
  models,
  activeModel,
  ramGb,
  lastInference,
  settings,
  onSettingsChange,
  onLoad,
  onDownload,
  onDelete,
  isOpen,
  onToggle,
  tpmHistory,
}: Props) {
  return (
    <>
      {/* Toggle button — always visible on md+ */}
      <button
        onClick={onToggle}
        className="hidden md:flex fixed top-3 right-3 z-50 items-center justify-center w-8 h-8 rounded-md bg-warm-card border border-warm-border text-warm-muted hover:text-warm-text hover:border-amber transition-colors duration-200"
        title={isOpen ? "Collapse panel" : "Expand panel"}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform duration-300 ${isOpen ? "rotate-0" : "rotate-180"}`}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>

      {/* Panel */}
      <div
        className={`hidden md:flex h-screen overflow-y-auto flex-col bg-warm-bg/75 backdrop-blur-[16px] border-l border-warm-border transition-all duration-300 ${
          isOpen ? "w-[320px] opacity-100" : "w-0 opacity-0 overflow-hidden border-l-0"
        }`}
      >
        <ModelSelector
          models={models}
          activeModel={activeModel}
          onLoad={onLoad}
          onDownload={onDownload}
          onDelete={onDelete}
        />

        <TpmChart points={tpmHistory} />

        <StatusBar
          activeModel={activeModel}
          ramGb={ramGb}
          lastInference={lastInference}
        />

        <GenerationSettings settings={settings} onChange={onSettingsChange} />
      </div>
    </>
  );
}
