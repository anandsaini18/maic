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
  onQuantize: (id: string, qBits?: number) => void;
  isOpen: boolean;
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
  onQuantize,
  isOpen,
  tpmHistory,
}: Props) {
  return (
    <>
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
          onQuantize={onQuantize}
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
