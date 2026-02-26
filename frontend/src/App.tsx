import { useCallback, useState } from "react";

import type { GenerationSettings, TpmPoint } from "./api/types";
import ChatPanel from "./components/chat/ChatPanel";
import InspectorPanel from "./components/inspector/InspectorPanel";
import Toast from "./components/ui/Toast";
import { useChat } from "./hooks/useChat";
import { useModels } from "./hooks/useModels";
import { useTheme } from "./hooks/useTheme";
import { useToast } from "./hooks/useToast";

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const { toast, showToast } = useToast();
  const [lastInference, setLastInference] = useState("--");

  const [settings, setSettings] = useState<GenerationSettings>({
    temperature: 0.7,
    topP: 0.9,
    topK: 0,
    minP: 0.0,
    repetitionPenalty: 1.1,
    maxTokens: 512,
    stream: true,
    systemPrompt: false,
  });

  const [tpmHistory, setTpmHistory] = useState<TpmPoint[]>([]);
  const addTpmPoint = useCallback(
    (pt: TpmPoint) => setTpmHistory((prev) => [...prev.slice(-19), pt]),
    [],
  );

  const { messages, isGenerating, sendMessage, clearConversation } = useChat({
    showToast,
    setLastInference,
    onTpmPoint: addTpmPoint,
  });

  const { models, activeModel, ramGb, download, load, remove } = useModels({
    showToast,
    clearConversation,
  });

  const [inspectorOpen, setInspectorOpen] = useState(true);

  return (
    <div className={`grid grid-cols-1 ${inspectorOpen ? "md:grid-cols-[1fr_320px]" : "md:grid-cols-[1fr]"} h-screen bg-bg-primary text-txt-primary font-body text-sm leading-relaxed overflow-hidden antialiased transition-all duration-300`}>
      <ChatPanel
        activeModel={activeModel}
        messages={messages}
        isGenerating={isGenerating}
        settings={settings}
        onSend={sendMessage}
        onClear={clearConversation}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <InspectorPanel
        models={models}
        activeModel={activeModel}
        ramGb={ramGb}
        lastInference={lastInference}
        settings={settings}
        onSettingsChange={setSettings}
        onLoad={load}
        onDownload={download}
        onDelete={remove}
        isOpen={inspectorOpen}
        onToggle={() => setInspectorOpen((o) => !o)}
        tpmHistory={tpmHistory}
      />

      <Toast toast={toast} />
    </div>
  );
}
