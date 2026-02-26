import { useCallback, useRef } from "react";

import type { GenerationSettings, UIMessage } from "../../api/types";
import type { Theme } from "../../hooks/useTheme";
import ChatHeader from "./ChatHeader";
import ChatInput from "./ChatInput";
import MessageList from "./MessageList";

interface Props {
  activeModel: string | null;
  messages: UIMessage[];
  isGenerating: boolean;
  settings: GenerationSettings;
  onSend: (text: string, settings: GenerationSettings) => void;
  onClear: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

export default function ChatPanel({
  activeModel,
  messages,
  isGenerating,
  settings,
  onSend,
  onClear,
  theme,
  onToggleTheme,
}: Props) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(
    (text: string) => onSend(text, settings),
    [onSend, settings],
  );

  const handleInsertPrompt = useCallback((text: string) => {
    if (inputRef.current) {
      inputRef.current.value = text;
      inputRef.current.focus();
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 160)}px`;
    }
  }, []);

  return (
    <div className="flex flex-col h-screen border-r border-warm-border">
      <ChatHeader activeModel={activeModel} onClear={onClear} hasMessages={messages.length > 0} theme={theme} onToggleTheme={onToggleTheme} />
      <MessageList messages={messages} onInsertPrompt={handleInsertPrompt} />
      <ChatInput
        isGenerating={isGenerating}
        maxTokens={settings.maxTokens}
        onSend={handleSend}
        inputRef={inputRef}
      />
    </div>
  );
}
