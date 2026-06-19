import { useEffect, useRef } from "react";

import type { UIMessage } from "../../api/types";
import Message from "./Message";
import WelcomeScreen from "./WelcomeScreen";

interface Props {
  messages: UIMessage[];
  onInsertPrompt: (text: string) => void;
}

export default function MessageList({ messages, onInsertPrompt }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto px-6">
        <WelcomeScreen onInsert={onInsertPrompt} />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-10 flex flex-col">
      <div className="flex-1" />
      {messages.map((m) => (
        <Message key={m.id} message={m} />
      ))}
      <div ref={bottomRef} className="pb-2" />
    </div>
  );
}
