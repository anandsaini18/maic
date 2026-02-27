import { useCallback, useRef, useState } from "react";

import { chatCompletion } from "../api/client";
import type { GenerationSettings, TpmPoint, UIMessage } from "../api/types";
import { escapeHtml, renderMarkdown } from "../lib/markdown";
import type { ToastType } from "./useToast";

let msgCounter = 0;
const uid = () => `msg-${++msgCounter}-${Date.now()}`;

interface UseChatOpts {
  showToast: (msg: string, type?: ToastType) => void;
  setLastInference: (s: string) => void;
  onTpmPoint: (point: TpmPoint) => void;
}

export function useChat({ showToast, setLastInference, onTpmPoint }: UseChatOpts) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const isGeneratingRef = useRef(false);
  const historyRef = useRef<{ role: string; content: string }[]>([]);

  const clearConversation = useCallback(() => {
    setMessages([]);
    historyRef.current = [];
  }, []);

  const sendMessage = useCallback(
    async (text: string, settings: GenerationSettings) => {
      if (!text.trim() || isGeneratingRef.current) return;

      /* User message */
      const userMsg: UIMessage = {
        id: uid(),
        role: "user",
        content: text,
      };
      historyRef.current.push({ role: "user", content: text });

      /* Placeholder assistant message */
      const assistantId = uid();
      const assistantMsg: UIMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      isGeneratingRef.current = true;
      setIsGenerating(true);

      const body = {
        model: "local",
        messages: historyRef.current,
        stream: settings.stream,
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
        min_p: settings.minP,
        repetition_penalty: settings.repetitionPenalty,
        max_tokens: settings.maxTokens,
      };

      const startTime = performance.now();
      let fullText = "";

      try {
        const res = await chatCompletion(body);

        let tokPerSec: number | null = null;

        if (settings.stream) {
          const result = await consumeSSE(res, assistantId, setMessages);
          fullText = result.text;
          tokPerSec = result.tokPerSec;
        } else {
          const data = await res.json();
          fullText = data.choices[0].message.content;
          tokPerSec = data.usage?.tokens_per_second ?? null;
        }

        historyRef.current.push({ role: "assistant", content: fullText });

        if (tokPerSec !== null) {
          onTpmPoint({ ts: Date.now(), tokPerSec });
        }

        const elapsed = (
          (performance.now() - startTime) /
          1000
        ).toFixed(1);
        const html = renderMarkdown(fullText);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: fullText, html, elapsed, streaming: false }
              : m,
          ),
        );
        setLastInference(`${elapsed}s`);
      } catch (e: unknown) {
        const errMsg = (e as Error).message;
        showToast(errMsg, "error");
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: `Error: ${errMsg}`,
                  html: `<span style="color:var(--red)">Error: ${escapeHtml(errMsg)}</span>`,
                  streaming: false,
                }
              : m,
          ),
        );
      } finally {
        isGeneratingRef.current = false;
        setIsGenerating(false);
      }
    },
    [showToast, setLastInference, onTpmPoint],
  );

  return {
    messages,
    isGenerating,
    sendMessage,
    clearConversation,
  } as const;
}

/* ── SSE consumer ──────────────────────────────────────────────────────── */

async function consumeSSE(
  res: Response,
  assistantId: string,
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>,
): Promise<{ text: string; tokPerSec: number | null }> {
  if (!res.body) {
    return { text: "", tokPerSec: null };
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  let tokPerSec: number | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let sseFinished = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data: ")) continue;
      const payload = trimmed.slice(6);
      if (payload === "[DONE]") { sseFinished = true; break; }

      try {
        const chunk = JSON.parse(payload);
        const delta = chunk.choices?.[0]?.delta;
        if (delta?.content) {
          fullText += delta.content;
          const snap = fullText;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: snap } : m,
            ),
          );
        }
        if (chunk.usage?.tokens_per_second != null) {
          tokPerSec = chunk.usage.tokens_per_second;
        }
      } catch {
        /* skip malformed chunk */
      }
    }
    if (sseFinished) break;
  }

  return { text: fullText, tokPerSec };
}
