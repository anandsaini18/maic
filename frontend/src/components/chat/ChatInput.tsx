import {
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useRef,
} from "react";

interface Props {
  isGenerating: boolean;
  maxTokens: number;
  onSend: (text: string) => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}

export default function ChatInput({
  isGenerating,
  maxTokens,
  onSend,
  inputRef,
}: Props) {
  const tokenCount = useRef(0);

  const handleSubmit = useCallback(
    (e?: FormEvent) => {
      e?.preventDefault();
      const text = inputRef.current?.value.trim() ?? "";
      if (!text || isGenerating) return;
      onSend(text);
      if (inputRef.current) {
        inputRef.current.value = "";
        inputRef.current.style.height = "auto";
        tokenCount.current = 0;
      }
    },
    [isGenerating, onSend, inputRef],
  );

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const autoResize = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    tokenCount.current = Math.ceil(el.value.trim().length / 4);
  }, [inputRef]);

  const pct =
    tokenCount.current > 0
      ? Math.round((tokenCount.current / maxTokens) * 100)
      : 0;

  return (
    <div className="px-6 md:px-10 pb-5 pt-3 shrink-0">
      <div
        className={`relative rounded-sm transition-all duration-300 ${
          isGenerating
            ? "animate-breathing-glow border border-amber/40"
            : "border border-warm-border focus-within:border-amber/30 focus-within:shadow-[0_0_0_3px_rgba(212,160,83,0.08)]"
        } bg-warm-card`}
      >
        <textarea
          ref={inputRef}
          className="w-full bg-transparent border-none outline-none text-warm-text font-body text-sm px-4 py-3.5 pr-[52px] resize-none min-h-[48px] max-h-[160px] leading-normal placeholder:text-warm-muted"
          placeholder="Message Maic..."
          rows={1}
          onKeyDown={handleKey}
          onInput={autoResize}
        />
        <button
          type="button"
          disabled={isGenerating}
          onClick={() => handleSubmit()}
          className="absolute right-2.5 bottom-2.5 w-8 h-8 rounded-xs bg-amber border-none cursor-pointer flex items-center justify-center transition-all duration-300 ease-spring text-bg-primary hover:scale-[1.08] hover:bg-amber-light active:scale-95 disabled:opacity-20 disabled:cursor-not-allowed disabled:transform-none"
          title="Send message"
        >
          <svg
            className="w-3.5 h-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </button>
      </div>

      <div className="flex justify-between px-1 pt-2 text-[10px] text-warm-muted font-display">
        <span className="tracking-wide">
          <span className="text-warm-muted/60">&#9166;</span> send
          <span className="mx-1.5 text-warm-border">&#183;</span>
          <span className="text-warm-muted/60">&#8679;&#9166;</span> newline
        </span>
        {tokenCount.current > 0 && (
          <span className={pct > 80 ? "text-red" : ""}>
            {tokenCount.current} / {maxTokens}
          </span>
        )}
      </div>
    </div>
  );
}
