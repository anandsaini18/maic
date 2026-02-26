import { useEffect, useRef } from "react";

import type { UIMessage } from "../../api/types";
import { highlightCodeBlocks } from "../../lib/markdown";
import TypingIndicator from "./TypingIndicator";

interface Props {
  message: UIMessage;
}

export default function Message({ message: m }: Props) {
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (m.html && contentRef.current) {
      highlightCodeBlocks(contentRef.current);
    }
  }, [m.html]);

  const isUser = m.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end opacity-0 animate-msg-slide py-3">
        <div className="max-w-[80%] md:max-w-[65%]">
          <div className="bg-warm-card border border-warm-border rounded-sm rounded-br-xs px-4 py-3">
            <div className="text-sm leading-relaxed text-warm-text font-body whitespace-pre-wrap break-words">
              {m.content}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="opacity-0 animate-msg-slide py-3">
      <div className="max-w-[90%] md:max-w-[75%]">
        {/* Role label */}
        <div className="flex items-center gap-2 mb-2">
          <div className="w-5 h-5 rounded-xs bg-gradient-to-br from-amber/20 to-amber/5 border border-amber/15 flex items-center justify-center">
            <span className="text-amber text-[9px] font-hero font-bold">M</span>
          </div>
          <span className="font-display text-[10px] text-amber-muted tracking-[0.5px] uppercase">
            maic
          </span>
        </div>

        {/* Content */}
        {m.streaming && !m.content ? (
          <TypingIndicator />
        ) : m.html ? (
          <div
            ref={contentRef}
            className="msg-text font-prose text-[15px] leading-[1.75] text-warm-text break-words"
            dangerouslySetInnerHTML={{ __html: m.html }}
          />
        ) : (
          <div className="msg-text font-prose text-[15px] leading-[1.75] text-warm-text break-words whitespace-pre-wrap">
            {m.content}
          </div>
        )}

        {/* Elapsed time */}
        {m.elapsed && (
          <div className="font-display text-[10px] text-warm-muted mt-2.5 flex items-center gap-1.5">
            <svg className="w-3 h-3 opacity-50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            {m.elapsed}s
          </div>
        )}
      </div>
    </div>
  );
}
