import { useEffect, useRef, useState } from "react";

import type { Model } from "../../api/types";
import ModelOption from "./ModelOption";

interface Props {
  models: Model[];
  activeModel: string | null;
  onLoad: (id: string) => void;
  onDownload: (id: string) => void;
  onDelete: (id: string) => void;
  onQuantize: (id: string, qBits?: number) => void;
}

/** Looks like a HuggingFace model ID: "org/model-name" */
const isHfId = (s: string) => /^[\w.-]+\/[\w.-]+$/.test(s.trim());

export default function ModelSelector({
  models,
  activeModel,
  onLoad,
  onDownload,
  onDelete,
  onQuantize,
}: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const wrapperRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);

  useEffect(() => {
    if (open) {
      searchInputRef.current?.focus();
    } else {
      const id = setTimeout(() => setSearch(""), 0);
      return () => clearTimeout(id);
    }
  }, [open]);

  const q = search.toLowerCase();
  const filteredModels = models.filter(
    (m) =>
      m.name.toLowerCase().includes(q) ||
      m.id.toLowerCase().includes(q),
  );

  const sortedModels = [...filteredModels].sort((a, b) => {
    const ad = a.disk_gb !== null && a.disk_gb > 0;
    const bd = b.disk_gb !== null && b.disk_gb > 0;
    if (ad === bd) return 0;
    return ad ? -1 : 1;
  });

  // Show "Load from HF" action when search looks like a model ID
  // and either no results or the exact ID isn't in the list
  const trimmed = search.trim();
  const showLoadAction =
    isHfId(trimmed) &&
    !models.some((m) => m.id.toLowerCase() === trimmed.toLowerCase());

  const handleLoadCustom = () => {
    onLoad(trimmed);
    setSearch("");
    setOpen(false);
  };

  const active = models.find((m) => m.id === activeModel);
  const sizeLabel = active?.disk_gb
    ? `${active.disk_gb} GB on disk`
    : active
      ? `~${active.size_gb} GB`
      : "";

  return (
    <div className="p-5 border-b border-warm-border">
      <div className="font-display text-[10px] font-medium uppercase tracking-[1.5px] text-warm-muted mb-3.5">
        Models
      </div>

      <div className="relative" ref={wrapperRef}>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className={`w-full flex justify-between items-center bg-warm-card border p-3 cursor-pointer transition-all duration-[250ms] ease-smooth gap-3 ${
            open
              ? "border-amber bg-amber-dim/30 rounded-t-sm rounded-b-none"
              : "border-warm-border hover:border-amber/30 hover:bg-warm-border/20 rounded-sm"
          }`}
        >
          <div className="flex-1 min-w-0 flex flex-col gap-1 text-left">
            <span className="font-display text-[11px] font-medium text-warm-text truncate">
              {active ? active.name : "Loading..."}
            </span>
            {sizeLabel && (
              <span className="font-display text-[10px] text-warm-muted">
                {sizeLabel}
              </span>
            )}
          </div>
          <span
            className={`text-xs text-amber transition-transform duration-300 ease-spring shrink-0 ${
              open ? "rotate-180" : ""
            }`}
          >
            ↓
          </span>
        </button>

        {open && (
          <div className="absolute top-full left-0 right-0 bg-warm-bg border border-amber border-t-0 rounded-b-sm z-[100] -mt-px overflow-hidden">
            <div className="bg-warm-card border-b border-warm-border/50 p-3 flex items-center gap-2">
              <svg className="w-3 h-3 text-warm-muted/60 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                ref={searchInputRef}
                type="text"
                placeholder="Search or paste model ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && showLoadAction) {
                    handleLoadCustom();
                  }
                }}
                className="w-full bg-transparent text-[11px] text-warm-text placeholder-warm-muted/60 focus:outline-none font-display"
              />
            </div>
            <div className="max-h-[240px] overflow-y-auto">
              {sortedModels.map((m) => (
                <ModelOption
                  key={m.id}
                  model={m}
                  onLoad={onLoad}
                  onDownload={onDownload}
                  onDelete={onDelete}
                  onQuantize={onQuantize}
                  onClose={() => setOpen(false)}
                />
              ))}

              {showLoadAction && (
                <button
                  type="button"
                  onClick={handleLoadCustom}
                  className="w-full flex items-center gap-2.5 px-4 py-3 text-left hover:bg-warm-border/20 transition-colors border-t border-warm-border"
                >
                  <span className="w-5 h-5 rounded-xs bg-gradient-to-br from-amber/25 to-amber/10 border border-amber/20 flex items-center justify-center shrink-0">
                    <svg className="w-3 h-3 text-amber" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="font-display text-[11px] font-medium text-amber truncate">
                      Load "{trimmed}"
                    </div>
                    <div className="font-display text-[10px] text-warm-muted">
                      from HuggingFace Hub
                    </div>
                  </div>
                  <span className="text-[10px] text-warm-muted font-display">
                    Enter ↵
                  </span>
                </button>
              )}

              {sortedModels.length === 0 && !showLoadAction && (
                <div className="px-3 py-4 text-center text-sm text-warm-muted">
                  No models found
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
