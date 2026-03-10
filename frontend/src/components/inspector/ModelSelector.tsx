import { useEffect, useRef, useState } from "react";

import type { Model } from "../../api/types";
import ModelOption from "./ModelOption";

interface Props {
  models: Model[];
  activeModel: string | null;
  onLoad: (id: string) => void;
  onDownload: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function ModelSelector({
  models,
  activeModel,
  onLoad,
  onDownload,
  onDelete,
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
      setSearch("");
    }
  }, [open]);

  const filteredModels = models.filter((m) =>
    m.name.toLowerCase().includes(search.toLowerCase()) ||
    m.id.toLowerCase().includes(search.toLowerCase())
  );

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
          className={`w-full flex justify-between items-center bg-warm-card border rounded-sm p-3 cursor-pointer transition-all duration-[250ms] ease-smooth gap-3 ${
            open
              ? "border-amber bg-amber-dim/30"
              : "border-warm-border hover:border-amber/30 hover:bg-warm-border/20"
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
          <div className="absolute top-full left-0 right-0 bg-warm-bg border border-warm-border border-t-0 rounded-b-sm z-[100] -mt-px flex flex-col">
            <input
              ref={searchInputRef}
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full px-3 py-2.5 bg-warm-card border-b border-warm-border text-sm text-warm-text placeholder-warm-muted focus:outline-none focus:bg-warm-border/20 transition-colors duration-[250ms]"
            />
            <div className="max-h-[240px] overflow-y-auto">
              {filteredModels.length > 0 ? (
                filteredModels.map((m) => (
                  <ModelOption
                    key={m.id}
                    model={m}
                    onLoad={onLoad}
                    onDownload={onDownload}
                    onDelete={onDelete}
                    onClose={() => setOpen(false)}
                  />
                ))
              ) : (
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
