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
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
        setSearch("");
      }
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (open) {
      const t = setTimeout(() => searchRef.current?.focus(), 50);
      return () => clearTimeout(t);
    } else {
      setSearch("");
    }
  }, [open]);

  const active = models.find((m) => m.id === activeModel);
  const sizeLabel = active?.disk_gb
    ? `${active.disk_gb} GB on disk`
    : active
      ? `~${active.size_gb} GB`
      : "";

  // Filter by search, then sort: active → downloaded → feasible → rest (all by size)
  const q = search.toLowerCase();
  const filtered = models
    .filter((m) => !q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q))
    .sort((a, b) => {
      if (a.active !== b.active) return a.active ? -1 : 1;
      if (a.downloaded !== b.downloaded) return a.downloaded ? -1 : 1;
      if (a.feasible !== b.feasible) return a.feasible ? -1 : 1;
      return a.size_gb - b.size_gb;
    });

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
            {/* Search */}
            <div className="p-2 border-b border-warm-border/60 sticky top-0 bg-warm-bg z-10">
              <input
                ref={searchRef}
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={`Search ${models.length} models…`}
                className="w-full font-display text-[11px] bg-warm-card border border-warm-border rounded-xs px-2.5 py-1.5 text-warm-text placeholder-warm-muted focus:outline-none focus:border-amber/50 transition-colors duration-200"
              />
            </div>

            {/* Model list */}
            <div className="overflow-y-auto max-h-[260px]">
              {filtered.length === 0 ? (
                <div className="p-4 text-center font-display text-[10px] text-warm-muted">
                  No models match "{search}"
                </div>
              ) : (
                filtered.map((m) => (
                  <ModelOption
                    key={m.id}
                    model={m}
                    onLoad={onLoad}
                    onDownload={onDownload}
                    onDelete={onDelete}
                    onClose={() => {
                      setOpen(false);
                      setSearch("");
                    }}
                  />
                ))
              )}
            </div>

            {/* Footer with count */}
            <div className="px-3 py-1.5 border-t border-warm-border/40 font-display text-[10px] text-warm-muted text-right">
              {filtered.length} of {models.length} models · mlx-community
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
