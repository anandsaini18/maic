import type { Model } from "../../api/types";
import ModelTag from "./ModelTag";

interface Props {
  model: Model;
  onLoad: (id: string) => void;
  onDownload: (id: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

export default function ModelOption({
  model: m,
  onLoad,
  onDownload,
  onDelete,
  onClose,
}: Props) {
  const sizeLabel = m.disk_gb
    ? `${m.disk_gb} GB on disk`
    : `~${m.size_gb} GB`;

  return (
    <div
      className={`bg-warm-card border-b border-warm-border/50 p-3 cursor-pointer transition-all duration-200 ease-smooth flex justify-between items-start gap-3 last:border-b-0 hover:bg-warm-border/20 hover:border-l-[3px] hover:border-l-amber hover:pl-[9px] ${
        m.active
          ? "bg-amber-dim/30 border-l-[3px] border-l-amber pl-[9px]"
          : ""
      }`}
      onClick={() => {
        if (m.feasible && m.downloaded && !m.active) {
          onLoad(m.id);
          onClose();
        }
      }}
    >
      <div className="flex-1 min-w-0 flex flex-col gap-1">
        <div
          className="font-display text-[11px] font-medium text-warm-text truncate"
          title={m.id}
        >
          {m.name}
        </div>
        <div className="font-display text-[10px] text-warm-muted">
          {sizeLabel}
        </div>
        <div className="flex gap-1 flex-wrap mt-1">
          {m.active && <ModelTag variant="active">active</ModelTag>}
          {m.downloaded && !m.active && (
            <ModelTag variant="downloaded">ready</ModelTag>
          )}
          {m.downloading && (
            <ModelTag variant="downloading">downloading</ModelTag>
          )}
          {!m.downloaded && !m.downloading && (
            <ModelTag variant="not-downloaded">not downloaded</ModelTag>
          )}
          {m.requires_token && <ModelTag variant="token">token</ModelTag>}
          {!m.feasible && <ModelTag variant="too-large">too large</ModelTag>}
        </div>
      </div>

      {/* Action buttons */}
      {m.feasible && !m.active && (
        <div className="flex gap-1.5 shrink-0 mt-1">
          {m.downloaded ? (
            <>
              <button
                className="px-2 py-1.5 rounded-xs border border-amber-dim bg-amber-dim text-amber hover:bg-amber-glow active:scale-95 focus-visible:ring-2 focus-visible:ring-amber/40 focus-visible:outline-none transition-all duration-[250ms] ease-spring"
                onClick={(e) => {
                  e.stopPropagation();
                  onLoad(m.id);
                  onClose();
                }}
                title="Activate"
              >
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
              </button>
              <button
                className="px-2 py-1.5 rounded-xs border border-red-dim text-red hover:bg-red-dim active:scale-95 focus-visible:ring-2 focus-visible:ring-red/40 focus-visible:outline-none transition-all duration-[250ms] ease-spring"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(m.id);
                }}
                title="Delete"
              >
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
              </button>
            </>
          ) : m.downloading ? (
            <button
              className="px-2 py-1 rounded-xs border border-warm-border bg-warm-card text-warm-muted font-display text-[10px] font-medium opacity-50 cursor-not-allowed"
              disabled
            >
              …
            </button>
          ) : (
            <button
              className="px-2 py-1.5 rounded-xs border border-amber-dim bg-amber-dim text-amber hover:bg-amber-glow active:scale-95 focus-visible:ring-2 focus-visible:ring-amber/40 focus-visible:outline-none transition-all duration-[250ms] ease-spring"
              onClick={(e) => {
                e.stopPropagation();
                onDownload(m.id);
              }}
              title="Download"
            >
              <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
