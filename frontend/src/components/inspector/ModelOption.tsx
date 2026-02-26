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
                className="px-2 py-1 rounded-xs border border-amber-dim bg-amber-dim text-amber font-display text-[10px] font-medium hover:bg-amber-glow active:scale-95 transition-all duration-[250ms] ease-spring"
                onClick={(e) => {
                  e.stopPropagation();
                  onLoad(m.id);
                  onClose();
                }}
                title="Activate"
              >
                ⚡
              </button>
              <button
                className="px-2 py-1 rounded-xs border border-red-dim text-red font-display text-[10px] font-medium hover:bg-red-dim active:scale-95 transition-all duration-[250ms] ease-spring"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(m.id);
                }}
                title="Delete"
              >
                🗑
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
              className="px-2 py-1 rounded-xs border border-amber-dim bg-amber-dim text-amber font-display text-[10px] font-medium hover:bg-amber-glow active:scale-95 transition-all duration-[250ms] ease-spring"
              onClick={(e) => {
                e.stopPropagation();
                onDownload(m.id);
              }}
              title="Download"
            >
              ⬇
            </button>
          )}
        </div>
      )}
    </div>
  );
}
