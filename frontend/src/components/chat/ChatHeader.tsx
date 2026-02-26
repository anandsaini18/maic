import type { Theme } from "../../hooks/useTheme";

interface Props {
  activeModel: string | null;
  onClear: () => void;
  hasMessages: boolean;
  theme: Theme;
  onToggleTheme: () => void;
}

export default function ChatHeader({ activeModel, onClear, hasMessages, theme, onToggleTheme }: Props) {
  const badge = activeModel ? activeModel.split("/").pop() : null;

  return (
    <div className="flex items-center justify-between px-6 py-3.5 border-b border-warm-border bg-warm-bg/60 backdrop-blur-sm shrink-0">
      <div className="flex items-center gap-3">
        {/* Brand mark */}
        <div className="relative w-8 h-8 rounded-sm bg-gradient-to-br from-amber to-amber-muted flex items-center justify-center overflow-hidden">
          <span className="font-hero text-sm font-bold text-bg-primary tracking-tight">M</span>
          <div className="absolute inset-0 bg-gradient-to-t from-transparent to-white/10" />
        </div>
        <div className="flex flex-col">
          <h1 className="font-hero text-sm font-semibold tracking-wide text-warm-text">
            Maic
          </h1>
          <span className="text-[9px] font-display text-warm-muted tracking-[0.5px] -mt-0.5">
            MLX · LOCAL
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Theme toggle */}
        <button
          onClick={onToggleTheme}
          className="text-warm-muted hover:text-amber transition-colors duration-200 p-1.5 rounded-xs hover:bg-warm-card"
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? (
            /* Sun icon — shown in dark mode, click to go light */
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="5" />
              <line x1="12" y1="1" x2="12" y2="3" />
              <line x1="12" y1="21" x2="12" y2="23" />
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
              <line x1="1" y1="12" x2="3" y2="12" />
              <line x1="21" y1="12" x2="23" y2="12" />
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
            </svg>
          ) : (
            /* Moon icon — shown in light mode, click to go dark */
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
            </svg>
          )}
        </button>

        {/* Clear conversation */}
        {hasMessages && (
          <button
            onClick={onClear}
            className="text-warm-muted hover:text-warm-text transition-colors duration-200 p-1.5 rounded-xs hover:bg-warm-card"
            title="Clear conversation"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
            </svg>
          </button>
        )}

        {/* Model badge */}
        {badge ? (
          <div className="flex items-center gap-2 font-display text-[10px] text-amber bg-amber-dim px-2.5 py-1 rounded-full border border-amber-dim/50">
            <span className="w-1.5 h-1.5 rounded-full bg-green animate-pulse" />
            <span className="truncate max-w-[200px]">{badge}</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 font-display text-[10px] text-warm-muted bg-warm-card px-2.5 py-1 rounded-full border border-warm-border">
            <span className="w-1.5 h-1.5 rounded-full bg-warm-muted" />
            <span>no model</span>
          </div>
        )}
      </div>
    </div>
  );
}
