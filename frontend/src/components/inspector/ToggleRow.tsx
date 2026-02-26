interface Props {
  label: string;
  description?: string;
  value: boolean;
  onChange: (v: boolean) => void;
}

export default function ToggleRow({ label, description, value, onChange }: Props) {
  return (
    <div className="flex justify-between items-center py-2.5 border-t border-warm-border/50 first:border-0">
      <div className="flex flex-col gap-0.5">
        <span className="font-display text-[11px] text-warm-text">
          {label}
        </span>
        {description && (
          <span className="font-display text-[9px] text-warm-muted leading-relaxed">
            {description}
          </span>
        )}
      </div>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`relative w-9 h-5 rounded-[10px] border cursor-pointer transition-all duration-300 ease-spring shrink-0 ml-3 ${
          value
            ? "bg-amber-dim border-amber"
            : "bg-warm-card border-warm-border"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-3.5 h-3.5 rounded-full transition-all duration-300 ease-spring ${
            value
              ? "translate-x-4 bg-amber"
              : "translate-x-0 bg-warm-muted"
          }`}
        />
      </button>
    </div>
  );
}
