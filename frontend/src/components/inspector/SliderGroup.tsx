interface Props {
  label: string;
  description?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}

export default function SliderGroup({
  label,
  description,
  value,
  min,
  max,
  step,
  onChange,
}: Props) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-1.5">
        <span className="font-display text-[11px] text-warm-text">
          {label}
        </span>
        <span className="font-display text-[11px] text-amber bg-amber-dim px-1.5 py-px rounded min-w-[36px] text-center">
          {value}
        </span>
      </div>
      {description && (
        <p className="font-display text-[9px] text-warm-muted mb-2 leading-relaxed">
          {description}
        </p>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
