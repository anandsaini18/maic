interface Props {
  activeModel: string | null;
  ramGb: number;
  lastInference: string;
}

export default function StatusBar({
  activeModel,
  ramGb,
  lastInference,
}: Props) {
  const dotClass = activeModel
    ? "bg-green animate-pulse"
    : "bg-red";

  return (
    <div className="px-5 py-3 border-t border-warm-border bg-warm-bg/60 mt-auto">
      <div className="flex justify-between py-0.5">
        <span className="font-display text-[10px] text-warm-muted uppercase tracking-wider">
          Status
        </span>
        <span className="font-display text-[10px] text-warm-text">
          <span
            className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${dotClass}`}
          />
          {activeModel ? "Online" : "No model"}
        </span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="font-display text-[10px] text-warm-muted uppercase tracking-wider">
          RAM
        </span>
        <span className="font-display text-[10px] text-warm-text">
          {ramGb ? `${ramGb} GB` : "--"}
        </span>
      </div>
      <div className="flex justify-between py-0.5">
        <span className="font-display text-[10px] text-warm-muted uppercase tracking-wider">
          Last inference
        </span>
        <span className="font-display text-[10px] text-warm-text">
          {lastInference}
        </span>
      </div>
    </div>
  );
}
