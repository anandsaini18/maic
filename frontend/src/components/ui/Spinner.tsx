export default function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-block w-3.5 h-3.5 border-2 border-warm-border border-t-amber rounded-full animate-spin align-middle mr-1.5 ${className}`}
    />
  );
}
