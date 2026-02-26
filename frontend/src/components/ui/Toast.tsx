import type { Toast as ToastData } from "../../hooks/useToast";

export default function Toast({ toast }: { toast: ToastData }) {
  const typeClass =
    toast.type === "error"
      ? "border-red-dim text-red"
      : toast.type === "success"
        ? "border-green-dim text-green"
        : "";

  return (
    <div
      className={`fixed bottom-6 left-1/2 -translate-x-1/2 bg-warm-card border border-warm-border rounded-sm px-5 py-2.5 font-display text-xs text-warm-text z-[10000] shadow-[0_8px_32px_rgba(0,0,0,0.4)] transition-all duration-[400ms] ease-spring ${
        toast.visible
          ? "opacity-100 translate-y-0"
          : "opacity-0 translate-y-5 pointer-events-none"
      } ${typeClass}`}
    >
      {toast.message}
    </div>
  );
}
