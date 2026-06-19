import { useCallback, useRef, useState } from "react";

export type ToastType = "" | "error" | "success";

export interface Toast {
  message: string;
  type: ToastType;
  visible: boolean;
}

export function useToast() {
  const [toast, setToast] = useState<Toast>({
    message: "",
    type: "",
    visible: false,
  });
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const showToast = useCallback((message: string, type: ToastType = "") => {
    clearTimeout(timerRef.current);
    setToast({ message, type, visible: true });
    timerRef.current = setTimeout(
      () => setToast((t) => ({ ...t, visible: false })),
      3000,
    );
  }, []);

  return { toast, showToast } as const;
}
