import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "../api/client";
import type { Model, ModelsStatus } from "../api/types";
import type { ToastType } from "./useToast";

interface UseModelsOpts {
  showToast: (msg: string, type?: ToastType) => void;
  clearConversation: () => void;
}

export function useModels({ showToast, clearConversation }: UseModelsOpts) {
  const [models, setModels] = useState<Model[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [ramGb, setRamGb] = useState<number>(0);
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const refresh = useCallback(async () => {
    try {
      const data: ModelsStatus = await api.fetchModelsStatus();
      setModels(data.models);
      setActiveModel(data.active_model);
      setRamGb(data.available_ram_gb);
    } catch {
      /* offline — silently skip */
    }
  }, []);

  /* Poll every 3 s, with an immediate first fetch. */
  useEffect(() => {
    pollRef.current = setInterval(refresh, 3000);
    /* Schedule first fetch outside the synchronous effect body. */
    const t = setTimeout(refresh, 0);
    return () => {
      clearInterval(pollRef.current);
      clearTimeout(t);
    };
  }, [refresh]);

  const download = useCallback(
    async (modelId: string) => {
      showToast(`Starting download: ${modelId.split("/").pop()}...`);
      try {
        await api.downloadModel(modelId);
      } catch (e: unknown) {
        showToast(`Download failed: ${(e as Error).message}`, "error");
      }
      refresh();
    },
    [showToast, refresh],
  );

  const load = useCallback(
    async (modelId: string) => {
      showToast(
        `Loading ${modelId.split("/").pop()}... this may take a moment.`,
      );
      try {
        await api.loadModel(modelId);
        showToast("Model activated!", "success");
        clearConversation();
      } catch (e: unknown) {
        showToast(`${(e as Error).message}`, "error");
      }
      refresh();
    },
    [showToast, refresh, clearConversation],
  );

  const remove = useCallback(
    async (modelId: string) => {
      if (
        !window.confirm(
          `Delete ${modelId.split("/").pop()} from disk? You'll need to re-download it later.`,
        )
      )
        return;
      try {
        await api.deleteModel(modelId);
        showToast("Model deleted.", "success");
      } catch (e: unknown) {
        showToast(`${(e as Error).message}`, "error");
      }
      refresh();
    },
    [showToast, refresh],
  );

  const quantize = useCallback(
    async (modelId: string, qBits = 4) => {
      showToast(`Quantizing ${modelId.split("/").pop()} to ${qBits}-bit...`);
      try {
        await api.quantizeModel(modelId, { q_bits: qBits });
      } catch (e: unknown) {
        showToast(`Quantize failed: ${(e as Error).message}`, "error");
      }
      refresh();
    },
    [showToast, refresh],
  );

  return { models, activeModel, ramGb, download, load, remove, quantize } as const;
}
