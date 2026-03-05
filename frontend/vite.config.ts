import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          highlight: ["highlight.js"],
          marked: ["marked"],
        },
      },
    },
  },
  server: {
    host: "localhost",
    port: 5173,
    proxy: {
      "/v1": "http://localhost:8001",
    },
  },
});
