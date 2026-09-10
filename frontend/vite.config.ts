import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the React app runs on :5173 and the API on :8000; `/api` is proxied so the
// browser sees one origin and no CORS preflight. Production is genuinely one
// origin - FastAPI serves the built `dist/` - so this proxy is dev-only.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
