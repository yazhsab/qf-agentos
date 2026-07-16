import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = process.env.QF_API_URL ?? "http://127.0.0.1:8000";

// In dev, proxy the REST API routes to a running `qf-agent serve` instance so the
// SPA can call /studio/run, /jobs, /examples, /runs without CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ["/studio", "/jobs", "/examples", "/runs", "/backends", "/skills", "/healthz"].map((p) => [
        p,
        { target: API, changeOrigin: true },
      ]),
    ),
  },
});
