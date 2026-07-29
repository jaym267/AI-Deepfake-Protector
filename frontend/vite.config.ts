import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port 5173 is not incidental: it is what the backend's CORS allowlist permits
// (backend/app/config.py, settings.cors_origins). Changing it here without
// changing that means every request fails preflight.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
