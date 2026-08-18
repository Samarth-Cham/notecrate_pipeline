import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The app always calls the API on a relative path (/api/...). In dev this
// proxy forwards to uvicorn; in the container Nginx does the same job. Same
// origin either way, so there is no CORS layer to configure or get wrong.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
