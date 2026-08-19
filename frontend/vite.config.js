import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The app always calls the API on a relative path (/api/...). In dev this
// proxy forwards to uvicorn; in the container Nginx does the same job. Same
// origin either way, so there is no CORS layer to configure or get wrong.
export default defineConfig(({ mode }) => {
  // loadEnv, not process.env: this reads frontend/.env (and .env.local), so
  // `npm run dev` works on its own instead of needing the caller to export a
  // variable — which is shell-specific and easy to get wrong on Windows.
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_TARGET || "http://127.0.0.1:8000";

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api/, ""),
        },
      },
    },
  };
});
