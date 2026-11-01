import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 1420, strictPort: true },
  build: { chunkSizeWarningLimit: 1000 },
});
