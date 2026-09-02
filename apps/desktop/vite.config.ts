import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { devLocalBridge } from "./dev/local-bridge";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react(), devLocalBridge() as Plugin],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
    cors: false,
  },
});
