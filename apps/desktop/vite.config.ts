import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { devLocalBridge } from "./dev/local-bridge";

const host = process.env.TAURI_DEV_HOST;

const bridge = devLocalBridge();

export default defineConfig({
  plugins: [react(), bridge as Plugin],
  // Exposed to the page as `__HIGGS_BRIDGE_TOKEN__` (see
  // src/vite-env.d.ts); `src/lib/api.ts` sends it back as the
  // `X-Higgs-Bridge` header so the bridge middleware can tell this page's
  // own requests apart from any other localhost-bound request.
  define: { __HIGGS_BRIDGE_TOKEN__: JSON.stringify(bridge.token) },
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
