/// <reference types="vite/client" />

/**
 * Per-session random token proving a browser-mode request actually came from
 * this Vite dev server's own page, not just from a loopback socket. Injected
 * by `vite.config.ts` via `define` from the token `dev/local-bridge.ts`
 * generates when its plugin is created; sent as the `X-Higgs-Bridge` header
 * by `src/lib/api.ts`'s `local()` and checked by the bridge middleware.
 */
declare const __HIGGS_BRIDGE_TOKEN__: string;
