# Higgs Desktop

Tauri 2 + React dashboard and request-debugging chat for a running Higgs server.

```bash
pnpm install
pnpm tauri dev      # native window
pnpm dev            # browser at http://localhost:1420 (server needs cors_origins)
pnpm tauri build    # .app / .dmg under src-tauri/target/release/bundle/
```

Checks:

```bash
pnpm typecheck && pnpm build
cd src-tauri && cargo fmt --check && cargo clippy && cargo test
```

Layout:

- `src/App.tsx` -- shell: left rail, header with health, profile, daemon Start/Stop/Restart
- `src/hooks/useServerData.ts` -- one polling loop for /health, /v1/models, /metrics, the metrics JSONL log, config, and daemon state
- `src/views/` -- Overview, ModelsView, RoutingView, RequestsView, ChatView, ConfigView, SettingsView
- `src/components/RequestInspector.tsx` -- per-reply timeline, throughput, tokens, request JSON (Replay, Copy as curl), raw SSE chunks
- `src/components/ConfigForm.tsx` -- structured editor for config.toml sections
- `src/lib/chat.ts` -- streaming delta merge, trace capture, local tool loop
- `src/lib/dashboard.ts` -- percentile, per-minute, and group-by math ported from crates/higgs/src/metrics.rs
- `src/lib/tools.ts` -- built-in tools the app executes (`get_current_time`, `calculate`)
- `src-tauri/src/lib.rs` -- HTTP commands and SSE streaming with cancellation
- `src-tauri/src/local.rs` -- config files, metrics log tail, pid files, allowlisted `higgs` CLI runs, model cache size
- `dev/local-bridge.ts` -- Vite dev middleware mirroring `local.rs` so `pnpm dev` in a browser has local access

Metrics only exist when Higgs runs in config mode (`higgs start` or `higgs serve --config/--profile`);
`higgs serve --model ...` does not record them.
