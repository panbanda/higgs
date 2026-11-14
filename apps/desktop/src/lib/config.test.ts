import assert from "node:assert/strict";
import { test } from "node:test";
import { sanitizeConfig } from "./config.ts";

test("sanitizeConfig keeps empty values inside sections the form does not edit", () => {
  const mcp = {
    servers: {
      files: { command: "files-server", args: [""], env: { API_TOKEN: "" }, headers: {} },
    },
  };
  const sanitized = sanitizeConfig({ server: { port: 9000, host: "" }, mcp });
  assert.deepEqual(sanitized.mcp, mcp);
  assert.deepEqual(sanitized.server, { port: 9000 });
});

test("sanitizeConfig drops empty values the form manages", () => {
  const sanitized = sanitizeConfig({
    providers: { upstream: { base_url: "https://example.test", api_key: "" }, "": { base_url: "x" } },
    models: [{ path: "m", aliases: [] }],
    routes: [],
  });
  assert.deepEqual(sanitized, {
    providers: { upstream: { base_url: "https://example.test" } },
    models: [{ path: "m" }],
  });
});
