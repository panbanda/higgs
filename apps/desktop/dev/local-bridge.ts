/**
 * Dev-only stand-in for the Rust local commands in src-tauri/src/local.rs so
 * the browser build (`pnpm dev`) can read config, the metrics log, and run
 * the CLI. Endpoints are POST /__local/<command> with a JSON body of the
 * same camelCase args the Tauri commands take. Never served by `vite build`.
 */
import crypto from "node:crypto";
import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";
import type { Plugin } from "vite";

type Handler = (args: Record<string, unknown>) => Promise<unknown>;

const configDir = () => process.env.HIGGS_CONFIG_DIR ?? path.join(os.homedir(), ".config/higgs");
const expandHome = (p: string) => (p.startsWith("~/") ? path.join(os.homedir(), p.slice(2)) : p);

/** Thrown by path/command guards; mapped to an HTTP 403 by the middleware. */
class ForbiddenError extends Error {}

/** True when `candidate` is `dir` itself or nested somewhere inside it,
 * purely lexically (no filesystem access). */
function withinDir(candidate: string, dir: string): boolean {
  const resolvedCandidate = path.resolve(candidate);
  const resolvedDir = path.resolve(dir);
  const relative = path.relative(resolvedDir, resolvedCandidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

/**
 * Symlink-safe containment check mirroring `is_contained` in
 * src-tauri/src/paths.rs: true when `candidate` lexically resolves under
 * `root` AND no existing ancestor directory between `root` and `candidate`
 * is itself a symlink, so a symlinked directory placed inside `root` cannot
 * redirect a read or write outside it.
 *
 * Deliberately does not inspect `candidate` itself: callers that create or
 * replace a symlink at `candidate` (the Hugging Face cache's blob links)
 * must be able to do so even when a previous, legitimately-contained
 * symlink already sits there.
 *
 * When `root` does not exist on disk yet, only the lexical check applies,
 * since there is nothing to canonicalize.
 */
async function isContained(root: string, candidate: string): Promise<boolean> {
  if (!withinDir(candidate, root)) return false;
  const resolvedRoot = path.resolve(root);
  const resolvedCandidate = path.resolve(candidate);
  let canonicalRoot: string;
  try {
    canonicalRoot = await fs.realpath(resolvedRoot);
  } catch {
    return true;
  }

  // Walk every ancestor directory from `candidate`'s parent up to `root`,
  // rejecting any that is itself a symlink. `lstat` reports the *last* path
  // component's own symlink-ness without following it, even though earlier
  // components are still resolved transparently by the OS, so this catches
  // a symlink at any depth once the walk reaches it as the final segment
  // being inspected.
  let dir = path.dirname(resolvedCandidate);
  let nearestExisting: string | null = null;
  while (dir !== resolvedRoot) {
    try {
      const stats = await fs.lstat(dir);
      if (stats.isSymbolicLink()) return false;
      if (nearestExisting === null) nearestExisting = dir;
    } catch {
      // does not exist; keep walking up
    }
    const parent = path.dirname(dir);
    if (parent === dir) break; // reached the filesystem root without matching `root`
    dir = parent;
  }

  const nearest = nearestExisting ?? resolvedRoot;
  try {
    const resolvedNearest = await fs.realpath(nearest);
    return resolvedNearest === canonicalRoot || withinDir(resolvedNearest, canonicalRoot);
  } catch {
    return false;
  }
}

/**
 * `logging.metrics.path` from every config file in the config dir, resolved
 * to an absolute path. Used to allow `read_metrics_log` to read a metrics
 * log configured outside the config dir (still constrained to the user's
 * home directory).
 */
async function configuredMetricsLogPaths(): Promise<Set<string>> {
  const dir = configDir();
  const paths = new Set<string>();
  let entries: string[];
  try {
    entries = await fs.readdir(dir);
  } catch {
    return paths;
  }
  for (const name of entries) {
    if (name !== "config.toml" && !/^config\..+\.toml$/.test(name)) continue;
    try {
      const raw = await fs.readFile(path.join(dir, name), "utf8");
      const parsed = parseToml(raw) as { logging?: { metrics?: { path?: unknown } } };
      const metricsPath = parsed.logging?.metrics?.path;
      if (typeof metricsPath === "string" && metricsPath) {
        paths.add(path.resolve(expandHome(metricsPath)));
      }
    } catch {
      // ignore unreadable or invalid config files
    }
  }
  return paths;
}

/**
 * Every handler that reads/writes a filesystem path supplied by the caller
 * must go through here first: the path must resolve inside the Higgs config
 * dir, except `read_metrics_log`, which may also point at whatever metrics
 * log path is configured, as long as that still resolves under the user's
 * home directory.
 */
async function assertAllowedPath(command: string, args: Record<string, unknown>): Promise<void> {
  const pathCommands = new Set(["read_config", "write_config_raw", "write_config_structured", "read_text_tail", "read_metrics_log"]);
  if (command === "daemon_status") {
    const profile = args.profile;
    if (profile != null && (typeof profile !== "string" || !/^[A-Za-z0-9_-]+$/.test(profile))) {
      throw new ForbiddenError("invalid profile name");
    }
    return;
  }
  if (!pathCommands.has(command)) return;
  const raw = args.path;
  if (typeof raw !== "string" || !raw) throw new ForbiddenError("path required");
  const resolved = path.resolve(expandHome(raw));
  const dir = configDir();
  if (await isContained(dir, resolved)) return;
  if (command === "read_metrics_log" && (await isContained(os.homedir(), resolved))) {
    const allowed = await configuredMetricsLogPaths();
    if (allowed.has(resolved)) return;
  }
  throw new ForbiddenError(`path outside the Higgs config dir: ${resolved}`);
}

async function exists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function dirSize(dir: string): Promise<number> {
  let total = 0;
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) total += await dirSize(full);
    else if (entry.isFile()) total += (await fs.stat(full)).size;
  }
  return total;
}

function run(program: string, args: string[]): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    execFile(program, args, { env: { ...process.env, NO_COLOR: "1" }, maxBuffer: 8 * 1024 * 1024 }, (error, stdout, stderr) => {
      const code = error && typeof (error as NodeJS.ErrnoException & { code?: number }).code === "number" ? ((error as { code?: number }).code ?? 1) : error ? 1 : 0;
      resolve({ code, stdout: String(stdout), stderr: String(stderr).replace(/\x1b\[[0-9;]*[A-Za-z]/g, "") });
    });
  });
}

/// Only whitelisted subcommands run from the UI so the bridge cannot be
/// turned into a general shell.
const ALLOWED_SUBCOMMANDS = new Set(["doctor", "start", "stop", "config", "--version"]);

/** Mirrors `is_profile_name` in src-tauri/src/local.rs. */
function isProfileName(value: string): boolean {
  return value.length > 0 && /^[A-Za-z0-9._-]+$/.test(value);
}

/**
 * Validates a trailing run of `--profile <name>` / `--config <path>`
 * selector pairs, the only flags `doctor`, `start`, `stop`, and `config`
 * accept beyond their own subcommand arguments. Mirrors
 * `validate_selector_args` in src-tauri/src/local.rs.
 */
async function assertValidSelectorArgs(rest: string[]): Promise<void> {
  let i = 0;
  while (i < rest.length) {
    const flag = rest[i];
    if (flag === "--profile") {
      const name = rest[i + 1];
      if (name === undefined) throw new Error("--profile requires a value");
      if (!isProfileName(name)) throw new Error(`invalid profile name: ${name}`);
      i += 2;
    } else if (flag === "--config") {
      const configPath = rest[i + 1];
      if (configPath === undefined) throw new Error("--config requires a value");
      const resolved = path.resolve(expandHome(configPath));
      if (!(await isContained(configDir(), resolved))) {
        throw new Error(`path ${resolved} is outside the Higgs config directory`);
      }
      i += 2;
    } else {
      throw new Error(`unexpected argument: ${flag}`);
    }
  }
}

/** Mirrors `validate_config_subcommand_args` in src-tauri/src/local.rs. */
async function assertValidConfigSubcommandArgs(rest: string[]): Promise<void> {
  const sub = rest[0];
  if (sub === "get") {
    const key = rest[1];
    if (key === undefined) throw new Error("`config get` requires a key");
    if (key.startsWith("--")) throw new Error(`unexpected argument in place of a key: ${key}`);
    await assertValidSelectorArgs(rest.slice(2));
  } else if (sub === "set") {
    const key = rest[1];
    const value = rest[2];
    if (key === undefined) throw new Error("`config set` requires a key");
    if (value === undefined) throw new Error("`config set` requires a value");
    if (key.startsWith("--")) throw new Error(`unexpected argument in place of a key: ${key}`);
    await assertValidSelectorArgs(rest.slice(3));
  } else if (sub === "path") {
    await assertValidSelectorArgs(rest.slice(1));
  } else if (sub === undefined) {
    throw new Error("`config` requires a subcommand");
  } else {
    throw new Error(`unexpected config subcommand: ${sub}`);
  }
}

/**
 * Validates every argument after the subcommand, so the bridge cannot be
 * used to smuggle arbitrary flags to the `higgs` binary. Mirrors
 * `validate_subcommand_args` in src-tauri/src/local.rs so the two backends
 * accept exactly the same argument shapes: `doctor`, `start`, and `stop`
 * accept only `--profile <name>` / `--config <path>` selectors; `config`
 * additionally accepts `get <key>`, `set <key> <value>`, or `path` before
 * those same selectors; `--version` accepts nothing else.
 */
async function assertValidSubcommandArgs(subcommand: string, rest: string[]): Promise<void> {
  switch (subcommand) {
    case "doctor":
    case "start":
    case "stop":
      await assertValidSelectorArgs(rest);
      return;
    case "config":
      await assertValidConfigSubcommandArgs(rest);
      return;
    case "--version":
      if (rest.length !== 0) throw new Error("--version accepts no further arguments");
      return;
    default:
      throw new Error(`subcommand not allowed: ${subcommand}`);
  }
}

// Hugging Face hub browsing, mirroring src-tauri/src/hub.rs so the dev
// bridge exposes the same command surface as the desktop app.

const HUB_BASE = "https://huggingface.co";
const DEFAULT_HUB_AUTHOR = "mlx-community";

interface HubModelSummary {
  id: string;
  downloads: number;
  likes: number;
  last_modified: string | null;
  tags: string[];
  gated: boolean;
}

interface HubSibling {
  rfilename: string;
  size: number | null;
}

interface HubModelDetail {
  id: string;
  sha: string;
  siblings: HubSibling[];
  total_bytes: number;
  config_model_type: string | null;
  quantization: string | null;
  tags: string[];
}

interface HubDownloadStatus {
  state: "idle" | "running" | "done" | "error" | "cancelled";
  file: string | null;
  file_index: number;
  file_count: number;
  bytes_done: number;
  bytes_total: number;
  total_done: number;
  total_bytes: number;
  message: string | null;
  path: string | null;
}

function defaultHubStatus(): HubDownloadStatus {
  return {
    state: "idle",
    file: null,
    file_index: 0,
    file_count: 0,
    bytes_done: 0,
    bytes_total: 0,
    total_done: 0,
    total_bytes: 0,
    message: null,
    path: null,
  };
}

interface HubJob {
  status: HubDownloadStatus;
  controller: AbortController;
  incompletePath: string | null;
}

const hubJobs = new Map<string, HubJob>();

function hubAuthHeaders(token: unknown): Record<string, string> {
  return typeof token === "string" && token ? { Authorization: `Bearer ${token}` } : {};
}

async function hubErrorBody(response: Response): Promise<string> {
  let message: string;
  if (response.status === 401) message = "Unauthorized: the Hugging Face token was rejected";
  else if (response.status === 403) message = "Forbidden: this repo is gated or private; a token with access is required";
  else if (response.status === 429) message = "Rate limited by huggingface.co; wait a moment and try again";
  else message = await response.text();
  return `HTTP ${response.status}: ${message}`;
}

function hubRepoDirName(repo: string): string {
  return `models--${repo.replace(/\//g, "--")}`;
}

/** Mirrors `is_valid_repo_id` in src-tauri/src/hub.rs. */
const REPO_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** Mirrors `is_valid_token` in src-tauri/src/hub.rs: a sha or an etag. */
const HUB_TOKEN_PATTERN = /^[A-Za-z0-9._-]+$/;

function assertValidRepoId(repo: string): void {
  if (!REPO_ID_PATTERN.test(repo)) throw new ForbiddenError(`invalid repo id: ${repo}`);
}

function assertValidHubToken(value: string, label: string): void {
  if (!HUB_TOKEN_PATTERN.test(value)) throw new ForbiddenError(`invalid ${label}: ${value}`);
}

/** Mirrors `is_safe_relative_path` in src-tauri/src/hub.rs: a repo-listed
 * `rfilename` must be a relative path made entirely of normal components. */
function assertSafeRelativePath(rfilename: string): void {
  if (rfilename.length === 0 || path.isAbsolute(rfilename)) {
    throw new ForbiddenError(`invalid rfilename: ${rfilename}`);
  }
  for (const part of rfilename.split("/")) {
    if (part === "" || part === "." || part === "..") {
      throw new ForbiddenError(`invalid rfilename: ${rfilename}`);
    }
  }
}

/** Defense in depth: confirms a joined path actually landed inside `dir`,
 * even accounting for a symlink placed somewhere in the cache layout. */
async function assertWithinDir(candidate: string, dir: string): Promise<void> {
  if (!(await isContained(dir, candidate))) throw new ForbiddenError(`path escaped its directory: ${candidate}`);
}

function hubCacheRoot(): string {
  return process.env.HF_HOME ? path.join(process.env.HF_HOME, "hub") : path.join(os.homedir(), ".cache/huggingface/hub");
}

const QUANT_PATTERNS: Array<[string, string]> = [
  ["4-bit", "4-bit"],
  ["4bit", "4-bit"],
  ["8-bit", "8-bit"],
  ["8bit", "8-bit"],
  ["6-bit", "6-bit"],
  ["6bit", "6-bit"],
  ["3-bit", "3-bit"],
  ["3bit", "3-bit"],
  ["2-bit", "2-bit"],
  ["2bit", "2-bit"],
  ["bf16", "bf16"],
  ["fp16", "fp16"],
  ["fp32", "fp32"],
];

function hubQuantizationHint(id: string, tags: string[]): string | null {
  for (const haystack of [id, ...tags]) {
    const lower = haystack.toLowerCase();
    const match = QUANT_PATTERNS.find(([needle]) => lower.includes(needle));
    if (match) return match[1];
  }
  return null;
}

interface RawHubModelDetail {
  id: string;
  sha: string;
  siblings?: HubSibling[];
  tags?: string[];
}

async function hubFetchModelDetail(repo: string, token: unknown): Promise<RawHubModelDetail> {
  const response = await fetch(`${HUB_BASE}/api/models/${repo}?blobs=true`, { headers: hubAuthHeaders(token) });
  if (!response.ok) throw new Error(await hubErrorBody(response));
  return (await response.json()) as RawHubModelDetail;
}

async function hubFetchConfigModelType(repo: string, sha: string, token: unknown): Promise<string | null> {
  try {
    const response = await fetch(`${HUB_BASE}/${repo}/resolve/${sha}/config.json`, { headers: hubAuthHeaders(token) });
    if (!response.ok) return null;
    const value = (await response.json()) as { model_type?: unknown };
    return typeof value.model_type === "string" ? value.model_type : null;
  } catch {
    return null;
  }
}

async function hubResolveEtag(fileUrl: string, token: unknown, sha: string, rfilename: string): Promise<string> {
  const response = await fetch(fileUrl, { method: "HEAD", redirect: "manual", headers: hubAuthHeaders(token) });
  const etag = response.headers.get("x-linked-etag") ?? response.headers.get("etag");
  return etag ? etag.replaceAll('"', "") : `${sha}-${rfilename.replaceAll("/", "_")}`;
}

async function hubLinkSnapshotFile(snapshotDir: string, rfilename: string, blobPath: string): Promise<void> {
  assertSafeRelativePath(rfilename);
  const linkPath = path.join(snapshotDir, rfilename);
  await assertWithinDir(linkPath, snapshotDir);
  await fs.mkdir(path.dirname(linkPath), { recursive: true });
  try {
    await fs.unlink(linkPath);
  } catch {
    // no existing link to remove
  }
  await fs.symlink(blobPath, linkPath);
}

async function hubDownloadToBlob(
  fileUrl: string,
  token: unknown,
  blobPath: string,
  job: HubJob,
  onProgress: (fileDone: number) => void,
): Promise<void> {
  const tempPath = `${blobPath}.incomplete`;
  job.incompletePath = tempPath;
  const response = await fetch(fileUrl, { headers: hubAuthHeaders(token), signal: job.controller.signal });
  if (!response.ok || !response.body) throw new Error(await hubErrorBody(response));
  const handle = await fs.open(tempPath, "w");
  let fileDone = 0;
  try {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      await handle.write(value);
      fileDone += value.byteLength;
      onProgress(fileDone);
    }
  } finally {
    await handle.close();
  }
  await fs.rename(tempPath, blobPath);
  job.incompletePath = null;
}

async function hubRunDownload(repo: string, token: unknown, job: HubJob): Promise<void> {
  try {
    assertValidRepoId(repo);
    const detail = await hubFetchModelDetail(repo, token);
    assertValidHubToken(detail.sha, "sha");
    const root = hubCacheRoot();
    const repoDir = path.join(root, hubRepoDirName(repo));
    await assertWithinDir(repoDir, root);
    const blobsDir = path.join(repoDir, "blobs");
    const snapshotDir = path.join(repoDir, "snapshots", detail.sha);
    const refsDir = path.join(repoDir, "refs");
    await fs.mkdir(blobsDir, { recursive: true });
    await fs.mkdir(snapshotDir, { recursive: true });
    await fs.mkdir(refsDir, { recursive: true });
    await fs.writeFile(path.join(refsDir, "main"), detail.sha);

    const siblings = detail.siblings ?? [];
    const totalBytes = siblings.reduce((sum, s) => sum + (s.size ?? 0), 0);
    let totalDone = 0;

    for (let index = 0; index < siblings.length; index += 1) {
      const sibling = siblings[index];
      assertSafeRelativePath(sibling.rfilename);
      job.status.file = sibling.rfilename;
      job.status.file_index = index + 1;
      job.status.file_count = siblings.length;
      job.status.bytes_done = 0;
      job.status.bytes_total = sibling.size ?? 0;
      job.status.total_done = totalDone;
      job.status.total_bytes = totalBytes;

      const fileUrl = `${HUB_BASE}/${repo}/resolve/${detail.sha}/${sibling.rfilename}`;
      const etag = await hubResolveEtag(fileUrl, token, detail.sha, sibling.rfilename);
      assertValidHubToken(etag, "etag");
      const blobPath = path.join(blobsDir, etag);
      await assertWithinDir(blobPath, blobsDir);

      const upToDate =
        (await exists(blobPath)) && (sibling.size == null || (await fs.stat(blobPath)).size === sibling.size);
      if (!upToDate) {
        const baseDone = totalDone;
        await hubDownloadToBlob(fileUrl, token, blobPath, job, (fileDone) => {
          job.status.bytes_done = fileDone;
          job.status.total_done = baseDone + fileDone;
        });
        totalDone = baseDone + (sibling.size ?? (await fs.stat(blobPath)).size);
      } else {
        totalDone += sibling.size ?? 0;
      }

      await hubLinkSnapshotFile(snapshotDir, sibling.rfilename, blobPath);
    }

    job.status.state = "done";
    job.status.path = snapshotDir;
  } catch (error) {
    if (job.controller.signal.aborted) {
      job.status.state = "cancelled";
      if (job.incompletePath) {
        await fs.unlink(job.incompletePath).catch(() => {
          /* nothing to clean up */
        });
      }
    } else {
      job.status.state = "error";
      job.status.message = String(error);
    }
  }
}

const handlers: Record<string, Handler> = {
  async list_profiles() {
    const dir = configDir();
    const profiles: Array<{ name: string | null; config_path: string }> = [{ name: null, config_path: path.join(dir, "config.toml") }];
    try {
      for (const name of (await fs.readdir(dir)).sort()) {
        const match = /^config\.(.+)\.toml$/.exec(name);
        if (match) profiles.push({ name: match[1], config_path: path.join(dir, name) });
      }
    } catch {
      // missing config dir is fine
    }
    return { config_dir: dir, profiles };
  },
  async read_config({ path: p }) {
    const file = expandHome(String(p));
    let raw = "";
    let fileExists = true;
    try {
      raw = await fs.readFile(file, "utf8");
    } catch {
      fileExists = false;
    }
    try {
      return { path: file, exists: fileExists, raw, parsed: fileExists ? parseToml(raw) : null, parse_error: null };
    } catch (error) {
      return { path: file, exists: fileExists, raw, parsed: null, parse_error: String(error) };
    }
  },
  async write_config_raw({ path: p, raw }) {
    parseToml(String(raw));
    const file = expandHome(String(p));
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, String(raw), { mode: 0o600 });
    return null;
  },
  async write_config_structured({ path: p, config }) {
    const raw = stringifyToml(config as Record<string, unknown>);
    const file = expandHome(String(p));
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, raw, { mode: 0o600 });
    return raw;
  },
  async read_metrics_log({ path: p, maxRecords, sinceOffset }) {
    const file = expandHome(String(p));
    if (!(await exists(file))) return { path: file, exists: false, records: [], offset: 0, reset: sinceOffset != null };
    const length = (await fs.stat(file)).size;
    const since = typeof sinceOffset === "number" ? sinceOffset : null;
    const reset = since !== null && since > length;
    const tail = 4 * 1024 * 1024;
    const start = since !== null && !reset ? since : Math.max(0, length - tail);
    const handle = await fs.open(file, "r");
    const buffer = Buffer.alloc(length - start);
    await handle.read(buffer, 0, buffer.length, start);
    await handle.close();
    const text = buffer.toString("utf8");
    const lastNewline = text.lastIndexOf("\n");
    const complete = lastNewline === -1 ? "" : text.slice(0, lastNewline + 1);
    let lines = complete.split("\n").filter(Boolean);
    if (since === null && start > 0) lines = lines.slice(1);
    const records = lines
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
    const max = Number(maxRecords) || 5000;
    return { path: file, exists: true, records: records.slice(-max), offset: start + Buffer.byteLength(complete), reset };
  },
  async daemon_status({ profile }) {
    const dir = configDir();
    const name = typeof profile === "string" ? profile : null;
    const pidPath = path.join(dir, name ? `higgs.${name}.pid` : "higgs.pid");
    const logPath = path.join(dir, name ? `higgs.${name}.log` : "higgs.log");
    let pid: number | null = null;
    try {
      pid = Number.parseInt((await fs.readFile(pidPath, "utf8")).trim(), 10) || null;
    } catch {
      pid = null;
    }
    let running = false;
    if (pid) {
      try {
        process.kill(pid, 0);
        running = true;
      } catch {
        running = false;
      }
    }
    return { running, pid, pid_path: pidPath, log_path: logPath };
  },
  async read_text_tail({ path: p, maxBytes }) {
    const file = expandHome(String(p));
    const length = (await fs.stat(file)).size;
    const start = Math.max(0, length - Number(maxBytes));
    const handle = await fs.open(file, "r");
    const buffer = Buffer.alloc(length - start);
    await handle.read(buffer, 0, buffer.length, start);
    await handle.close();
    return buffer.toString("utf8");
  },
  async run_higgs({ binary, args }) {
    const list = Array.isArray(args) ? args.map(String) : [];
    const subcommand = list[0] ?? "";
    if (!ALLOWED_SUBCOMMANDS.has(subcommand)) throw new Error(`subcommand not allowed: ${subcommand}`);
    await assertValidSubcommandArgs(subcommand, list.slice(1));
    let program = typeof binary === "string" && binary.trim() ? binary : "";
    if (program) {
      const resolvedProgram = path.resolve(expandHome(program));
      if (path.basename(resolvedProgram) !== "higgs") {
        throw new Error(`binary must resolve to a file named \`higgs\`: ${resolvedProgram}`);
      }
      let stat;
      try {
        stat = await fs.stat(resolvedProgram);
      } catch {
        throw new Error(`binary not found: ${resolvedProgram}`);
      }
      if (!stat.isFile()) throw new Error(`binary is not a file: ${resolvedProgram}`);
      program = resolvedProgram;
    } else {
      const found = await run("/bin/zsh", ["-lc", "command -v higgs"]);
      program = found.stdout.trim();
      if (!program) throw new Error("could not find `higgs` on PATH; set the binary path in Settings");
    }
    const result = await run(program, list);
    return { program, exit_code: result.code, stdout: result.stdout, stderr: result.stderr };
  },
  // Mirrors src-tauri/src/local.rs's `higgs_binary_info`, minus the
  // "bundled" outcome: the dev bridge has no app bundle to fall back to, so
  // it only ever reports "settings", "path", or "missing".
  async higgs_binary_info({ binary }) {
    const missing = { path: null, source: "missing" as const, version: null };
    let program = typeof binary === "string" && binary.trim() ? binary : "";
    let source: "settings" | "path" = "path";
    if (program) {
      source = "settings";
      const resolvedProgram = path.resolve(expandHome(program));
      if (path.basename(resolvedProgram) !== "higgs") return missing;
      try {
        const stat = await fs.stat(resolvedProgram);
        if (!stat.isFile()) return missing;
      } catch {
        return missing;
      }
      program = resolvedProgram;
    } else {
      const found = await run("/bin/zsh", ["-lc", "command -v higgs"]);
      program = found.stdout.trim();
      if (!program) return missing;
    }
    const result = await run(program, ["--version"]);
    const version = result.code === 0 ? result.stdout.trim() || result.stderr.trim() || null : null;
    return { path: program, source, version };
  },
  async model_cache_info({ path: p }) {
    const model = String(p);
    const direct = expandHome(model);
    if (await exists(direct)) return { path: model, cached: true, size_bytes: await dirSize(direct), location: direct };
    const hub = process.env.HF_HOME ? path.join(process.env.HF_HOME, "hub") : path.join(os.homedir(), ".cache/huggingface/hub");
    const repo = path.join(hub, `models--${model.replace(/\//g, "--")}`);
    if (await exists(repo)) return { path: model, cached: true, size_bytes: await dirSize(path.join(repo, "blobs")), location: repo };
    return { path: model, cached: false, size_bytes: 0, location: null };
  },
  async hub_search({ query, author, pipelineTag, token, limit }) {
    const params = new URLSearchParams();
    params.set("sort", "downloads");
    params.set("direction", "-1");
    params.set("limit", String(limit ?? 30));
    for (const field of ["downloads", "likes", "lastModified", "tags", "gated"]) params.append("expand[]", field);
    const resolvedAuthor = typeof author === "string" ? author : DEFAULT_HUB_AUTHOR;
    if (resolvedAuthor) params.set("author", resolvedAuthor);
    if (typeof query === "string" && query.trim()) params.set("search", query.trim());
    if (typeof pipelineTag === "string" && pipelineTag) params.set("pipeline_tag", pipelineTag);
    const response = await fetch(`${HUB_BASE}/api/models?${params.toString()}`, { headers: hubAuthHeaders(token) });
    if (!response.ok) throw new Error(await hubErrorBody(response));
    const raw = (await response.json()) as Array<{
      id: string;
      downloads?: number;
      likes?: number;
      lastModified?: string | null;
      tags?: string[];
      gated?: unknown;
    }>;
    const results: HubModelSummary[] = raw.map((item) => ({
      id: item.id,
      downloads: item.downloads ?? 0,
      likes: item.likes ?? 0,
      last_modified: item.lastModified ?? null,
      tags: item.tags ?? [],
      gated: item.gated !== false && item.gated != null,
    }));
    return results;
  },
  async hub_model({ repo, token }) {
    const detail = await hubFetchModelDetail(String(repo), token);
    const siblings = detail.siblings ?? [];
    const totalBytes = siblings.reduce((sum, s) => sum + (s.size ?? 0), 0);
    const configModelType = await hubFetchConfigModelType(String(repo), detail.sha, token);
    const result: HubModelDetail = {
      id: detail.id,
      sha: detail.sha,
      siblings,
      total_bytes: totalBytes,
      config_model_type: configModelType,
      quantization: hubQuantizationHint(detail.id, detail.tags ?? []),
      tags: detail.tags ?? [],
    };
    return result;
  },
  async hub_download_start({ repo, token }) {
    const key = String(repo);
    const existing = hubJobs.get(key);
    if (existing && existing.status.state === "running") return null;
    const job: HubJob = { status: defaultHubStatus(), controller: new AbortController(), incompletePath: null };
    job.status.state = "running";
    hubJobs.set(key, job);
    void hubRunDownload(key, token, job);
    return null;
  },
  async hub_download_status({ repo }) {
    return hubJobs.get(String(repo))?.status ?? defaultHubStatus();
  },
  async hub_cancel({ repo }) {
    const job = hubJobs.get(String(repo));
    job?.controller.abort();
    return null;
  },
  async hub_delete({ repo }) {
    const repoId = String(repo);
    assertValidRepoId(repoId);
    const root = hubCacheRoot();
    const repoDir = path.join(root, hubRepoDirName(repoId));
    await assertWithinDir(repoDir, root);
    await fs.rm(repoDir, { recursive: true, force: true });
    return null;
  },
};

/**
 * True for the loopback addresses Node reports on `socket.remoteAddress`:
 * the raw IPv4/IPv6 forms and the IPv4-mapped IPv6 form a dual-stack socket
 * can report. Unlike the `Host` header (client-supplied and trivially
 * spoofed), this comes from the OS's own view of the connection.
 */
function isLoopbackAddress(address: string | undefined): boolean {
  if (!address) return false;
  if (address === "127.0.0.1" || address === "::1") return true;
  return address === "::ffff:127.0.0.1";
}

const MAX_BODY_BYTES = 1024 * 1024;

export function devLocalBridge(): Plugin & { token: string } {
  // A fresh random token per dev-server run: exposed to the page via Vite's
  // `define` (see vite.config.ts) as `__HIGGS_BRIDGE_TOKEN__` and sent back
  // as the `X-Higgs-Bridge` header, so only this page's own tab can call the
  // bridge even when the loopback check above is satisfied (e.g. another
  // local process, or a page from a different origin bound to localhost).
  const token = crypto.randomBytes(32).toString("hex");
  return {
    name: "higgs-dev-local-bridge",
    apply: "serve",
    token,
    configureServer(server) {
      server.middlewares.use("/__local", (request, response) => {
        const command = (request.url ?? "/").replace(/^\/+/, "").split("?")[0];
        const handler = handlers[command];
        if (request.method !== "POST" || !handler) {
          response.statusCode = 404;
          response.end(JSON.stringify({ error: `unknown local command: ${command}` }));
          return;
        }

        // When Vite is started with TAURI_DEV_HOST set, the dev server binds
        // to an external interface; refuse to serve local commands to a
        // connection that didn't actually arrive over loopback, regardless
        // of what the (client-controlled) Host header claims.
        if (!isLoopbackAddress(request.socket.remoteAddress)) {
          response.statusCode = 403;
          response.end(JSON.stringify({ error: "forbidden: non-loopback connection" }));
          return;
        }

        const bridgeToken = request.headers["x-higgs-bridge"];
        if (bridgeToken !== token) {
          response.statusCode = 401;
          response.end(JSON.stringify({ error: "unauthorized: missing or invalid bridge token" }));
          return;
        }

        const origin = request.headers.origin;
        if (typeof origin === "string" && origin) {
          const expectedOrigin = `http://${request.headers.host ?? ""}`;
          if (origin !== expectedOrigin) {
            response.statusCode = 403;
            response.end(JSON.stringify({ error: "forbidden: origin mismatch" }));
            return;
          }
        }
        const secFetchSite = request.headers["sec-fetch-site"];
        if (typeof secFetchSite === "string" && secFetchSite && secFetchSite !== "same-origin" && secFetchSite !== "none") {
          response.statusCode = 403;
          response.end(JSON.stringify({ error: "forbidden: cross-site request" }));
          return;
        }

        let body = "";
        let bodyBytes = 0;
        let rejected = false;
        request.on("data", (chunk: Buffer) => {
          if (rejected) return;
          bodyBytes += chunk.length;
          if (bodyBytes > MAX_BODY_BYTES) {
            rejected = true;
            response.statusCode = 413;
            response.end(JSON.stringify({ error: "payload too large" }));
            request.destroy();
            return;
          }
          body += chunk;
        });
        request.on("end", () => {
          if (rejected) return;
          void (async () => {
            try {
              const args = body ? (JSON.parse(body) as Record<string, unknown>) : {};
              await assertAllowedPath(command, args);
              const result = await handler(args);
              response.setHeader("Content-Type", "application/json");
              response.end(JSON.stringify({ ok: true, result }));
            } catch (error) {
              response.statusCode = error instanceof ForbiddenError ? 403 : 500;
              response.setHeader("Content-Type", "application/json");
              response.end(JSON.stringify({ ok: false, error: (error as Error).message ?? String(error) }));
            }
          })();
        });
      });
    },
  };
}
