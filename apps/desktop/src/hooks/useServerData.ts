import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  checkHealth,
  connectionFrom,
  daemonStatus,
  fetchMetrics,
  fetchSystem,
  listModels,
  listProfiles,
  readConfig,
  readMetricsLog,
  localAvailable as localCommandsAvailable,
} from "../lib/api";
import { metricsLogPath, metricsLoggingEnabled, parseConfig, type HiggsConfig } from "../lib/config";
import type {
  ConfigFile,
  DaemonStatus,
  HealthStatus,
  Metrics,
  ModelInfo,
  ProfileList,
  RequestRecord,
  Settings,
  SystemInfo,
} from "../lib/types";

const MAX_LOG_RECORDS = 5000;

export interface ServerData {
  health: HealthStatus | null;
  models: ModelInfo[];
  modelsError: string | null;
  metrics: Metrics | null;
  metricsError: string | null;
  /** GET /v1/system; null until the first successful poll or on old servers. */
  system: SystemInfo | null;
  /**
   * Inference requests from the metrics JSONL log, oldest first. The server
   * logs every HTTP request including this dashboard's own polling of
   * /metrics and /v1/models; those carry no model and are left out here.
   */
  records: RequestRecord[];
  /** Every logged request, including non-inference ones. */
  allRecords: RequestRecord[];
  logPath: string | null;
  logError: string | null;
  profiles: ProfileList | null;
  configFile: ConfigFile | null;
  config: HiggsConfig;
  daemon: DaemonStatus | null;
  localAvailable: boolean;
  lastRefresh: number;
  refresh: () => void;
  /** Re-read config and profiles (after the editor saves). */
  reloadConfig: () => Promise<void>;
}

/**
 * One polling loop for everything the dashboard shows. HTTP data comes from
 * the server; config, log, and daemon state come from the local machine and
 * are only available inside the Tauri shell.
 */
export function useServerData(settings: Settings): ServerData {
  const connection = useMemo(() => connectionFrom(settings), [settings.baseUrl, settings.apiKey]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [records, setRecords] = useState<RequestRecord[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);
  const [logError, setLogError] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<ProfileList | null>(null);
  const [configFile, setConfigFile] = useState<ConfigFile | null>(null);
  const [daemon, setDaemon] = useState<DaemonStatus | null>(null);
  const [lastRefresh, setLastRefresh] = useState(0);
  const [tick, setTick] = useState(0);
  const logOffset = useRef<number | null>(null);
  const logPathRef = useRef<string | null>(null);
  const pollInFlight = useRef(false);

  const config = useMemo(() => parseConfig(configFile?.parsed), [configFile]);

  const reloadConfig = useCallback(async () => {
    if (!localCommandsAvailable) return;
    try {
      const list = await listProfiles();
      setProfiles(list);
      const profile = list.profiles.find((p) => p.name === settings.profile) ?? list.profiles[0];
      const file = await readConfig(profile.config_path);
      setConfigFile(file);
    } catch (error) {
      setLogError(String(error));
    }
  }, [settings.profile]);

  useEffect(() => {
    void reloadConfig();
  }, [reloadConfig]);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (pollInFlight.current) return;
      pollInFlight.current = true;
      try {
        await poll();
      } finally {
        pollInFlight.current = false;
      }
    };
    const poll = async () => {
      const [healthResult, modelsResult, metricsResult, systemResult] = await Promise.allSettled([
        checkHealth(connection),
        listModels(connection),
        fetchMetrics(connection),
        fetchSystem(connection),
      ]);
      if (cancelled) return;
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      if (modelsResult.status === "fulfilled") {
        setModels(modelsResult.value);
        setModelsError(null);
      } else {
        setModels([]);
        setModelsError(String(modelsResult.reason));
      }
      if (metricsResult.status === "fulfilled") {
        setMetrics(metricsResult.value);
        setMetricsError(null);
      } else {
        setMetricsError(String(metricsResult.reason));
      }
      setSystem(systemResult.status === "fulfilled" ? systemResult.value : null);

      if (localCommandsAvailable) {
        const [daemonResult] = await Promise.allSettled([daemonStatus(settings.profile)]);
        if (cancelled) return;
        if (daemonResult.status === "fulfilled") setDaemon(daemonResult.value);

        if (profiles) {
          const path = metricsLogPath(config, settings.profile, profiles.config_dir);
          if (path !== logPathRef.current) {
            logPathRef.current = path;
            logOffset.current = null;
            setRecords([]);
            setLogPath(path);
          }
          if (!metricsLoggingEnabled(config)) {
            setLogError("Metrics logging is disabled in this config ([logging.metrics] enabled = false)");
          } else {
            try {
              const log = await readMetricsLog(path, MAX_LOG_RECORDS, logOffset.current);
              if (cancelled) return;
              if (!log.exists) {
                setLogError(`No metrics log at ${log.path} yet (the daemon writes it after the first request)`);
              } else {
                setLogError(null);
                const fullRead = log.reset || logOffset.current === null;
                logOffset.current = log.offset;
                setRecords((current) => {
                  const merged = fullRead ? log.records : [...current, ...log.records];
                  return merged.length > MAX_LOG_RECORDS ? merged.slice(merged.length - MAX_LOG_RECORDS) : merged;
                });
              }
            } catch (error) {
              if (!cancelled) setLogError(String(error));
            }
          }
        }
      }
      if (!cancelled) setLastRefresh(Date.now());
    };
    void run();
    const timer = setInterval(run, Math.max(1, settings.refreshSeconds) * 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [connection, settings.refreshSeconds, settings.profile, profiles, config, tick]);

  const inferenceRecords = useMemo(() => records.filter((record) => record.model !== null), [records]);

  return {
    health,
    models,
    modelsError,
    metrics,
    metricsError,
    system,
    records: inferenceRecords,
    allRecords: records,
    logPath,
    logError,
    profiles,
    configFile,
    config,
    daemon,
    localAvailable: localCommandsAvailable,
    lastRefresh,
    refresh: () => setTick((n) => n + 1),
    reloadConfig,
  };
}
