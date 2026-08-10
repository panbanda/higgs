#![allow(
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::manual_let_else
)]

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use higgs_models::mlx_exec::{async_eval, eval};
use higgs_models::{
    AnyCache, AnyModel, LogprobArrays, SamplingParams, Speculation, apply_penalties,
    dflash::{DFlashCache, DFlashConfig, DFlashDrafter, accept_prefix},
    sample,
    turboquant::KvCacheConfig,
};
use mlx_rs::{
    Array, Dtype, Stream,
    ops::indexing::{IndexOp, NewAxis},
    with_new_default_stream,
};
use tokenizers::Tokenizer;

use crate::{
    cache::{
        DiskPrefixCache, DiskPrefixCacheConfig, PagedKvCache,
        paired::{
            DflashTapFrontier, LivePair, LivePairDecodeLease, PairedCacheError, RetainedState,
            SessionDsparkPublication,
        },
    },
    chat_template::{ChatMessage, ChatTemplateRenderer},
    decode::token_ledger::{LedgerError, RetentionAction, SpeculativeTicket, TokenLedger},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    mlx_tuning::MlxRuntimeTuning,
    model_loader,
    paged_prefix_cache::{
        DEFAULT_BLOCK_SIZE, PagedPairedLookupPlan, PairedPrepareTicket, PairedTouchToken,
    },
    scheduler::RoundRobinScheduler,
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;
const DEFAULT_PAGED_KV_BLOCK_SIZE: usize = 64;
/// Default `<think>` budget (tokens) when a request omits `reasoning_budget`.
const DEFAULT_THINKING_BUDGET: u32 = 256;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GenerationPromptSuffixes {
    no_thinking: Box<[u32]>,
    thinking: Box<[u32]>,
}

impl GenerationPromptSuffixes {
    #[must_use]
    fn for_request(&self, enable_thinking: bool) -> &[u32] {
        if enable_thinking {
            &self.thinking
        } else {
            &self.no_thinking
        }
    }
}

fn generation_prompt_suffix(
    template: &ChatTemplateRenderer,
    tokenizer: &Tokenizer,
    enable_thinking: bool,
) -> Option<Box<[u32]>> {
    let test_msg = [ChatMessage {
        role: "user".to_owned(),
        content: "x".to_owned(),
        tool_calls: None,
    }];
    let with_gen = template
        .apply_with_thinking(&test_msg, None, true, enable_thinking)
        .ok()?;
    let without_gen = template
        .apply_with_thinking(&test_msg, None, false, enable_thinking)
        .ok()?;
    let toks_with = tokenizer.encode(with_gen.as_str(), false).ok()?;
    let toks_without = tokenizer.encode(without_gen.as_str(), false).ok()?;
    toks_with
        .get_ids()
        .strip_prefix(toks_without.get_ids())
        .map(Into::into)
}

fn exact_generation_body<'a>(prompt: &'a [u32], generation_suffix: &[u32]) -> Option<&'a [u32]> {
    if generation_suffix.is_empty() || generation_suffix.len() >= prompt.len() {
        return None;
    }
    prompt.strip_suffix(generation_suffix)
}

/// Acquire a `Mutex` lock, recovering from poison by reusing the inner data.
/// Used in this crate to keep session-management methods infallible while
/// still satisfying `clippy::unwrap_used`.
fn lock_or_recover<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// How many leading tokens a retained cache already covers, if `prior_tokens`
/// is a strict prefix of `full` AND `full` actually extends it. `None` means the
/// conversation diverged (client edited/retried/reordered) or didn't grow — the
/// caller must then fall back to a clean full prefill. The cache-poisoning guard.
fn continuation_prior_len(prior_tokens: &[u32], full: &[u32]) -> Option<usize> {
    let prior = prior_tokens.len();
    if prior == 0 || prior >= full.len() || full.get(..prior) != Some(prior_tokens) {
        None
    } else {
        Some(prior)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeculationRoute {
    DFlash,
    MtpFamily,
    Autoregressive,
}

fn paired_dflash_exact_domain(params: &SamplingParams, thinking_enabled: bool) -> bool {
    !thinking_enabled
        && params.temperature <= f32::EPSILON
        && (params.top_p - 1.0).abs() <= f32::EPSILON
        && params.top_k.is_none()
        && params.min_p.is_none()
        && !params.has_penalties()
}

fn resolve_speculation_route(
    selector: Speculation,
    dflash_eligible: bool,
    mtp_family_eligible: bool,
) -> SpeculationRoute {
    match selector {
        Speculation::None => SpeculationRoute::Autoregressive,
        Speculation::Mtp => {
            if mtp_family_eligible {
                SpeculationRoute::MtpFamily
            } else {
                SpeculationRoute::Autoregressive
            }
        }
        Speculation::DFlash => {
            if dflash_eligible {
                SpeculationRoute::DFlash
            } else {
                SpeculationRoute::Autoregressive
            }
        }
        Speculation::Auto => {
            if dflash_eligible {
                SpeculationRoute::DFlash
            } else if mtp_family_eligible {
                SpeculationRoute::MtpFamily
            } else {
                SpeculationRoute::Autoregressive
            }
        }
    }
}

#[allow(clippy::float_cmp)]
fn stateless_mtp_family_eligible(
    params: &SamplingParams,
    constraint_active: bool,
    logprobs: bool,
) -> bool {
    !constraint_active && !logprobs && params.temperature == 0.0 && !params.has_penalties()
}

fn session_ledger_error(error: LedgerError) -> EngineError {
    EngineError::Generation(format!("session token ledger: {error}"))
}

fn session_pair_error(error: PairedCacheError) -> EngineError {
    EngineError::Generation(format!("session live dSpark pair: {error}"))
}

fn radix_pair_error(error: PairedCacheError) -> EngineError {
    EngineError::Generation(format!("radix live dSpark pair: {error}"))
}

/// Insert a retained KV cache for `session_id`, enforcing the resident-memory
/// bounds: a per-session token cap (drop instead of retain once the
/// conversation's KV exceeds `max_session_tokens`; `0` = unlimited) and a count
/// cap (LRU-evict until at most `max_sessions`, clamped to >= 1). Pure map logic,
/// so the bounds are unit-testable without loading a model.
fn stash_into(
    map: &mut std::collections::HashMap<u64, RetainedKv>,
    session_id: u64,
    state: RetainedState,
    max_sessions: usize,
    max_session_tokens: usize,
) -> usize {
    if max_session_tokens > 0 && state.tokens().len() > max_session_tokens {
        // Too large to retain — also forget any prior smaller cache for this id
        // so it can't linger past the cap. Not counted as an eviction.
        map.remove(&session_id);
        return 0;
    }
    map.insert(
        session_id,
        RetainedKv {
            state,
            last_used: std::time::Instant::now(),
        },
    );
    let cap = max_sessions.max(1);
    let mut evicted = 0;
    while map.len() > cap {
        let oldest = map
            .iter()
            .min_by_key(|(_, kept)| kept.last_used)
            .map(|(&id, _)| id);
        match oldest {
            Some(id) => {
                map.remove(&id);
                evicted += 1;
            }
            None => break,
        }
    }
    evicted
}

fn retention_token_cap(
    config: KvCacheConfig,
    target_only_cap_exempt: bool,
    state: &RetainedState,
) -> usize {
    if target_only_cap_exempt && state.allows_target_only_cap_exemption() {
        0
    } else {
        config.max_session_tokens
    }
}

/// Drop retained caches idle longer than `ttl`; returns how many were removed.
/// Pure map logic, unit-testable without a model.
fn evict_idle_from(
    map: &mut std::collections::HashMap<u64, RetainedKv>,
    ttl: std::time::Duration,
) -> usize {
    let before = map.len();
    map.retain(|_, kept| kept.last_used.elapsed() < ttl);
    before.saturating_sub(map.len())
}

fn parse_enabled_flag(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => Some(true),
        Some("0" | "false" | "off" | "no") => Some(false),
        _ => None,
    }
}

fn experimental_paged_kv_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_EXPERIMENTAL_PAGED_KV").ok().as_deref())
        .unwrap_or(false)
}

fn prefix_cache_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_PREFIX_CACHE").ok().as_deref()).unwrap_or(true)
}

fn prompt_lookup_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_PROMPT_LOOKUP").ok().as_deref()).unwrap_or(false)
}

fn unchecked_prompt_lookup_enabled() -> bool {
    parse_enabled_flag(
        std::env::var("HIGGS_PROMPT_LOOKUP_UNCHECKED")
            .ok()
            .as_deref(),
    )
    .unwrap_or(false)
}

fn mtp_adaptive_draft_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_ADAPTIVE_DRAFT").ok().as_deref()).unwrap_or(false)
}

fn mtp_prompt_lookup_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_PROMPT_LOOKUP").ok().as_deref()).unwrap_or(false)
}

fn adaptive_draft_depth_for_cap(configured_max: usize) -> crate::mtp::AdaptiveDraftDepth {
    crate::mtp::AdaptiveDraftDepth::new(configured_max, configured_max)
}

fn mtp_prefill_priming_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_PRIME_PREFILL").ok().as_deref()).unwrap_or(true)
}

/// DFlash confidence-truncation threshold (`HIGGS_DFLASH_CONF_TRUNC`, a
/// drafter top-1 probability in (0,1]); `None` disables truncation.
///
/// Training-free port of DSpark's confidence scheduler: verify only the
/// prefix of the draft block the drafter itself is confident about, skip the
/// doomed tail. Extra profitable on MoE targets, where verify cost scales
/// with verified positions (unique experts activated), unlike dense targets.
/// Default 0.5 (measured on Qwen3.6-35B-A3B: code 56->70 tok/s now beating
/// MTP, deterministic 87->92, prose unchanged and still MTP-floored, all
/// byte-exact gates green). Set to `0` to disable.
fn dflash_conf_trunc_threshold() -> Option<f32> {
    static THRESHOLD: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| match std::env::var("HIGGS_DFLASH_CONF_TRUNC") {
        Err(_) => Some(0.5),
        Ok(raw) => raw
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|t| *t > 0.0 && *t <= 1.0),
    })
}

fn validate_dspark_target(model: &AnyModel, config: &DFlashConfig) -> Result<(), EngineError> {
    if !config.is_dspark() {
        return Ok(());
    }
    let AnyModel::Qwen3Next(target) = model else {
        return Err(EngineError::Generation(
            "Prism dSpark requires a Qwen3.5/Qwen3Next target".to_owned(),
        ));
    };
    if target.args.hidden_size != config.hidden_size {
        return Err(EngineError::Generation(format!(
            "dSpark/target hidden-size mismatch: draft={} target={}",
            config.hidden_size, target.args.hidden_size
        )));
    }
    if target.args.vocab_size != config.vocab_size {
        return Err(EngineError::Generation(format!(
            "dSpark/target vocabulary mismatch: draft={} target={}",
            config.vocab_size, target.args.vocab_size
        )));
    }
    let target_layers = usize::try_from(target.args.num_hidden_layers)
        .map_err(|_| EngineError::Generation("negative target layer count".to_owned()))?;
    let taps = config.target_layer_ids();
    if taps.is_empty()
        || taps.iter().any(|&layer| layer >= target_layers)
        || !taps
            .windows(2)
            .all(|pair| matches!(pair, [left, right] if left < right))
    {
        return Err(EngineError::Generation(format!(
            "invalid dSpark target taps {taps:?} for {target_layers} target layers"
        )));
    }
    Ok(())
}

/// Device-resident draft tokens plus an optional already-materialized host copy.
struct DflashProposal {
    tokens: Array,
    host_tokens: Option<Vec<u32>>,
}

impl DflashProposal {
    fn host_tokens(&self) -> Result<Vec<u32>, EngineError> {
        self.host_tokens.clone().map_or_else(
            || {
                Ok(self
                    .tokens
                    .reshape(&[-1])
                    .map_err(EngineError::Mlx)?
                    .as_slice::<u32>()
                    .to_vec())
            },
            Ok,
        )
    }
}

fn dflash_verify_input(anchor: i32, proposal: &DflashProposal) -> Result<Array, EngineError> {
    let draft_count = *proposal
        .tokens
        .shape()
        .get(1)
        .ok_or_else(|| EngineError::Generation("draft proposal missing T axis".to_owned()))?;
    if draft_count <= 0 {
        return Err(EngineError::Generation(
            "draft proposal must contain at least one token".to_owned(),
        ));
    }
    let anchor_array = Array::from_slice(&[anchor], &[1, 1]);
    let draft_i32 = proposal
        .tokens
        .as_dtype(mlx_rs::Dtype::Int32)
        .map_err(EngineError::Mlx)?;
    mlx_rs::ops::concatenate_axis(&[&anchor_array, &draft_i32], 1).map_err(EngineError::Mlx)
}

fn dflash_canonical_target_is_terminal(
    position: usize,
    target: u32,
    draft_tokens: &[u32],
    eos_token_ids: &[u32],
    textual_stop: bool,
    think_close_token: Option<u32>,
) -> bool {
    draft_tokens
        .get(position)
        .is_none_or(|draft| *draft != target)
        || eos_token_ids.contains(&target)
        || textual_stop
        || think_close_token == Some(target)
}

fn dflash_new_stop_prefix_len<E, F>(
    existing: &[u32],
    candidates: &[u32],
    stop_sequences: &[String],
    mut decode: F,
) -> Result<Option<usize>, E>
where
    F: FnMut(&[u32]) -> Result<String, E>,
{
    if stop_sequences.is_empty() {
        return Ok(None);
    }
    let mut prefix = Vec::with_capacity(existing.len() + candidates.len());
    prefix.extend_from_slice(existing);
    let mut previous_text = decode(&prefix)?;
    for (index, &candidate) in candidates.iter().enumerate() {
        prefix.push(candidate);
        let text = decode(&prefix)?;
        let mut common_prefix = previous_text
            .as_bytes()
            .iter()
            .zip(text.as_bytes())
            .take_while(|(left, right)| left == right)
            .count();
        while common_prefix > 0 && !text.is_char_boundary(common_prefix) {
            common_prefix -= 1;
        }
        let new_len = text.len().saturating_sub(common_prefix);
        if new_len > 0 && find_stop_in_tail(&text, new_len, stop_sequences).is_some() {
            return Ok(Some(index + 1));
        }
        previous_text = text;
    }
    Ok(None)
}

fn dflash_resolve_target_then_draft<T, D, FT, FD>(
    resolve_target: FT,
    resolve_draft: FD,
) -> Result<(T, D), EngineError>
where
    FT: FnOnce() -> Result<T, EngineError>,
    FD: FnOnce() -> Result<D, EngineError>,
{
    // This ordering is a performance invariant: target resolution is the
    // round's synchronization barrier, while the proposal remains on-device.
    let target = resolve_target()?;
    let draft = resolve_draft()?;
    Ok((target, draft))
}

/// Narrow only dSpark's sequential proposal/verify prefix at a hard output
/// boundary. The diffusion trunk retains its fixed trained block size.
fn dflash_tail_draft_cap(configured: i32, remaining: usize, is_dspark: bool) -> i32 {
    if !is_dspark {
        return configured;
    }
    let useful = remaining.saturating_sub(1).max(1);
    let useful = i32::try_from(useful).unwrap_or(i32::MAX);
    configured.min(useful)
}

/// Turn a DFlash/dSpark trunk output into draft tokens.
///
/// Modal DFlash uses the target output head on positions `1..` and retains the
/// existing training-free confidence truncation. Prism dSpark instead owns a
/// Q4 output head and sequential Markov resampler. dSpark tokens stay lazy and
/// device-resident so target verification is the round's only host barrier.
fn dflash_propose_tokens(
    model: &AnyModel,
    drafter: &DFlashDrafter,
    draft_hidden: &Array,
    anchor: i32,
    draft_cap: i32,
    dspark_target_head: bool,
) -> Result<DflashProposal, EngineError> {
    let dspark_hidden = if drafter.config.is_dspark() {
        draft_hidden.index((.., ..draft_cap, ..))
    } else {
        draft_hidden.clone()
    };
    let target_logits = if drafter.config.is_dspark() && dspark_target_head {
        Some(
            model
                .forward_all_logits_from_hidden(&dspark_hidden)
                .map_err(EngineError::Mlx)?,
        )
    } else {
        None
    };
    if let Some(tokens) = drafter
        .propose_dspark_tokens(&dspark_hidden, anchor, target_logits.as_ref())
        .map_err(EngineError::Mlx)?
    {
        return Ok(DflashProposal {
            tokens,
            host_tokens: None,
        });
    }

    let draft_hidden_sliced = draft_hidden.index((.., 1.., ..));
    let draft_logits = model
        .forward_all_logits_from_hidden(&draft_hidden_sliced)
        .map_err(EngineError::Mlx)?;
    let draft_token_arr = mlx_rs::argmax_axis!(&draft_logits, -1).map_err(EngineError::Mlx)?;
    let conf_logp = if dflash_conf_trunc_threshold().is_some() {
        let max_l = mlx_rs::ops::max_axis(&draft_logits, -1, None).map_err(EngineError::Mlx)?;
        let lse = mlx_rs::ops::logsumexp_axis(&draft_logits, -1, None).map_err(EngineError::Mlx)?;
        Some(
            max_l
                .subtract(&lse)
                .map_err(EngineError::Mlx)?
                .as_dtype(mlx_rs::Dtype::Float32)
                .map_err(EngineError::Mlx)?,
        )
    } else {
        None
    };
    match conf_logp.as_ref() {
        Some(logp) => eval([&draft_token_arr, logp]).map_err(EngineError::Mlx)?,
        None => eval([&draft_token_arr]).map_err(EngineError::Mlx)?,
    }
    let mut draft = draft_token_arr
        .reshape(&[-1])
        .map_err(EngineError::Mlx)?
        .as_slice::<u32>()
        .to_vec();
    if let (Some(threshold), Some(logp)) = (dflash_conf_trunc_threshold(), conf_logp.as_ref()) {
        let logp_flat = logp.reshape(&[-1]).map_err(EngineError::Mlx)?;
        let logp_values = logp_flat.as_slice::<f32>();
        let log_threshold = threshold.ln();
        let keep = logp_values
            .iter()
            .take(draft.len())
            .take_while(|&&value| value >= log_threshold)
            .count()
            .max(1);
        draft.truncate(keep);
    }
    let draft_len = i32::try_from(draft.len())
        .map_err(|_| EngineError::Generation("draft length overflow i32".to_owned()))?;
    Ok(DflashProposal {
        tokens: Array::from_slice(&draft, &[1, draft_len]),
        host_tokens: Some(draft),
    })
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn prompt_lookup_config() -> crate::mtp::PromptLookupConfig {
    let defaults = crate::mtp::PromptLookupConfig::default();
    crate::mtp::PromptLookupConfig {
        max_drafts: parse_env_usize("HIGGS_PROMPT_LOOKUP_DRAFT_N_MAX", defaults.max_drafts),
        max_ngram: parse_env_usize("HIGGS_PROMPT_LOOKUP_NGRAM_MAX", defaults.max_ngram),
        max_window: parse_env_usize("HIGGS_PROMPT_LOOKUP_WINDOW", defaults.max_window),
    }
}

fn estimate_paged_kv_blocks(
    target_bytes: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> usize {
    let bytes_per_token = num_kv_heads
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<half::f16>())
        .max(1);
    let bytes_per_block = bytes_per_token.saturating_mul(block_size).max(1);
    (target_bytes / bytes_per_block).clamp(256, 4096)
}

#[allow(unsafe_code)]
pub(crate) fn should_clear_mlx_cache_after_prefill() -> bool {
    parse_enabled_flag(
        std::env::var("HIGGS_CLEAR_CACHE_AFTER_PREFILL")
            .ok()
            .as_deref(),
    )
    .unwrap_or(false)
}

#[allow(unsafe_code)]
pub fn maybe_clear_mlx_cache(enabled: bool, reason: &str) {
    if !enabled {
        return;
    }

    let rc = unsafe { mlx_sys::mlx_clear_cache() };
    if rc == 0 {
        tracing::debug!(reason, "Cleared MLX allocator cache");
    } else {
        tracing::warn!(reason, rc, "Failed to clear MLX allocator cache");
    }
}

/// Configure MLX's large-model working set on Apple Silicon.
///
/// By default we mirror upstream `mlx-lm` and raise MLX's wired limit to the
/// device's `max_recommended_working_set_size`. Set
/// `HIGGS_WIRED_LIMIT_MODE=legacy` to restore the older conservative
/// `memory_limit/cache_limit` behavior, or `HIGGS_NO_MEM_LIMIT=1` to skip both.
#[allow(unsafe_code)]
pub(crate) fn set_wired_limit_to_max(enabled: bool) {
    if !enabled {
        tracing::info!("MLX wired-limit escalation disabled by config");
        return;
    }
    unsafe {
        let mut info = mlx_sys::mlx_device_info_new();
        let mut dev = mlx_sys::mlx_device_new();
        mlx_sys::mlx_get_default_device(&raw mut dev);
        if mlx_sys::mlx_device_info_get(&raw mut info, dev) == 0 {
            let mut max_rec: usize = 0;
            let key = c"max_recommended_working_set_size";
            if mlx_sys::mlx_device_info_get_size(&raw mut max_rec, info, key.as_ptr()) == 0
                && max_rec > 0
            {
                let wired_mode = std::env::var("HIGGS_WIRED_LIMIT_MODE").ok();
                let use_legacy_limits =
                    matches!(wired_mode.as_deref(), Some("legacy" | "safe" | "caps"));
                let mut prev_mem: usize = 0;
                let mut prev_cache: usize = 0;
                let mut prev_wired: usize = 0;

                let limits_enabled = std::env::var("HIGGS_NO_MEM_LIMIT").is_err();

                if limits_enabled {
                    if use_legacy_limits {
                        let mem_limit = max_rec * 3 / 4;
                        let cache_limit = max_rec / 2;
                        mlx_sys::mlx_set_memory_limit(&raw mut prev_mem, mem_limit);
                        mlx_sys::mlx_set_cache_limit(&raw mut prev_cache, cache_limit);
                        tracing::info!(
                            mode = "legacy",
                            max_recommended_mb = max_rec / (1024 * 1024),
                            memory_limit_mb = mem_limit / (1024 * 1024),
                            cache_limit_mb = cache_limit / (1024 * 1024),
                            prev_mem_mb = prev_mem / (1024 * 1024),
                            prev_cache_mb = prev_cache / (1024 * 1024),
                            "Configured MLX legacy memory/cache caps",
                        );
                    } else {
                        mlx_sys::mlx_set_wired_limit(&raw mut prev_wired, max_rec);
                        tracing::info!(
                            mode = "mlx_wired_limit",
                            max_recommended_mb = max_rec / (1024 * 1024),
                            wired_limit_mb = max_rec / (1024 * 1024),
                            prev_wired_mb = prev_wired / (1024 * 1024),
                            "Configured MLX wired limit",
                        );
                    }
                } else {
                    tracing::info!(
                        mode = if use_legacy_limits {
                            "legacy"
                        } else {
                            "mlx_wired_limit"
                        },
                        max_recommended_mb = max_rec / (1024 * 1024),
                        "Skipped MLX memory-limit configuration",
                    );
                }
            }
        }
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
    }
}

/// Session state for batched generation.
#[derive(Debug, Clone)]
pub struct Session {
    /// Stable session identifier used by the scheduler and paged KV cache.
    pub id: u64,
    /// Prompt plus generated token IDs accumulated so far for this session.
    pub tokens: Vec<u32>,
    /// Whether generation for this session has already terminated.
    pub finished: bool,
    /// Maximum number of completion tokens allowed for this session.
    pub max_tokens: usize,
}

/// Cumulative cache-effectiveness counters (lock-free), surfaced via
/// [`SimpleEngine::cache_stats`] for observability.
#[derive(Default)]
struct CacheMetrics {
    radix_lookups: AtomicU64,
    radix_hits: AtomicU64,
    paired_radix_lookups: AtomicU64,
    paired_radix_hits: AtomicU64,
    prefill_saved_tokens: AtomicU64,
    continuations: AtomicU64,
    sessions_evicted: AtomicU64,
}

/// Snapshot of cache effectiveness for the `/metrics` endpoint.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Radix prefix-cache lookups on the normal generate path.
    pub radix_lookups: u64,
    /// Radix prefix-cache hits (a stored prefix was reused).
    pub radix_hits: u64,
    /// Memory-only paired target/dSpark radix lookups.
    pub paired_radix_lookups: u64,
    /// Paired radix hits that materialized and forked both cache halves.
    pub paired_radix_hits: u64,
    /// Prompt tokens NOT re-prefilled thanks to reuse (radix + continuation).
    pub prefill_saved_tokens: u64,
    /// Per-session continuations (a retained cache was reused).
    pub continuations: u64,
    /// Retained sessions evicted (count cap + idle TTL).
    pub sessions_evicted: u64,
    /// Currently retained per-session caches.
    pub retained_sessions: usize,
    /// Currently retained sessions that own an inseparable target/dSpark pair.
    pub retained_paired_sessions: usize,
    /// Conservative target bytes retained by paired sessions.
    pub retained_paired_target_bytes: usize,
    /// Conservative dSpark bytes retained by paired sessions.
    pub retained_paired_dflash_bytes: usize,
    /// Currently stored radix prefixes.
    pub radix_entries: usize,
    /// Currently stored paired radix endpoints.
    pub paired_radix_entries: usize,
    /// Conservative target bytes retained by paired radix endpoints.
    pub paired_radix_target_bytes: usize,
    /// Conservative dSpark bytes retained by paired radix endpoints.
    pub paired_radix_dflash_bytes: usize,
}

/// Simple single-request inference engine with paged KV caching.
///
/// Uses paged KV cache for efficient memory management during single-request
/// generation. Session-based batched stepping APIs are not wired to real decode
/// yet and return explicit errors instead of placeholder output.
/// `DFlash` block-diffusion speculative decoding state, held alongside the
/// target model when a drafter is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DFlashVerifyMode {
    /// Fold the target's ordinary one-token transition until first rejection.
    /// This is the fail-closed dSpark default and never needs rollback.
    CanonicalS1,
    /// Experimental S>1 target forward plus GDN innovation-tape commit.
    BatchedTape,
}

impl DFlashVerifyMode {
    fn for_drafter(is_dspark: bool, configured: Option<&str>) -> Self {
        if !is_dspark {
            return Self::BatchedTape;
        }
        match configured
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("block" | "batched" | "batched-tape") => Self::BatchedTape,
            Some("canonical" | "sequential" | "s1") | None => Self::CanonicalS1,
            // Unknown values fail closed instead of silently enabling an
            // unproven numerical schedule.
            Some(_) => Self::CanonicalS1,
        }
    }

    /// Resolve the verifier inside the policy boundary it can preserve.
    ///
    /// The tape verifier does not yet own a transactional RNG/history or a
    /// forced-token policy. Those requests therefore use the commit-only S=1
    /// oracle even when block mode was selected for greedy profiling.
    #[allow(clippy::float_cmp)]
    fn for_request(
        self,
        is_dspark: bool,
        params: &SamplingParams,
        forced_token_policy: bool,
    ) -> Self {
        if is_dspark && (params.temperature != 0.0 || params.has_penalties() || forced_token_policy)
        {
            Self::CanonicalS1
        } else {
            self
        }
    }
}

/// Target-state ownership for one `DFlash` verification round.
///
/// A canonical round contains only already-committed S=1 transitions. A tape
/// round owns every artifact needed to select one prefix from tentative S>1
/// state. Keeping those cases distinct prevents a later output policy from
/// accidentally applying rollback to committed state, or skipping rollback on
/// tentative state.
enum DFlashVerifyRound {
    Committed {
        target_tokens: Vec<u32>,
        output_tokens: Vec<u32>,
        taps: Vec<Array>,
        pending_replaced: bool,
        expected_taps: usize,
    },
    TentativeTape {
        target_rows: i32,
        target_tokens: Vec<u32>,
        output_tokens: Vec<u32>,
        taps: Vec<Array>,
        layer_tapes: Vec<Option<higgs_models::qwen3_next::GdnLayerTape>>,
        pending_replaced: bool,
        expected_taps: usize,
    },
}

struct CanonicalDflashRound {
    anchor_ticket: SpeculativeTicket,
    successors: Vec<u32>,
    tap_frontier: DflashTapFrontier,
    draft_tokens: Vec<u32>,
    draft_matches: usize,
}

struct CanonicalDflashCommit {
    anchor_ticket: SpeculativeTicket,
    successors: Vec<u32>,
    draft_tokens: Vec<u32>,
    draft_matches: usize,
}

struct CanonicalCommittedPayload {
    successors: Vec<u32>,
    draft_tokens: Vec<u32>,
    draft_matches: usize,
}

/// Drive one canonical dSpark ledger transaction through a caller-supplied
/// cache-transition backend.
///
/// This is the sole implementation of anchor-ticket creation, exact history
/// derivation, and speculative ledger commit. Stateless decoding supplies a
/// raw-frontier backend; retained sessions supply a move-only pair lease that
/// validates the replacement frontier before returning the commit capability.
/// Delivery and termination policy intentionally remain outside this boundary.
fn drive_canonical_dspark_round<F>(
    ledger: &mut TokenLedger,
    advance_and_validate: F,
) -> Result<CanonicalCommittedPayload, EngineError>
where
    F: FnOnce(SpeculativeTicket, &[u32]) -> Result<CanonicalDflashCommit, EngineError>,
{
    let pending_anchor = ledger.pending_token().ok_or_else(|| {
        EngineError::Generation("canonical dSpark ledger lost its pending token".to_owned())
    })?;
    if ledger.emitted_tokens().last().copied() != Some(pending_anchor) {
        return Err(EngineError::Generation(
            "canonical dSpark output tail does not match its pending ledger anchor".to_owned(),
        ));
    }
    let history_end = ledger
        .emitted_tokens()
        .len()
        .checked_sub(1)
        .ok_or_else(|| {
            EngineError::Generation("canonical dSpark ledger has no pending anchor".to_owned())
        })?;
    let anchor_ticket = ledger
        .begin_speculative_round()
        .map_err(session_ledger_error)?;
    let history_before_anchor = ledger.emitted_tokens().get(..history_end).ok_or_else(|| {
        EngineError::Generation(
            "canonical dSpark history boundary exceeds emitted tokens".to_owned(),
        )
    })?;
    let commit = advance_and_validate(anchor_ticket, history_before_anchor)?;
    commit.commit_ledger(ledger)
}

/// Correct-by-construction retained-session decode ownership.
///
/// The live pair lease and its pair-bound token ledger move together through
/// every speculative round and the final cache-only transition. Response
/// tokens are read from the ledger rather than mirrored in a second mutable
/// vector, so the publication boundary has exactly one authority.
struct SessionDsparkDecodeState {
    lease: LivePairDecodeLease,
    ledger: TokenLedger,
}

impl SessionDsparkDecodeState {
    fn begin(pair: LivePair, first_token: u32) -> Result<Self, EngineError> {
        let (lease, pair_key) = pair.begin_decode().map_err(session_pair_error)?;
        let mut ledger = TokenLedger::new_paired(pair_key);
        ledger
            .emit_pending(first_token)
            .map_err(session_ledger_error)?;
        Ok(Self { lease, ledger })
    }

    #[must_use]
    fn tokens(&self) -> &[u32] {
        self.ledger.emitted_tokens()
    }

    #[must_use]
    fn len(&self) -> usize {
        self.tokens().len()
    }

    #[must_use]
    fn pending_token(&self) -> Option<u32> {
        self.ledger.pending_token()
    }

    /// Advance one canonical target-authoritative round.
    ///
    /// The model transition runs inside the move-only pair lease. Only after the
    /// lease validates the replacement frontier may the canonical commit update
    /// the token ledger.
    fn run_round<F>(self, advance: F) -> Result<(Self, CanonicalCommittedPayload), EngineError>
    where
        F: FnOnce(
            &mut AnyCache,
            &mut DFlashCache,
            DflashTapFrontier,
            SpeculativeTicket,
            &[u32],
        ) -> Result<(DflashTapFrontier, CanonicalDflashCommit), EngineError>,
    {
        let Self { lease, mut ledger } = self;
        let mut continued_lease = None;
        let committed =
            drive_canonical_dspark_round(&mut ledger, |anchor_ticket, history_before_anchor| {
                let (lease, commit) = lease
                    .run(|target_cache, draft_cache, tap_frontier| {
                        advance(
                            target_cache,
                            draft_cache,
                            tap_frontier,
                            anchor_ticket,
                            history_before_anchor,
                        )
                    })
                    .map_err(session_pair_error)?;
                continued_lease = Some(lease);
                Ok(commit)
            })?;
        let lease = continued_lease.ok_or_else(|| {
            EngineError::Generation(
                "session dSpark round completed without its validated live-pair lease".to_owned(),
            )
        })?;
        Ok((Self { lease, ledger }, committed))
    }

    /// Align the response-visible tail, reunite the exact pair-bound ledger
    /// proof with its lease, and return the sole publishable live pair.
    fn finish<F>(
        self,
        eos_token_ids: &[u32],
        cache_only_forward: F,
    ) -> Result<(LivePair, Vec<u32>), EngineError>
    where
        F: FnOnce(
            &mut AnyCache,
            &mut DFlashCache,
            DflashTapFrontier,
            u32,
        ) -> Result<DflashTapFrontier, EngineError>,
    {
        let Self { lease, mut ledger } = self;
        let lease = match ledger.retention_action(eos_token_ids) {
            RetentionAction::Ready { .. } => lease,
            RetentionAction::ExcludeEos { .. } => {
                ledger
                    .exclude_pending_eos(eos_token_ids)
                    .map_err(session_ledger_error)?;
                lease
            }
            RetentionAction::CacheOnlyForward { token } => {
                let ticket = ledger
                    .begin_cache_only_forward()
                    .map_err(session_ledger_error)?;
                let (lease, ()) = lease
                    .run(|target_cache, draft_cache, tap_frontier| {
                        let next_frontier =
                            cache_only_forward(target_cache, draft_cache, tap_frontier, token)?;
                        Ok::<_, EngineError>((next_frontier, ()))
                    })
                    .map_err(session_pair_error)?;
                ledger
                    .complete_cache_only_forward(ticket)
                    .map_err(session_ledger_error)?;
                lease
            }
            RetentionAction::ForwardInFlight { token } => {
                return Err(EngineError::Generation(format!(
                    "session token {token} still has a cache-only forward in flight"
                )));
            }
        };

        let visible_tokens = ledger.emitted_tokens().to_vec();
        let proof = ledger.into_paired_proof().map_err(session_ledger_error)?;
        let pair = lease.finish(proof).map_err(session_pair_error)?;
        Ok((pair, visible_tokens))
    }
}

impl CanonicalDflashRound {
    /// Replace only the sole successor that remains absent from target KV.
    ///
    /// Canonical verification has already forwarded every earlier successor
    /// as the next S=1 input. Replacing any of those would make the published
    /// token sequence disagree with target state; the final successor is safe
    /// because it is deliberately pending.
    fn replace_pending_successor(&mut self, token: u32) -> Result<(), EngineError> {
        let pending = self.successors.last_mut().ok_or_else(|| {
            EngineError::Generation(
                "canonical DFlash round has no pending successor to replace".to_owned(),
            )
        })?;
        *pending = token;
        Ok(())
    }

    /// Publish a canonical round into any output path backed by a token ledger.
    ///
    /// Both buffered/streaming stateless generation and retained sessions use
    /// this adapter. The move-only ticket binds the target transition to the
    /// exact visible anchor; the typed frontier binds its successors to the
    /// exact target rows advanced by that transition.
    fn into_lease_parts(self) -> Result<(DflashTapFrontier, CanonicalDflashCommit), EngineError> {
        let Self {
            anchor_ticket,
            successors,
            tap_frontier,
            draft_tokens,
            draft_matches,
        } = self;
        let accepted_rows = i32::try_from(successors.len()).map_err(|_| {
            EngineError::Generation("session dSpark accepted row count overflow".to_owned())
        })?;
        let frontier_rows = tap_frontier.rows()?;
        if frontier_rows != accepted_rows {
            return Err(EngineError::Generation(format!(
                "session dSpark tap/token mismatch: taps={frontier_rows} tokens={accepted_rows}"
            )));
        }
        Ok((
            tap_frontier,
            CanonicalDflashCommit {
                anchor_ticket,
                successors,
                draft_tokens,
                draft_matches,
            },
        ))
    }
}

impl CanonicalDflashCommit {
    fn commit_ledger(
        self,
        ledger: &mut TokenLedger,
    ) -> Result<CanonicalCommittedPayload, EngineError> {
        let Self {
            anchor_ticket,
            successors,
            draft_tokens,
            draft_matches,
        } = self;
        let anchor = anchor_ticket.token();
        if ledger.emitted_tokens().last().copied() != Some(anchor) {
            return Err(EngineError::Generation(format!(
                "canonical dSpark anchor mismatch: ledger={anchor} emitted={:?}",
                ledger.emitted_tokens().last()
            )));
        }
        ledger
            .complete_speculative_round(anchor_ticket, &successors)
            .map_err(session_ledger_error)?;
        Ok(CanonicalCommittedPayload {
            successors,
            draft_tokens,
            draft_matches,
        })
    }
}

impl DFlashVerifyRound {
    fn committed(target_tokens: Vec<u32>, taps: Vec<Array>, expected_taps: usize) -> Self {
        let output_tokens = target_tokens.clone();
        Self::Committed {
            target_tokens,
            output_tokens,
            taps,
            pending_replaced: false,
            expected_taps,
        }
    }

    fn output_tokens(&self) -> &[u32] {
        match self {
            Self::Committed { output_tokens, .. } | Self::TentativeTape { output_tokens, .. } => {
                output_tokens
            }
        }
    }

    fn target_tokens(&self) -> &[u32] {
        match self {
            Self::Committed { target_tokens, .. } | Self::TentativeTape { target_tokens, .. } => {
                target_tokens
            }
        }
    }

    fn target_rows(&self) -> Result<i32, EngineError> {
        match self {
            Self::Committed { target_tokens, .. } => i32::try_from(target_tokens.len())
                .map_err(|_| EngineError::Generation("committed target rows overflow".to_owned())),
            Self::TentativeTape { target_rows, .. } => Ok(*target_rows),
        }
    }

    fn validate_output_boundary(&self) -> Result<i32, EngineError> {
        let target_rows = self.target_rows()?;
        let (taps, expected_taps) = match self {
            Self::Committed {
                taps,
                expected_taps,
                ..
            }
            | Self::TentativeTape {
                taps,
                expected_taps,
                ..
            } => (taps, *expected_taps),
        };
        if taps.len() != expected_taps {
            return Err(EngineError::Generation(format!(
                "verifier returned {} taps for {expected_taps} configured layers",
                taps.len()
            )));
        }
        let target_count = i32::try_from(self.target_tokens().len())
            .map_err(|_| EngineError::Generation("target token count overflow".to_owned()))?;
        if target_count != target_rows {
            return Err(EngineError::Generation(format!(
                "verifier produced {target_count} target tokens for {target_rows} rows"
            )));
        }
        let output_count = i32::try_from(self.output_tokens().len())
            .map_err(|_| EngineError::Generation("output token count overflow".to_owned()))?;
        if output_count <= 0 || output_count > target_rows {
            return Err(EngineError::Generation(format!(
                "verifier retained {output_count} outputs for {target_rows} target rows"
            )));
        }
        if matches!(self, Self::Committed { .. }) && output_count != target_rows {
            return Err(EngineError::Generation(format!(
                "committed verifier advanced {target_rows} rows but output policy retained {output_count}"
            )));
        }
        let (target_tokens, output_tokens, pending_replaced) = match self {
            Self::Committed {
                target_tokens,
                output_tokens,
                pending_replaced,
                ..
            }
            | Self::TentativeTape {
                target_tokens,
                output_tokens,
                pending_replaced,
                ..
            } => (target_tokens, output_tokens, pending_replaced),
        };
        let output_len = usize::try_from(output_count)
            .map_err(|_| EngineError::Generation("negative verifier output count".to_owned()))?;
        let exact_prefix_len = output_len.saturating_sub(usize::from(*pending_replaced));
        if target_tokens.get(..exact_prefix_len) != output_tokens.get(..exact_prefix_len) {
            return Err(EngineError::Generation(
                "verifier outputs are not a prefix of target tokens".to_owned(),
            ));
        }
        for (tap_index, tap) in taps.iter().enumerate() {
            let tap_rows = tap.shape().get(1).copied().ok_or_else(|| {
                EngineError::Generation(format!("verifier tap {tap_index} has no token axis"))
            })?;
            if tap_rows != target_rows {
                return Err(EngineError::Generation(format!(
                    "verifier tap {tap_index} has {tap_rows} rows for {target_rows} target rows"
                )));
            }
        }
        Ok(output_count)
    }

    /// Replace the final pending output token without pretending that a later
    /// target transition was conditioned on the replacement. Tentative rows
    /// after `index` are discarded transactionally; committed rounds may only
    /// replace their already-final row.
    fn replace_pending_output(&mut self, index: usize, token: u32) -> Result<(), EngineError> {
        let committed = matches!(self, Self::Committed { .. });
        let (output_tokens, pending_replaced) = match self {
            Self::Committed {
                output_tokens,
                pending_replaced,
                ..
            }
            | Self::TentativeTape {
                output_tokens,
                pending_replaced,
                ..
            } => (output_tokens, pending_replaced),
        };
        if index >= output_tokens.len() {
            return Err(EngineError::Generation(format!(
                "pending replacement index {index} exceeds {} verifier outputs",
                output_tokens.len()
            )));
        }
        if committed && index + 1 != output_tokens.len() {
            return Err(EngineError::Generation(
                "committed verifier cannot replace a non-final output".to_owned(),
            ));
        }
        output_tokens.truncate(index + 1);
        let pending = output_tokens.last_mut().ok_or_else(|| {
            EngineError::Generation("pending replacement removed every output".to_owned())
        })?;
        *pending = token;
        *pending_replaced = true;
        Ok(())
    }

    fn commit_target_state(
        self,
        model: &AnyModel,
        cache: &mut AnyCache,
    ) -> Result<(Vec<u32>, Vec<Array>), EngineError> {
        let output_count = self.validate_output_boundary()?;
        match &self {
            Self::TentativeTape {
                target_rows,
                layer_tapes,
                ..
            } if output_count < *target_rows => {
                model
                    .replay_tape_rollback(
                        layer_tapes,
                        cache,
                        output_count,
                        *target_rows - output_count,
                    )
                    .map_err(EngineError::Mlx)?;
            }
            Self::Committed { .. } | Self::TentativeTape { .. } => {}
        }
        Ok(match self {
            Self::Committed {
                output_tokens,
                taps,
                ..
            }
            | Self::TentativeTape {
                output_tokens,
                taps,
                ..
            } => (output_tokens, taps),
        })
    }
}

struct DFlashState {
    drafter: Mutex<DFlashDrafter>,
    tap_layers: Vec<usize>,
    block_size: i32,
    /// Number of dSpark positions sent through its vocabulary head and target
    /// verifier. The non-causal trunk still runs at its full trained block.
    draft_cap: i32,
    is_dspark: bool,
    /// Use Bonsai's packed Q1 target head for base logits instead of dSpark's
    /// frozen Q4 copy. Verification keeps output exact either way.
    dspark_target_head: bool,
    verify_mode: DFlashVerifyMode,
    mask_token_id: i32,
}

#[derive(Clone, Copy, Debug)]
struct DflashPromptPartition<'a> {
    prompt: &'a [u32],
    body_end: usize,
}

impl<'a> DflashPromptPartition<'a> {
    #[must_use]
    fn new(prompt: &'a [u32], generation_suffix: &[u32]) -> Option<Self> {
        let body_end = exact_generation_body(prompt, generation_suffix)?.len();
        Some(Self { prompt, body_end })
    }

    #[must_use]
    fn body(self) -> &'a [u32] {
        self.prompt.get(..self.body_end).unwrap_or_default()
    }

    #[must_use]
    fn from_boundary(self, boundary: usize) -> Option<&'a [u32]> {
        (boundary <= self.prompt.len())
            .then(|| self.prompt.get(boundary..))
            .flatten()
    }
}

struct DflashPrefillOutcome {
    cache: AnyCache,
    draft_cache: DFlashCache,
    logits: Array,
    taps: Vec<Array>,
    reused_prefix_len: usize,
}

struct DflashPairedPrefixPlan {
    lookup: Option<PagedPairedLookupPlan>,
    publication_ticket: PairedPrepareTicket,
}

/// Defers paired-radix LRU mutation until every model/MLX guard has dropped.
///
/// The lookup plan owns immutable target/dSpark handles, so cache eviction
/// after selection cannot invalidate the in-flight request. The one-shot touch
/// is only an LRU refresh and may safely fail closed if the endpoint was
/// cleared or replaced while the request ran.
struct DeferredPairedTouch<'a> {
    prefix_cache: &'a Mutex<DiskPrefixCache>,
    token: Option<PairedTouchToken>,
}

impl<'a> DeferredPairedTouch<'a> {
    const fn new(prefix_cache: &'a Mutex<DiskPrefixCache>) -> Self {
        Self {
            prefix_cache,
            token: None,
        }
    }

    fn arm(&mut self, token: PairedTouchToken) {
        debug_assert!(
            self.token.is_none(),
            "one dSpark request may arm at most one paired-radix touch"
        );
        self.token = Some(token);
    }
}

impl Drop for DeferredPairedTouch<'_> {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        if higgs_models::mlx_exec::held() {
            debug_assert!(
                false,
                "paired-radix LRU touch must run after releasing the MLX execution gate"
            );
            return;
        }
        let touched = match self.prefix_cache.try_lock() {
            Ok(mut cache) => Some(cache.touch_memory_paired(token)),
            Err(std::sync::TryLockError::Poisoned(error)) => {
                Some(error.into_inner().touch_memory_paired(token))
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                tracing::debug!("Skipping contended paired dSpark radix LRU touch");
                None
            }
        };
        if touched == Some(false) {
            tracing::debug!("Skipping stale paired dSpark radix LRU touch");
        }
    }
}

impl DflashPrefillOutcome {
    #[allow(clippy::too_many_arguments)]
    fn validated(
        cache: AnyCache,
        draft_cache: DFlashCache,
        logits: Array,
        taps: Vec<Array>,
        reused_prefix_len: usize,
        expected_target_boundary: i32,
        expected_taps: usize,
    ) -> Result<Self, EngineError> {
        cache
            .validate_absolute_boundary(expected_target_boundary)
            .map_err(EngineError::Mlx)?;
        DflashTapFrontier::validate_parts(
            draft_cache.position(),
            expected_target_boundary,
            &taps,
            expected_taps,
        )?;
        Ok(Self {
            cache,
            draft_cache,
            logits,
            taps,
            reused_prefix_len,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
struct GenerationPhaseTiming {
    prefill: std::time::Duration,
    decode: std::time::Duration,
    request: std::time::Duration,
    decode_tokens: u32,
}

pub struct SimpleEngine {
    model: Mutex<AnyModel>,
    prefix_cache: Mutex<DiskPrefixCache>,
    /// Paged KV cache for session-based generation
    paged_cache: Option<Mutex<PagedKvCache>>,
    /// Session scheduler for continuous batching
    scheduler: Mutex<RoundRobinScheduler>,
    /// Active sessions
    sessions: Mutex<std::collections::HashMap<u64, Session>>,
    /// Retained per-conversation live KV caches for cache-resident multi-turn
    /// tool loops (suffix-only prefill across tool hops). Keyed by session id.
    retained: Mutex<std::collections::HashMap<u64, RetainedKv>>,
    /// Per-`session_id` serialization locks, held for the whole duration of a
    /// `generate_continued` call so two concurrent requests for the same
    /// conversation can never interleave their take/generate/stash of the
    /// retained cache — the second queues behind the first and then continues
    /// from its result (or full-prefills if it diverged). Pruned in
    /// `evict_idle_retained`. Distinct sessions do not contend on this lock; they
    /// still serialize on the model lock for GPU work.
    session_locks: Mutex<std::collections::HashMap<u64, std::sync::Arc<Mutex<()>>>>,
    /// Cumulative cache-effectiveness counters (observability only).
    cache_metrics: CacheMetrics,
    tokenizer: Tokenizer,
    template: Option<ChatTemplateRenderer>,
    model_name: String,
    eos_token_ids: Vec<u32>,
    /// Control tokens stripped from decoded output (EOS + `<|…|>` chat
    /// delimiters + classic sentinels), while content-bearing special tokens
    /// (tool-call markup, `<think>`) are preserved. See [`Self::decode_tokens`].
    /// Wrapped in `Arc` so each request's streaming [`IncrementalDetok`] shares
    /// the same set and strips identically to `decode_tokens`.
    decode_skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
    /// Whether to enable thinking mode (Qwen3.5 `<think>` tags).
    enable_thinking: bool,
    /// Token ID for `</think>`, resolved from the tokenizer at load time.
    /// `None` if the tokenizer doesn't know this token (thinking will be disabled).
    think_close_token: Option<u32>,
    /// Request-mode-specific exact trailing tokens added by
    /// `add_generation_prompt=true`. A reusable boundary is accepted only when
    /// the request prompt ends in the corresponding proven suffix.
    gen_prompt_suffixes: GenerationPromptSuffixes,
    kv_cache_config: KvCacheConfig,
    tuning: MlxRuntimeTuning,
    /// Optional `DFlash` block-diffusion speculative decoding state, enabled
    /// when `HIGGS_DFLASH_PATH` points at a drafter checkpoint.
    dflash: Option<DFlashState>,
    last_dflash_accepts: std::sync::Mutex<Vec<u32>>,
    /// Proposed draft tokens that matched the target before output truncation,
    /// recorded once per DFlash round for exact acceptance telemetry.
    last_dflash_draft_matches: std::sync::Mutex<Vec<u32>>,
    /// Proposed draft-token count per round (the final length-bounded round may
    /// be narrower than the configured cap).
    last_dflash_draft_counts: std::sync::Mutex<Vec<u32>>,
    /// Most recent plain-AR phase timing. Kept alongside the existing DFlash
    /// telemetry so the paired release harness can compare identical phase
    /// boundaries without scraping rounded tracing output.
    last_ar_timing: std::sync::Mutex<Option<GenerationPhaseTiming>>,
    /// Most recent DFlash phase timing, using the same first-token/prefill and
    /// remaining-token/decode convention as [`Self::last_ar_timing`].
    last_dflash_timing: std::sync::Mutex<Option<GenerationPhaseTiming>>,
}

/// A live KV cache retained across tool turns for a conversation, so the next
/// turn prefills only the new suffix instead of re-prefilling the whole history.
/// The retained state owns both cache and exact prefix key; this wrapper adds
/// only the last-touched stamp used for idle eviction.
struct RetainedKv {
    state: RetainedState,
    last_used: std::time::Instant,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RetainedPairedStats {
    entries: usize,
    target_bytes: usize,
    dflash_bytes: usize,
}

fn retained_paired_stats(
    retained: &std::collections::HashMap<u64, RetainedKv>,
) -> RetainedPairedStats {
    retained
        .values()
        .fold(RetainedPairedStats::default(), |mut stats, entry| {
            if let Some((target_bytes, dflash_bytes)) = entry.state.paired_estimated_bytes() {
                stats.entries = stats.entries.saturating_add(1);
                stats.target_bytes = stats.target_bytes.saturating_add(target_bytes);
                stats.dflash_bytes = stats.dflash_bytes.saturating_add(dflash_bytes);
            }
            stats
        })
}

/// Intermediate state after prefix cache lookup and model locking.
struct PreparedGeneration<'a> {
    model: MutexGuard<'a, AnyModel>,
    cache: AnyCache,
    actual_prompt_tokens: Vec<u32>,
    prompt_array: Array,
    prompt_len: u32,
    pixel_values: Option<Array>,
    /// Snapshot of the cache at the conversation boundary (full prompt minus
    /// the generation-prompt suffix), captured for HYBRID caches only. Hybrid
    /// (GDN/SSM) state can't be trimmed after the fact, so the only way to store
    /// a clone that excludes the non-recurring gen-suffix (`<|im_start|>
    /// assistant\n…`, which never reappears at the same position once the
    /// assistant's real reply is in history) is to snapshot it BEFORE the suffix
    /// is prefilled. `None` for dense caches (block-pageable, suffix stripped at
    /// store time) and for multimodal/suffix-less paths.
    stored_clone: Option<AnyCache>,
    /// Process-global MLX-execution gate, held for the whole prefill + decode +
    /// stash scope. This is the single sanctioned acquisition that makes every
    /// `eval` / `async_eval` on this path pass the gate's `debug_assert`. Declared
    /// last so it is dropped *after* the model guard — the gate is released only
    /// once no more eval can occur on this generation.
    _mlx_gate: higgs_models::mlx_exec::MlxExecToken,
}

/// Result of [`SimpleEngine::generate_with_prune`], carrying the sweep metrics
/// alongside the text.
#[derive(Debug, Clone)]
pub struct PrunedGeneration {
    /// Decoded completion text.
    pub text: String,
    /// Number of completion tokens generated.
    pub completion_tokens: u32,
    /// Peak resident KV length (tokens) across the decode — the memory headline.
    pub peak_resident_kv: u32,
    /// Wall-clock decode seconds (excludes prefill), for tokens/s.
    pub decode_seconds: f32,
    /// How many decode steps triggered a prune.
    pub pruned_steps: u32,
}

/// Configuration for [`SimpleEngine::generate_self_maintained`].
#[derive(Debug, Clone, Copy)]
pub struct SelfMaintainCfg {
    /// Per-segment generation budget. A segment that hits this without finishing
    /// (no EOS) triggers a checkpoint — this is the context-length trigger.
    pub seg_max_tokens: u32,
    /// Token budget for each self-summary checkpoint.
    pub summary_max_tokens: u32,
    /// Safety cap on the number of segments.
    pub max_segments: u32,
    /// Whether to render prompts in thinking mode.
    pub enable_thinking: bool,
}

/// Result of [`SimpleEngine::generate_self_maintained`].
#[derive(Debug, Clone)]
pub struct SelfMaintainedOutput {
    /// Text of the final (answer-bearing) segment.
    pub text: String,
    /// Number of segments generated (1 = finished in one shot, no checkpoint).
    pub segments: u32,
    /// Peak resident KV across all segments — bounded by task + segment budget,
    /// constant in the number of segments (the self-maintenance memory win).
    pub peak_resident_kv: u32,
    /// Total completion tokens across segments and summaries.
    pub total_tokens: u32,
    /// The model-authored progress summaries used to carry state across segments.
    pub summaries: Vec<String>,
}

/// Result of [`SimpleEngine::generate_session`] — a cache-resident turn whose
/// prefill cost is `prefilled_tokens` (the suffix), not the whole conversation.
#[derive(Debug, Clone)]
pub struct SessionGeneration {
    /// Decoded completion text for this turn.
    pub text: String,
    /// Completion tokens generated this turn.
    pub completion_tokens: u32,
    /// Total prompt length for this turn (full conversation).
    pub prompt_tokens: u32,
    /// Tokens actually prefilled this turn — the headline win. On a continued
    /// turn this is just the new suffix (tool result + generation prompt), not
    /// `prompt_tokens`.
    pub prefilled_tokens: u32,
    /// Whether a retained cache was reused (true) or a clean prefill ran (false).
    /// A `true` is a best-effort latency win, NOT an exact-replay guarantee: the
    /// reused KV is TurboQuant-compressed, so the turn's output may differ
    /// slightly from a stateless full prefill (see `generate_continued`).
    pub continued: bool,
}

impl SimpleEngine {
    #[allow(clippy::float_cmp)]
    fn sampled_dspark_requires_ar(&self, params: &SamplingParams) -> bool {
        self.dflash
            .as_ref()
            .is_some_and(|dflash| dflash.is_dspark && params.temperature != 0.0)
    }

    /// Load a model and tokenizer from a directory.
    pub fn load<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        Self::load_with_dflash(dir, kv_cache_config, tuning, raise_wired_limit, None, None)
    }

    /// Load a model and tokenizer from a directory with an optional disk prefix
    /// cache (no `DFlash` drafter).
    pub fn load_with_disk_cache<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
        disk_cache_config: Option<DiskPrefixCacheConfig>,
    ) -> Result<Self, EngineError> {
        Self::load_with_dflash(
            dir,
            kv_cache_config,
            tuning,
            raise_wired_limit,
            None,
            disk_cache_config,
        )
    }

    /// Load a model with an optional `DFlash` speculative-decoding drafter and
    /// an optional disk prefix cache.
    ///
    /// The drafter path is taken from `dflash_path` when `Some`, otherwise from
    /// the `HIGGS_DFLASH_PATH` env var. When a drafter is present, `generate`
    /// dispatches to the block-diffusion draft-verify loop. `disk_cache_config`
    /// enables persisting prefix KV snapshots to disk; `None` keeps the cache
    /// memory-only.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn load_with_dflash<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
        dflash_path: Option<&Path>,
        disk_cache_config: Option<DiskPrefixCacheConfig>,
    ) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref();
        let model_name = derive_model_name(model_dir);

        tracing::info!(model_dir = %model_dir.display(), "Loading model");

        let model = model_loader::load_model(model_dir)?;
        let _ = model
            .make_cache_with_config(kv_cache_config)
            .map_err(EngineError::Mlx)?;
        let tokenizer = model_loader::load_tokenizer(model_dir)?;
        let template = ChatTemplateRenderer::try_from_model_dir(model_dir)?;
        if template.is_none() {
            tracing::warn!("No chat template found; /v1/chat/completions will be unavailable");
        }

        // Some Qwen checkpoints (e.g. VibeThinker-3B) list only `<|endoftext|>`
        // in `eos_token_id`, but the chat template ends every turn with
        // `<|im_end|>`. Resolve the turn terminator from the tokenizer itself so
        // chat generation stops instead of running to max_tokens.
        let im_end_id = single_special_token_id(&tokenizer, "<|im_end|>");
        let base_eos = extract_eos_tokens(model_dir);
        if let Some(id) = im_end_id {
            if !base_eos.contains(&id) {
                tracing::info!(im_end = id, "Added <|im_end|> to EOS tokens from tokenizer");
            }
        }
        let eos_token_ids = with_chat_terminator(base_eos, im_end_id);

        // Control tokens that must never surface in decoded text (EOS + the
        // `<|…|>` chat-control delimiters + classic sentinels). Shared via `Arc`
        // with each request's streaming `IncrementalDetok` so streaming strips
        // identically to `decode_tokens`.
        let decode_skip_ids =
            std::sync::Arc::new(content_preserving_skip_ids(&tokenizer, &eos_token_ids));

        // Auto-detect thinking mode: Qwen3.5 models support <think> tags.
        // Override with HIGGS_ENABLE_THINKING=0/1, off/true, yes/no etc.
        let mut enable_thinking = std::env::var("HIGGS_ENABLE_THINKING")
            .ok()
            .as_deref()
            .map_or_else(
                || detect_thinking_support(model_dir),
                |v| {
                    parse_enabled_flag(Some(v))
                        .unwrap_or_else(|| detect_thinking_support(model_dir))
                },
            );

        // Resolve </think> token ID from the tokenizer. If the tokenizer
        // doesn't know this token, disable thinking to avoid injecting
        // out-of-vocab IDs into the embedding lookup.
        let think_close_token = tokenizer.encode("</think>", false).ok().and_then(|enc| {
            let ids = enc.get_ids();
            // Must encode to exactly one token to be usable as a forced stop.
            if let [single] = ids {
                Some(*single)
            } else {
                None
            }
        });
        if enable_thinking && think_close_token.is_none() {
            tracing::warn!("Tokenizer has no single </think> token; disabling thinking mode");
            enable_thinking = false;
        }
        if enable_thinking {
            tracing::info!(think_close_token, "Thinking mode enabled");
        }

        set_wired_limit_to_max(raise_wired_limit);

        // Compute both request variants. A no-thinking override on a
        // thinking-capable model may remove `<think>\n` from the generation
        // prompt, so one load-time default is not a valid cache boundary.
        let gen_prompt_suffixes =
            template
                .as_ref()
                .map_or_else(GenerationPromptSuffixes::default, |tmpl| {
                    GenerationPromptSuffixes {
                        no_thinking: generation_prompt_suffix(tmpl, &tokenizer, false)
                            .unwrap_or_default(),
                        thinking: generation_prompt_suffix(tmpl, &tokenizer, true)
                            .unwrap_or_default(),
                    }
                });
        tracing::info!(
            no_thinking = gen_prompt_suffixes.no_thinking.len(),
            thinking = gen_prompt_suffixes.thinking.len(),
            default = gen_prompt_suffixes.for_request(enable_thinking).len(),
            "Proved request-specific generation prompt suffixes for prefix cache"
        );

        let experimental_paged_kv = experimental_paged_kv_enabled();
        if experimental_paged_kv {
            tracing::info!(
                model_name = %model_name,
                eos_tokens = ?eos_token_ids,
                requested_mlx_profile = tuning.requested_profile().as_str(),
                effective_mlx_profile = tuning.resolved_profile().as_str(),
                chunked_prefill_threshold = tuning.chunked_prefill_threshold(),
                chunked_prefill_chunk_size = tuning.chunked_prefill_chunk_size(),
                clear_cache_after_prefill = tuning.clear_cache_after_prefill(),
                mtp_enabled = tuning.enable_mtp(),
                mtp_draft_n_max = tuning.mtp_draft_n_max(),
                paged_kv_target_mb = tuning.paged_kv_target_bytes() / (1024 * 1024),
                "Engine ready"
            );
        } else {
            tracing::info!(
                model_name = %model_name,
                eos_tokens = ?eos_token_ids,
                requested_mlx_profile = tuning.requested_profile().as_str(),
                effective_mlx_profile = tuning.resolved_profile().as_str(),
                chunked_prefill_threshold = tuning.chunked_prefill_threshold(),
                chunked_prefill_chunk_size = tuning.chunked_prefill_chunk_size(),
                clear_cache_after_prefill = tuning.clear_cache_after_prefill(),
                mtp_enabled = tuning.enable_mtp(),
                mtp_draft_n_max = tuning.mtp_draft_n_max(),
                "Engine ready"
            );
            tracing::debug!("Experimental paged KV disabled; session cache allocation skipped");
        }

        let (raw_num_kv_heads, raw_head_dim) = model
            .kv_cache_geometry()
            .map_err(|e| EngineError::Generation(format!("paged cache geometry: {e}")))?;
        let num_kv_heads = usize::try_from(raw_num_kv_heads)
            .map_err(|_| EngineError::Generation("paged cache num_kv_heads overflow".to_owned()))?;
        let head_dim = usize::try_from(raw_head_dim)
            .map_err(|_| EngineError::Generation("paged cache head_dim overflow".to_owned()))?;
        let paged_cache = experimental_paged_kv
            .then(|| {
                let num_blocks = estimate_paged_kv_blocks(
                    tuning.paged_kv_target_bytes(),
                    num_kv_heads,
                    head_dim,
                    DEFAULT_PAGED_KV_BLOCK_SIZE,
                );
                PagedKvCache::new(
                    num_blocks,
                    DEFAULT_PAGED_KV_BLOCK_SIZE,
                    num_kv_heads,
                    head_dim,
                )
                .map_err(|e| EngineError::Generation(format!("paged cache init: {e}")))
            })
            .transpose()?;

        // DFlash speculative decoding: load the drafter from the explicit path
        // or HIGGS_DFLASH_PATH. generate_inner then dispatches to the
        // draft-verify loop for unconstrained text generation.
        let dflash = dflash_path
            .map(Path::to_path_buf)
            .or_else(|| {
                std::env::var("HIGGS_DFLASH_PATH")
                    .ok()
                    .map(std::path::PathBuf::from)
            })
            .map(|dp| -> Result<DFlashState, EngineError> {
                tracing::info!(drafter = %dp.display(), "Loading DFlash drafter");
                let drafter = model_loader::load_dflash_drafter(&dp, model_dir)?;
                validate_dspark_target(&model, &drafter.config)?;
                let tap_layers = drafter.config.target_layer_ids().to_vec();
                // Decode block size: HIGGS_DFLASH_BLOCK_SIZE overrides the
                // trained block_size, clamped to [1, trained]. Smaller blocks
                // amortize the per-round verify + lm_head cost better when the
                // accept length plateaus below the trained block.
                let trained_block = drafter.config.block_size;
                let block_size = if drafter.config.is_dspark() {
                    // Prism's log-SNR pattern and Markov head are trained for
                    // the checkpoint's fixed block; match the public runtime.
                    trained_block
                } else {
                    std::env::var("HIGGS_DFLASH_BLOCK_SIZE")
                        .ok()
                        .and_then(|v| v.parse::<i32>().ok())
                        .filter(|&v| v >= 1)
                        .map_or(trained_block, |v| v.min(trained_block))
                };
                let draft_cap = if drafter.config.is_dspark() {
                    // Prism trains and publishes dSpark with four proposal
                    // positions; keep the reference schedule for both the full
                    // Q4 head and target-head sidecars. Any smaller cap is an
                    // explicit experiment, never an implicit Apple default.
                    let tuned_default = trained_block;
                    std::env::var("HIGGS_DSPARK_DRAFT_CAP")
                        .ok()
                        .and_then(|v| v.parse::<i32>().ok())
                        .filter(|&v| v >= 1)
                        .map_or(tuned_default, |v| v.min(trained_block))
                } else {
                    trained_block
                };
                let dspark_target_head = drafter.config.is_dspark()
                    && (drafter.config.reuse_target_head()
                        || std::env::var("HIGGS_DSPARK_TARGET_HEAD").is_ok_and(|v| v != "0"));
                let target_is_bonsai_q1 = model.is_qwen3next_affine_1bit();
                let verify_mode_raw = std::env::var("HIGGS_DFLASH_VERIFY_MODE").ok();
                let verify_mode_raw = verify_mode_raw.or_else(|| {
                    if drafter.config.is_dspark() && target_is_bonsai_q1 {
                        Some("block".to_owned())
                    } else {
                        None
                    }
                });
                let requested_verify_mode = DFlashVerifyMode::for_drafter(
                    drafter.config.is_dspark(),
                    verify_mode_raw.as_deref(),
                );
                let verify_rows = draft_cap.checked_add(1).ok_or_else(|| {
                    EngineError::Generation("dSpark verify row count overflow".to_owned())
                })?;
                let verify_mode = if drafter.config.is_dspark()
                    && requested_verify_mode == DFlashVerifyMode::BatchedTape
                {
                    let AnyModel::Qwen3Next(target) = &model else {
                        return Err(EngineError::Generation(
                            "dSpark target is not Qwen3Next".to_owned(),
                        ));
                    };
                    let domain = target.validate_dflash_block_domain(verify_rows);
                    if let Err(reason) = domain {
                        tracing::warn!(
                            %reason,
                            verify_rows,
                            "dSpark block verifier is outside its exact model domain; using canonical S=1"
                        );
                        DFlashVerifyMode::CanonicalS1
                    } else {
                        requested_verify_mode
                    }
                } else {
                    requested_verify_mode
                };
                let is_dspark = drafter.config.is_dspark();
                let mask_token_id = drafter.config.mask_token_id();
                tracing::info!(
                    tap_layers = ?tap_layers,
                    block_size,
                    draft_cap,
                    dspark_target_head,
                    ?verify_mode,
                    mask_token_id,
                    "DFlash drafter loaded — speculative decoding enabled"
                );
                Ok(DFlashState {
                    drafter: Mutex::new(drafter),
                    tap_layers,
                    block_size,
                    draft_cap,
                    is_dspark,
                    dspark_target_head,
                    verify_mode,
                    mask_token_id,
                })
            })
            .transpose()?;

        let mut prefix_cache = disk_cache_config.map_or_else(
            || DiskPrefixCache::memory_only(DEFAULT_PREFIX_CACHE_SIZE, DEFAULT_BLOCK_SIZE),
            |config| match DiskPrefixCache::new(
                DEFAULT_PREFIX_CACHE_SIZE,
                DEFAULT_BLOCK_SIZE,
                config,
                num_kv_heads,
                head_dim,
            ) {
                Ok(cache) => cache,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "Failed to initialize disk prefix cache; falling back to memory-only cache"
                    );
                    DiskPrefixCache::memory_only(DEFAULT_PREFIX_CACHE_SIZE, DEFAULT_BLOCK_SIZE)
                }
            },
        );
        prefix_cache.set_paired_idle_ttl(
            (kv_cache_config.retained_idle_secs > 0)
                .then(|| std::time::Duration::from_secs(kv_cache_config.retained_idle_secs)),
        );

        Ok(Self {
            model: Mutex::new(model),
            prefix_cache: Mutex::new(prefix_cache),
            paged_cache: paged_cache.map(Mutex::new),
            scheduler: Mutex::new(RoundRobinScheduler::new()),
            sessions: Mutex::new(std::collections::HashMap::new()),
            retained: Mutex::new(std::collections::HashMap::new()),
            session_locks: Mutex::new(std::collections::HashMap::new()),
            cache_metrics: CacheMetrics::default(),
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            decode_skip_ids,
            enable_thinking,
            think_close_token,
            gen_prompt_suffixes,
            kv_cache_config,
            tuning,
            dflash,
            last_dflash_accepts: Mutex::new(Vec::new()),
            last_dflash_draft_matches: Mutex::new(Vec::new()),
            last_dflash_draft_counts: Mutex::new(Vec::new()),
            last_ar_timing: Mutex::new(None),
            last_dflash_timing: Mutex::new(None),
        })
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Number of conversations holding a retained live KV cache.
    pub fn retained_session_count(&self) -> usize {
        lock_or_recover(&self.retained).len()
    }

    /// Snapshot of cache effectiveness (hit rate, prefill saved, evictions,
    /// resident sizes) for observability / the `/metrics` endpoint.
    pub fn cache_stats(&self) -> CacheStats {
        let (retained_sessions, retained_paired) = {
            let retained = lock_or_recover(&self.retained);
            (retained.len(), retained_paired_stats(&retained))
        };
        let (radix_entries, paired_radix) = {
            let mut prefix_cache = lock_or_recover(&self.prefix_cache);
            prefix_cache.evict_idle_paired();
            (prefix_cache.len(), prefix_cache.paired_stats())
        };
        CacheStats {
            radix_lookups: self.cache_metrics.radix_lookups.load(Ordering::Relaxed),
            radix_hits: self.cache_metrics.radix_hits.load(Ordering::Relaxed),
            paired_radix_lookups: self
                .cache_metrics
                .paired_radix_lookups
                .load(Ordering::Relaxed),
            paired_radix_hits: self.cache_metrics.paired_radix_hits.load(Ordering::Relaxed),
            prefill_saved_tokens: self
                .cache_metrics
                .prefill_saved_tokens
                .load(Ordering::Relaxed),
            continuations: self.cache_metrics.continuations.load(Ordering::Relaxed),
            sessions_evicted: self.cache_metrics.sessions_evicted.load(Ordering::Relaxed),
            retained_sessions,
            retained_paired_sessions: retained_paired.entries,
            retained_paired_target_bytes: retained_paired.target_bytes,
            retained_paired_dflash_bytes: retained_paired.dflash_bytes,
            radix_entries,
            paired_radix_entries: paired_radix.entries,
            paired_radix_target_bytes: paired_radix.target_bytes,
            paired_radix_dflash_bytes: paired_radix.dflash_bytes,
        }
    }

    /// Number of stored prefixes in the radix prefix cache. Observability hook
    /// (and a test seam for proving prefix reuse actually happened).
    pub fn prefix_cache_len(&self) -> usize {
        lock_or_recover(&self.prefix_cache).len()
    }

    /// Drop every entry in the radix prefix cache, forcing the next generation
    /// to prefill densely from scratch. Lets a caller establish a cold baseline.
    pub fn clear_prefix_cache(&self) {
        lock_or_recover(&self.prefix_cache).clear();
    }

    /// The exact token sequence the retained KV cache for `session_id` covers
    /// (prompt + generated), or `None` if nothing is retained. This is the
    /// ground truth the continuation guard ([`continuation_prior_len`]) matches
    /// against — exposed so callers can build a genuine token-prefix extension
    /// for the next turn without round-tripping generated text through the
    /// tokenizer (BPE detok→retok is not always stable).
    ///
    /// [`continuation_prior_len`]: continuation_prior_len
    pub fn retained_session_tokens(&self, session_id: u64) -> Option<Vec<u32>> {
        lock_or_recover(&self.retained)
            .get(&session_id)
            .map(|kept| kept.state.tokens().to_vec())
    }

    /// Drop a conversation's retained KV cache, freeing its KV memory.
    pub fn drop_retained_session(&self, session_id: u64) {
        lock_or_recover(&self.retained).remove(&session_id);
    }

    /// Evict retained caches idle longer than `ttl`; returns how many were
    /// dropped. Retained caches pin real KV memory, so this must be called
    /// periodically (and `generate_continued` will also enforce a count cap).
    pub fn evict_idle_retained(&self, ttl: std::time::Duration) -> usize {
        let dropped = evict_idle_from(&mut lock_or_recover(&self.retained), ttl);
        if dropped > 0 {
            self.cache_metrics
                .sessions_evicted
                .fetch_add(u64::try_from(dropped).unwrap_or(0), Ordering::Relaxed);
        }
        // Drop per-session locks no longer referenced by any in-flight request
        // (strong_count == 1 ⇒ only this map holds the Arc). Bounds the lock map
        // so a long-lived server doesn't accumulate one entry per distinct id.
        lock_or_recover(&self.session_locks)
            .retain(|_, lock| std::sync::Arc::strong_count(lock) > 1);
        dropped
    }

    /// Look up the retained cache for `session_id` whose tokens are a prefix of
    /// `full_tokens`, returning the cache and the suffix that still needs
    /// prefilling. Returns `None` (and drops any stale entry) when there is no
    /// retained cache or the conversation diverged from it — the caller then
    /// falls back to a clean full prefill. This is the cache-poisoning guard.
    fn take_continuable(
        &self,
        session_id: u64,
        full_tokens: &[u32],
    ) -> Option<(RetainedState, usize)> {
        let mut map = lock_or_recover(&self.retained);
        let prior = match map.get(&session_id) {
            Some(entry) => continuation_prior_len(entry.state.tokens(), full_tokens),
            None => return None,
        };
        if let Some(p) = prior {
            map.remove(&session_id).map(|kept| (kept.state, p))
        } else {
            map.remove(&session_id); // drop the diverged/stale entry
            None
        }
    }

    /// Stash a live cache and the exact tokens it now represents for `session_id`,
    /// so the next turn prefills only the suffix. Caps the number of resident
    /// sessions (each pins GB-scale KV), evicting the least-recently-used.
    ///
    /// Before stashing, the cache's resident KV is TurboQuant-compressed
    /// ([`AnyCache::quantize_for_retention`]) so a conversation that decoded dense
    /// — the default, or any turn below the TQ activation threshold — does not pin
    /// fp16 KV between turns. The next turn continues by appending to the now-TQ
    /// cache (the ordinary TurboQuant decode-append path). Compression is
    /// best-effort: a layer that can't be packed stays dense and continuation
    /// still works, just uncompressed.
    #[allow(clippy::doc_markdown)]
    /// Publish an ALREADY-PREPARED retained cache (TurboQuant-compressed AND
    /// evaluated by the caller while the model lock was held — see
    /// `generate_continued`) into the session map. This does NO MLX work: MLX's
    /// Metal command buffer is process-global and aborts on concurrent eval, so
    /// all GPU work must stay serialized under the model lock, which this
    /// function does not hold.
    fn stash_retained(&self, session_id: u64, state: RetainedState, target_only_cap_exempt: bool) {
        // `target_only_cap_exempt`: the caller compressed target KV to
        // TurboQuant because it exceeded the dense token cap. Preserve the
        // historical exemption for target-only retention, but never apply it
        // to a pair whose dSpark snapshot remains uncompressed and grows with
        // context.
        let effective_cap =
            retention_token_cap(self.kv_cache_config, target_only_cap_exempt, &state);
        let token_len = state.tokens().len();
        #[allow(clippy::print_stderr)] // env-gated diagnostic
        if effective_cap > 0
            && token_len > effective_cap
            && std::env::var("HIGGS_DIAG_SESSION_TIMING").is_ok_and(|v| v == "1")
        {
            eprintln!(
                "DIAG session-retain-drop: reason=max_session_tokens session_id={session_id} tokens={} cap={}",
                token_len, effective_cap
            );
        }
        let evicted = stash_into(
            &mut lock_or_recover(&self.retained),
            session_id,
            state,
            self.kv_cache_config.max_retained_sessions,
            effective_cap,
        );
        if evicted > 0 {
            self.cache_metrics
                .sessions_evicted
                .fetch_add(u64::try_from(evicted).unwrap_or(0), Ordering::Relaxed);
        }
    }

    /// Get a reference to the tokenizer.
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the model's EOS token IDs.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Whether the engine has thinking mode enabled.
    pub const fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }

    pub fn last_dflash_accepts(&self) -> Vec<u32> {
        self.last_dflash_accepts
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Exact pre-truncation matched-draft counts from the most recent request.
    #[must_use]
    pub fn last_dflash_draft_matches(&self) -> Vec<u32> {
        self.last_dflash_draft_matches
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Actual proposal width for every round in the most recent request.
    #[must_use]
    pub fn last_dflash_draft_counts(&self) -> Vec<u32> {
        self.last_dflash_draft_counts
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Decode-only throughput from the most recent dSpark request.
    ///
    /// This deliberately excludes model load, prefill, cache publication, and
    /// request framing. It is a narrow observability seam for real-model release
    /// gates that compare cached and uncached execution of the same decoder.
    #[doc(hidden)]
    #[must_use]
    pub fn last_dflash_decode_tokens_per_second(&self) -> Option<f64> {
        let timing = self.last_dflash_timing.lock().ok()?.as_ref().copied()?;
        let seconds = timing.decode.as_secs_f64();
        (timing.decode_tokens > 0 && seconds > 0.0)
            .then(|| f64::from(timing.decode_tokens) / seconds)
    }

    /// Prefill-phase wall time from the most recent dSpark request.
    ///
    /// This narrow observability seam lets real-model release gates distinguish
    /// saved paired prefill work from steady-state dSpark decode throughput.
    #[doc(hidden)]
    #[must_use]
    pub fn last_dflash_prefill_seconds(&self) -> Option<f64> {
        let timing = self.last_dflash_timing.lock().ok()?.as_ref().copied()?;
        Some(timing.prefill.as_secs_f64())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn last_ar_timing(&self) -> Option<GenerationPhaseTiming> {
        self.last_ar_timing.lock().map_or(None, |timing| *timing)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn last_dflash_timing(&self) -> Option<GenerationPhaseTiming> {
        self.last_dflash_timing
            .lock()
            .map_or(None, |timing| *timing)
    }

    /// Render the chat template to its prompt STRING with explicit thinking
    /// control — the exact text [`prepare_chat_prompt_with_thinking`] tokenizes.
    ///
    /// Exposed so the HTTP layer can compute a continuation delta in TEXT space:
    /// the route decodes the retained tokens (special tokens preserved) and, when
    /// this rendered string starts with that decoded prefix, re-tokenizes only
    /// the trailing slice. That keeps the cached prefix tokens byte-exact instead
    /// of round-tripping the model's generated tokens through detok→retok (BPE is
    /// not round-trip stable), so the continuation guard actually matches.
    pub fn render_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<String, EngineError> {
        let renderer = self.template.as_ref().ok_or_else(|| {
            EngineError::Template(
                "This model has no chat template; use /v1/completions instead".to_owned(),
            )
        })?;
        renderer.apply_with_thinking(messages, tools, true, enable_thinking)
    }

    /// Apply chat template and tokenize messages with explicit thinking control.
    pub fn prepare_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        let prompt = self.render_chat_prompt_with_thinking(messages, tools, enable_thinking)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        self.prepare_chat_prompt_with_thinking(messages, tools, self.enable_thinking)
    }

    /// Whether the loaded model is a vision-language model.
    pub fn is_vlm(&self) -> bool {
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.is_vlm()
    }

    /// The expected image size for the VLM's vision encoder, or `None`.
    pub fn vlm_image_size(&self) -> Option<i32> {
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.image_size()
    }

    /// Replace image placeholder tokens with `IMAGE_TOKEN_INDEX` in the token
    /// sequence. The `<image>` token ID is looked up from the tokenizer.
    #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        let Some(image_token_id) = self.tokenizer.token_to_id("<image>") else {
            return;
        };
        let image_token_u32 = higgs_models::llava_qwen2::IMAGE_TOKEN_INDEX as u32;
        for token in tokens.iter_mut() {
            if *token == image_token_id {
                *token = image_token_u32;
            }
        }
    }

    /// Convert prompt length to u32, returning a descriptive error on overflow.
    fn prompt_len(prompt_tokens: &[u32]) -> Result<u32, EngineError> {
        prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))
    }

    /// Look up the prefix cache, lock the model, and resolve the actual tokens
    /// to feed into the forward pass.
    fn prepare_generation(
        &self,
        prompt_tokens: &[u32],
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<PreparedGeneration<'_>, EngineError> {
        let prompt_len = Self::prompt_len(prompt_tokens)?;
        let has_images = pixel_values.is_some();

        // Disk lookup can scan many block-aligned candidates and hash the prompt.
        // Keep that CPU-only candidate selection outside the model lock / MLX
        // execution gate; the later materialization step still runs under the
        // gate because it rebuilds MLX arrays.
        let disk_prefix_candidate = if has_images {
            None
        } else {
            let pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.find_disk_prefix_candidate(prompt_tokens, checkpoint_id, 0)
        };

        let model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;

        // Acquire the MLX-execution gate the moment we own the model, before any
        // forward/eval. Held via PreparedGeneration for the entire generation.
        let mlx_gate = higgs_models::mlx_exec::acquire();

        // Skip prefix caching for multimodal requests: different images
        // produce different KV states even with identical token sequences.
        //
        // Cache materialization MUST run under the model lock + MLX gate: a
        // radix hit on a hybrid (GDN) entry `deep_clone`s the stored cache, and
        // a disk-snapshot hit rebuilds `KvBlock`s — both eval MLX graphs. Doing
        // that materialization before the gate ran eval concurrently with
        // another request's in-flight generation on the shared default stream,
        // and MLX's process-global Metal command buffer aborts on concurrent use
        // (observed as
        // `-[_MTLCommandBuffer addCompletedHandler:]` / `IOGPUMetalCommandBuffer
        // validate` failed assertions during a session-continuation fallback
        // prefill racing a concurrent request's prefix lookup).
        let use_prefix_cache = !has_images && prefix_cache_enabled();
        let prefix_match = if !use_prefix_cache {
            None
        } else {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            let memory_match = pc.find_memory_prefix(prompt_tokens);
            let min_disk_prefix_len = memory_match
                .as_ref()
                .map_or(0, |matched| matched.prefix_len.saturating_add(1));
            let disk_match = disk_prefix_candidate
                .filter(|candidate| candidate.prefix_len() >= min_disk_prefix_len)
                .and_then(|candidate| pc.load_disk_prefix_candidate(prompt_tokens, candidate));
            match (disk_match, memory_match) {
                (Some(disk), Some(memory)) if memory.prefix_len > disk.prefix_len => Some(memory),
                (Some(disk), _) => Some(disk),
                (None, memory) => memory,
            }
        };
        if use_prefix_cache {
            self.cache_metrics
                .radix_lookups
                .fetch_add(1, Ordering::Relaxed);
            if let Some(ref m) = prefix_match {
                debug_assert!(
                    m.prefix_len <= prompt_tokens.len(),
                    "radix prefix_len {} exceeds prompt length {}",
                    m.prefix_len,
                    prompt_tokens.len()
                );
                self.cache_metrics
                    .radix_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.cache_metrics
                    .prefill_saved_tokens
                    .fetch_add(u64::try_from(m.prefix_len).unwrap_or(0), Ordering::Relaxed);
            }
        }

        let (actual_prompt_tokens, cache) = if let Some(matched) = prefix_match {
            tracing::debug!(
                prefix_len = matched.prefix_len,
                total_len = prompt_tokens.len(),
                suffix_len = prompt_tokens.len() - matched.prefix_len,
                "Prefix cache hit — reusing cached prefix"
            );
            let suffix = prompt_tokens.get(matched.prefix_len..).unwrap_or_default();
            if suffix.is_empty() {
                tracing::debug!(
                    prompt_len = prompt_tokens.len(),
                    "Full prefix hit requires cached logits; falling back to fresh prefill"
                );
                (
                    prompt_tokens.to_vec(),
                    model
                        .make_cache_with_config(self.kv_cache_config)
                        .map_err(EngineError::Mlx)?,
                )
            } else {
                (suffix.to_vec(), matched.cache)
            }
        } else {
            (
                prompt_tokens.to_vec(),
                model
                    .make_cache_with_config(self.kv_cache_config)
                    .map_err(EngineError::Mlx)?,
            )
        };

        let prompt_array = Array::from(actual_prompt_tokens.as_slice()).index(NewAxis);

        Ok(PreparedGeneration {
            model,
            cache,
            actual_prompt_tokens,
            prompt_array,
            prompt_len,
            pixel_values,
            stored_clone: None,
            _mlx_gate: mlx_gate,
        })
    }

    /// Run the prefill forward pass and sample the first token. Stores the
    /// post-prefill KV state back into the prefix cache (skipped for multimodal).
    /// Optionally computes logprobs for the first token.
    #[allow(clippy::too_many_arguments)]
    fn run_prefill(
        &self,
        prompt_tokens: &[u32],
        generation_suffix: &[u32],
        prepared: &mut PreparedGeneration<'_>,
        params: &SamplingParams,
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
        capture_hidden: bool,
        store_prefix_cache: bool,
        checkpoint_id: Option<&str>,
        // When Some, the single-pass capture_hidden prefill ALSO collects tap
        // hiddens for these layers (session DFlash drafter context). Chunked
        // and two-phase prefills skip taps — callers must tolerate None.
        prefill_tap_layers: Option<&[usize]>,
        prefill_taps_out: Option<&mut Vec<Array>>,
    ) -> Result<(Array, Option<LogprobArrays>, Option<Array>), EngineError> {
        // DIAGNOSTIC (HIGGS_DIAG_WARM_DRIFT=1): when reuse is active (the
        // prepared cache is a reused clone with offset > 0), forward the FULL
        // prompt on a fresh cache and compare the reused clone's KV to the fresh
        // full forward's prefix KV. Localizes whether the warm incremental reuse
        // path drifts from a cold full forward.
        if Self::probe_warm_drift_enabled(prompt_tokens, prepared.cache.resident_len()) {
            self.probe_warm_drift(prompt_tokens, prepared);
        }
        let mut prefill_hidden = None;
        let logits = if let Some(ref pixel_values) = prepared.pixel_values {
            // Multimodal path: full forward (VLMs need all tokens for vision)
            prepared
                .model
                .forward_multimodal(&prepared.prompt_array, pixel_values, &mut prepared.cache)
                .map_err(EngineError::Mlx)?
        } else {
            // Text-only prefill: use chunked prefill for long sequences to bound
            // peak memory, otherwise single-pass with last-token-only LM head.
            let seq_len = prepared.prompt_array.shape().get(1).copied().unwrap_or(0);
            let chunked_threshold = self.tuning.chunked_prefill_threshold();
            let chunked_size = self.tuning.chunked_prefill_chunk_size();

            // Hybrid (GDN/SSM) two-phase split. A hybrid cache can't be trimmed
            // after prefill, so the only way to store a clone that excludes the
            // non-recurring generation-prompt suffix (which diverges cross-turn
            // once the assistant's real reply is in history) is to snapshot the
            // cache BEFORE the suffix is prefilled. Forward the conversation body,
            // snapshot, then forward the suffix to obtain the generation logits.
            // Bit-identical to a single-pass prefill: chunked prefill already
            // advances the cache token-by-token, so body-then-suffix == full.
            let is_hybrid = matches!(prepared.cache, AnyCache::Hybrid(_));
            let split_at = if store_prefix_cache && is_hybrid {
                exact_generation_body(&prepared.actual_prompt_tokens, generation_suffix)
                    .map_or(0, <[u32]>::len)
            } else {
                0
            };

            if split_at > 0 {
                // Phase 1: advance the cache over the conversation body. When the
                // caller wants hidden states (MTP priming), capture the body's
                // hidden here so the full-prompt hidden can be reconstructed by
                // concatenation after Phase 2 (the MTP head needs every prompt
                // token's hidden, not just the suffix's).
                // `split_at` came from a successful exact `strip_suffix`, so the
                // body slice is structurally within the prompt.
                let body_ids = prepared
                    .actual_prompt_tokens
                    .get(..split_at)
                    .unwrap_or_default();
                let body_array = Array::from(body_ids).index(NewAxis);
                let chunked_threshold_len =
                    usize::try_from(chunked_threshold).unwrap_or(usize::MAX);
                let body_hidden = if capture_hidden {
                    let (hidden, _logits) = prepared
                        .model
                        .forward_with_hidden(&body_array, None, &mut prepared.cache)
                        .map_err(EngineError::Mlx)?;
                    Some(hidden)
                } else if body_ids.len() > chunked_threshold_len {
                    prepared
                        .model
                        .forward_chunked(&body_array, &mut prepared.cache, chunked_size)
                        .map_err(EngineError::Mlx)?;
                    None
                } else {
                    prepared
                        .model
                        .forward_last_token(&body_array, None, &mut prepared.cache)
                        .map_err(EngineError::Mlx)?;
                    None
                };
                // Snapshot at the clean conversation boundary. KV layers update
                // in place (append to the same buffer), so a shallow `clone()`
                // would share that buffer and Phase 2's suffix forward below would
                // append the gen-suffix KV into the snapshot, corrupting it (silent
                // decode divergence on reuse, ~token 10). `deep_clone()` evals and
                // deep-copies the in-place KV buffers into independent storage
                // (GDN/SSM layers are functional and stay cheap-shallow), freezing
                // the snapshot independent of Phase 2.
                prepared.stored_clone = Some(prepared.cache.deep_clone());
                // DIAGNOSTIC (env HIGGS_DIAG_STORE_DRIFT=1): throwaway cold
                // forward over the FULL prompt on a second fresh cache, compare
                // its [..split_at] KV to the two-phase stored_clone. If they
                // differ, the body snapshot's KV drifts from a cold single-pass
                // forward at a different seq_len — the store-side drift Route A
                // targets. Runs once per process when the body is large enough
                // (>= HIGGS_DIAG_STORE_DRIFT_MIN_LEN, default 0) to also probe
                // the large-body regime where the secondary divergence appears.
                if Self::probe_store_drift_enabled(split_at) {
                    self.probe_store_drift(prepared, split_at);
                }
                // Phase 2: forward the gen-suffix to obtain the final logits.
                let suffix_ids = prepared
                    .actual_prompt_tokens
                    .get(split_at..)
                    .unwrap_or_default();
                let suffix_array = Array::from(suffix_ids).index(NewAxis);
                if capture_hidden {
                    let (suffix_hidden, logits) = prepared
                        .model
                        .forward_with_hidden(&suffix_array, None, &mut prepared.cache)
                        .map_err(EngineError::Mlx)?;
                    // Reconstruct the full-prompt hidden for the MTP head.
                    prefill_hidden = match body_hidden {
                        Some(bh) => Some(
                            mlx_rs::ops::concatenate_axis(&[&bh, &suffix_hidden], 1)
                                .map_err(EngineError::Mlx)?,
                        ),
                        None => Some(suffix_hidden),
                    };
                    logits
                } else if suffix_ids.len() > chunked_threshold_len {
                    prepared
                        .model
                        .forward_chunked(&suffix_array, &mut prepared.cache, chunked_size)
                        .map_err(EngineError::Mlx)?
                } else {
                    prepared
                        .model
                        .forward_last_token(&suffix_array, None, &mut prepared.cache)
                        .map_err(EngineError::Mlx)?
                }
            } else if capture_hidden && seq_len <= chunked_threshold {
                let (hidden, logits) = match prefill_tap_layers {
                    Some(layers) if !layers.is_empty() => {
                        let (hidden, logits, taps) = prepared
                            .model
                            .forward_with_hidden_taps(
                                &prepared.prompt_array,
                                None,
                                &mut prepared.cache,
                                layers,
                            )
                            .map_err(EngineError::Mlx)?;
                        if let Some(out) = prefill_taps_out {
                            *out = taps;
                        }
                        (hidden, logits)
                    }
                    _ => prepared
                        .model
                        .forward_with_hidden(&prepared.prompt_array, None, &mut prepared.cache)
                        .map_err(EngineError::Mlx)?,
                };
                prefill_hidden = Some(hidden);
                logits
            } else if seq_len > chunked_threshold {
                prepared
                    .model
                    .forward_chunked(&prepared.prompt_array, &mut prepared.cache, chunked_size)
                    .map_err(EngineError::Mlx)?
            } else {
                prepared
                    .model
                    .forward_last_token(&prepared.prompt_array, None, &mut prepared.cache)
                    .map_err(EngineError::Mlx)?
            }
        };
        let last_logits = logits.index((.., -1, ..));

        let constrained_logits = if let Some(cg) = constraint {
            cg.apply_mask(&last_logits).map_err(EngineError::Mlx)?
        } else {
            last_logits
        };

        let current_token = sample(&constrained_logits, params).map_err(EngineError::Mlx)?;

        let logprob_data = if let Some(top_n) = logprob_top_n {
            let scaled = if params.temperature <= f32::EPSILON {
                constrained_logits
            } else {
                constrained_logits
                    .multiply(Array::from_f32(1.0 / params.temperature))
                    .map_err(EngineError::Mlx)?
            };
            Some(
                LogprobArrays::compute(&scaled, &current_token, Some(top_n))
                    .map_err(EngineError::Mlx)?,
            )
        } else {
            None
        };

        {
            let mut eval_targets: Vec<&Array> = vec![&current_token];
            if let Some(ref lp) = logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            if let Some(ref hidden) = prefill_hidden {
                eval_targets.push(hidden);
            }
            eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        // Skip prefix cache for multimodal (image-specific KV states)
        if store_prefix_cache && prepared.pixel_values.is_none() {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            // Strip the generation-prompt suffix from the key so multi-turn
            // conversations share their common history prefix (the suffix tokens
            // `<|im_start|>assistant\n…` change between turns).
            //
            // Dense KV caches block-page: the suffix is dropped at block
            // boundaries and reconstruction stays exact.
            //
            // Hybrid (GDN/SSM) caches are stored as a whole CLONE and their
            // sequential state cannot be truncated after the fact (mlx-lm #980).
            // `run_prefill` therefore two-phase splits a hybrid prefill: it
            // snapshots `stored_clone` at the conversation boundary (BEFORE the
            // suffix) and keys that snapshot at the stripped length. The
            // snapshot's offset == stripped key length, so reuse is exact (no
            // RoPE/SSM shift) AND cross-turn matching fires (the conversation
            // boundary recurs verbatim in the next turn). This replaces the old
            // "key hybrid at full length" path, which was correct but never
            // matched cross-turn (the suffix diverges), forcing a full re-prefill
            // every turn.
            let stripped =
                exact_generation_body(prompt_tokens, generation_suffix).unwrap_or(prompt_tokens);
            let cache_to_store = prepared.stored_clone.as_ref().unwrap_or(&prepared.cache);
            pc.store(stripped, cache_to_store, checkpoint_id);
        }
        maybe_clear_mlx_cache(
            self.tuning.clear_cache_after_prefill(),
            "simple_post_prefill",
        );

        Ok((current_token, logprob_data, prefill_hidden))
    }

    /// Env-gated, runs-once flag for the store-drift diagnostic probe. Only
    /// fires once the body length reaches `HIGGS_DIAG_STORE_DRIFT_MIN_LEN`
    /// (default 0), so it can target the large-body regime.
    fn probe_store_drift_enabled(split_at: usize) -> bool {
        static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let on = std::env::var("HIGGS_DIAG_STORE_DRIFT")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let min_len = std::env::var("HIGGS_DIAG_STORE_DRIFT_MIN_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        on && split_at >= min_len && !FIRED.swap(true, Ordering::Relaxed)
    }

    /// Env-gated flag for the warm-path drift probe. Fires on EVERY turn where
    /// reuse is active (prepared cache offset > 0) and the full prompt reaches
    /// `HIGGS_DIAG_WARM_DRIFT_MIN_LEN` (default 400) — to track per-turn
    /// accumulation.
    fn probe_warm_drift_enabled(prompt_tokens: &[u32], resident: i32) -> bool {
        let on = std::env::var("HIGGS_DIAG_WARM_DRIFT")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let min_len = std::env::var("HIGGS_DIAG_WARM_DRIFT_MIN_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(400);
        on && resident > 0 && prompt_tokens.len() >= min_len
    }

    /// Warm-path drift diagnostic: forward the FULL prompt on a fresh cache and
    /// compare the reused clone's KV (prepared.cache[..offset]) to the fresh full
    /// forward's prefix KV. If they differ, the warm incremental reuse path
    /// (clone built over many turns) drifts from a cold full forward.
    // Diagnostic-only probe (env-gated, runs once): stderr reporting and direct
    // indexing over probe-owned buffers are the point, not a hazard.
    #[allow(
        clippy::print_stderr,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::shadow_unrelated,
        clippy::shadow_reuse
    )]
    fn probe_warm_drift(&self, prompt_tokens: &[u32], prepared: &mut PreparedGeneration<'_>) {
        use higgs_models::LayerCache;
        let resident = prepared.cache.resident_len();
        let mut fresh = match prepared
            .model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DIAG warm-drift: make_cache err {e:?}");
                return;
            }
        };
        let full_array = Array::from(prompt_tokens).index(NewAxis);
        // Fresh-vs-fresh control: forward the body and the full prompt on TWO
        // independent fresh caches and compare. (Both caches must be genuinely
        // fresh — forwarding `fresh` twice would pollute its conv_state and
        // confound the comparison.)
        let mut fresh_body = match prepared
            .model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)
        {
            Ok(c) => c,
            Err(_) => {
                eprintln!("DIAG warm-drift: make_cache(fresh_body) err");
                return;
            }
        };
        let body_tokens = &prompt_tokens[..(resident as usize).min(prompt_tokens.len())];
        let body_array = Array::from(body_tokens).index(NewAxis);
        higgs_models::diag_request_hidden_capture();
        higgs_models::diag_request_gdn_capture();
        let _ = prepared
            .model
            .forward_with_hidden(&body_array, None, &mut fresh_body);
        let body_h = higgs_models::diag_take_hidden_capture();
        let body_gdn = higgs_models::diag_take_gdn_capture();
        higgs_models::diag_request_hidden_capture();
        higgs_models::diag_request_gdn_capture();
        let _ = prepared
            .model
            .forward_with_hidden(&full_array, None, &mut fresh);
        let full_h = higgs_models::diag_take_hidden_capture();
        let full_gdn = higgs_models::diag_take_gdn_capture();
        if let (Some(bh), Some(fh)) = (body_h.as_ref(), full_h.as_ref()) {
            higgs_models::diag_report_hidden_diff("WARM-FRESH body-vs-full", bh, fh);
        }
        if let (Some(bg), Some(fg)) = (body_gdn.as_ref(), full_gdn.as_ref()) {
            higgs_models::diag_report_gdn_diff("WARM-FRESH body-vs-full", bg, fg);
        }
        let resident_i = resident;
        if resident_i <= 0 {
            return;
        }
        let cmp = |a: Option<&Array>, b: Option<&Array>| -> (f32, usize, Vec<f32>) {
            let (Some(a), Some(b)) = (a, b) else {
                return (f32::INFINITY, 0, vec![]);
            };
            let af = match a.as_dtype(Dtype::Float32) {
                Ok(v) => v,
                Err(_) => return (f32::INFINITY, 0, vec![]),
            };
            let bf = match b.as_dtype(Dtype::Float32) {
                Ok(v) => v,
                Err(_) => return (f32::INFINITY, 0, vec![]),
            };
            let pa = af.index((.., .., 0..resident_i, ..));
            let pb = bf.index((.., .., 0..resident_i, ..));
            if eval([&pa, &pb]).is_err() {
                return (f32::INFINITY, 0, vec![]);
            }
            let sa = pa.as_slice::<f32>();
            let sb = pb.as_slice::<f32>();
            let d = (*af.shape().get(3).unwrap_or(&1)).max(1) as usize;
            let per_pos = resident_i as usize;
            let mut per_pos_max = vec![0.0f32; per_pos];
            let mut max_abs = 0.0f32;
            let mut diffs = 0usize;
            for (i, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
                let diff = (x - y).abs();
                let pos = (i / d) % per_pos;
                if diff > per_pos_max[pos] {
                    per_pos_max[pos] = diff;
                }
                if diff > max_abs {
                    max_abs = diff;
                }
                if x.to_bits() != y.to_bits() {
                    diffs += 1;
                }
            }
            (max_abs, diffs, per_pos_max)
        };
        let (AnyCache::Hybrid(reused_layers), AnyCache::Hybrid(fresh_layers)) =
            (&prepared.cache, &fresh)
        else {
            eprintln!("DIAG warm-drift: not Hybrid");
            return;
        };
        eprintln!(
            "DIAG warm-drift: comparing reused clone[..{resident_i}] vs fresh full({}) forward",
            prompt_tokens.len()
        );
        let mut global_max = 0.0f32;
        let mut all_per_pos: Vec<f32> = Vec::new();
        for (i, (rl, fl)) in reused_layers.iter().zip(fresh_layers.iter()).enumerate() {
            let (Some(LayerCache::KV(rk)), Some(LayerCache::KV(fk))) = (rl, fl) else {
                continue;
            };
            let (kmax, _kdiff, kpos) = cmp(rk.keys(), fk.keys());
            let (vmax, _vdiff, _vpos) = cmp(rk.values(), fk.values());
            global_max = global_max.max(kmax).max(vmax);
            if i == 3 {
                // Capture the first FA layer's per-position pattern (layer 3).
                all_per_pos = kpos;
            }
        }
        // Characterize the huge-tiny-huge band: count positions in tiny (<0.1)
        // vs huge (>1.0) buckets, and report the longest contiguous tiny run.
        let tiny = all_per_pos.iter().filter(|m| **m < 0.1).count();
        let huge = all_per_pos.iter().filter(|m| **m > 1.0).count();
        let mut best_run = 0usize;
        let mut cur = 0usize;
        let mut best_start = 0usize;
        let mut cur_start = 0usize;
        for (p, m) in all_per_pos.iter().enumerate() {
            if *m < 0.1 {
                if cur == 0 {
                    cur_start = p;
                }
                cur += 1;
                if cur > best_run {
                    best_run = cur;
                    best_start = cur_start;
                }
            } else {
                cur = 0;
            }
        }
        eprintln!(
            "DIAG warm-drift SUMMARY: global_max={global_max:.3e} resident={resident_i} tiny(<0.1)={tiny}/{} huge(>1)={huge} longest_tiny_run=p{best_start}..{}",
            all_per_pos.len(),
            best_start + best_run
        );
        // Fresh-vs-fresh control: fresh_body(=forward body) vs fresh_full[..resident].
        let mut ff_max = 0.0f32;
        if let (AnyCache::Hybrid(bl), AnyCache::Hybrid(fl)) = (&fresh_body, &fresh) {
            for (bl, fl) in bl.iter().zip(fl.iter()) {
                let (Some(LayerCache::KV(bk)), Some(LayerCache::KV(fk))) = (bl, fl) else {
                    continue;
                };
                let (km, _, _) = cmp(bk.keys(), fk.keys());
                let (vm, _, _) = cmp(bk.values(), fk.values());
                ff_max = ff_max.max(km).max(vm);
            }
        }
        eprintln!(
            "DIAG warm-drift FRESH-vs-FRESH (forward(body) vs forward(full)[..body]): max={ff_max:.3e} — clean=>store corrupts clone; dirty=>forward length-dependent at this size"
        );
    }

    /// Store-side drift diagnostic (see [`Self::probe_store_drift_enabled`]).
    /// Forwards the full prompt on a second fresh cache (mirroring a cold
    /// single-pass) and compares `[..split_at]` KV to the stored body snapshot.
    // Diagnostic-only probe — see `probe_warm_drift`.
    #[allow(
        clippy::print_stderr,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::shadow_unrelated,
        clippy::shadow_reuse
    )]
    fn probe_store_drift(&self, prepared: &mut PreparedGeneration<'_>, split_at: usize) {
        use higgs_models::LayerCache;

        // Capture per-layer hidden states for the FULL forward (cold-like).
        higgs_models::diag_request_hidden_capture();
        higgs_models::diag_request_attn_capture();
        let mut fresh = match prepared
            .model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DIAG store-drift: make_cache err {e:?}");
                return;
            }
        };
        let chunk_body = std::env::var("HIGGS_DIAG_CHUNK_BODY").is_ok_and(|v| v == "1");
        let chunk_sz = std::env::var("HIGGS_DIAG_CHUNK_SZ")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(self.tuning.chunked_prefill_chunk_size());
        let fwd_diag = |model: &mut AnyModel, inputs: &Array, cache: &mut AnyCache| -> bool {
            if chunk_body {
                model.forward_chunked(inputs, cache, chunk_sz).is_ok()
            } else {
                model.forward_with_hidden(inputs, None, cache).is_ok()
            }
        };
        if !fwd_diag(&mut prepared.model, &prepared.prompt_array, &mut fresh) {
            eprintln!("DIAG store-drift: forward(full) failed");
            return;
        }
        let full_hidden = higgs_models::diag_take_hidden_capture();
        let full_attn = higgs_models::diag_take_attn_capture();
        // deep_clone `fresh` IMMEDIATELY after its forward, mirroring how
        // production `snap` is captured. The cache stores LAZY slice_update
        // nodes; reading `fresh` later (after other forwards) would let MLX
        // recycle its temporaries and corrupt the comparison. This clone gives a
        // concrete, independent snapshot to compare against `snap`.
        let fresh_snapshot = fresh.deep_clone();
        let Some(snap) = prepared.stored_clone.as_ref() else {
            return;
        };
        let split_i = i32::try_from(split_at).unwrap_or(i32::MAX);
        // CONTROL A (determinism): forward the BODY twice on two fresh caches.
        // If their KV differs, the drift is Metal run-to-run nondeterminism,
        // NOT a seq-len effect — and Route A cannot fix nondeterminism.
        let body_arr = Array::from(&prepared.actual_prompt_tokens[..split_at]).index(NewAxis);
        let mut body_a = match prepared
            .model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DIAG controlA: make_cache err {e:?}");
                return;
            }
        };
        let mut body_b = match prepared
            .model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DIAG controlA: make_cache err {e:?}");
                return;
            }
        };
        // Capture per-layer hidden states for the BODY forward (store-like).
        higgs_models::diag_request_hidden_capture();
        higgs_models::diag_request_attn_capture();
        let _ = fwd_diag(&mut prepared.model, &body_arr, &mut body_a);
        let body_hidden = higgs_models::diag_take_hidden_capture();
        let body_attn = higgs_models::diag_take_attn_capture();
        let _ = prepared
            .model
            .forward_with_hidden(&body_arr, None, &mut body_b);
        // Direct per-layer hidden-state diff: forward(body=90) vs forward(full=95),
        // the exact pair that the KV store-drift measures. Localizes the layer
        // where the body's hidden state first diverges.
        if let (Some(bh), Some(fh)) = (body_hidden.as_ref(), full_hidden.as_ref()) {
            higgs_models::diag_report_hidden_diff("BODY-vs-FULL", bh, fh);
        }
        // Direct first-FA-layer keys diff: pre-write (k_proj+rope) and post-write
        // (cache-stored). If PRE-WRITE differs, h differs (or k_proj/rope is
        // length-dependent). If PRE-WRITE is identical but POST-WRITE differs,
        // the cache write path (update_and_view / mlx_slice_update) corrupts.
        if let (Some(ba), Some(fa)) = (body_attn.as_ref(), full_attn.as_ref()) {
            higgs_models::diag_report_attn_diff("BODY-vs-FULL", ba, fa);
        }
        // DECISIVE: for the FULL forward, compare PRE-write keys (k_proj+rope,
        // eval'd directly) vs POST-write keys (the slice_update result) of the
        // SAME forward. If they differ, `slice_update` corrupts the
        // non-contiguous keys when materializing — the cache write is the bug.
        if let Some((_, Some(pre), Some(post), _, _, _)) = full_attn.as_ref() {
            let a = &pre.0;
            let b = &post.0;
            let d = (*pre.1.get(3).unwrap_or(&1)).max(1) as usize;
            let per_pos = split_at;
            let elems = per_pos * d;
            let mut max_abs = 0.0f32;
            let mut diffs = 0usize;
            let mut per_pos_max = vec![0.0f32; per_pos];
            for i in 0..elems.min(a.len()).min(b.len()) {
                let diff = (a[i] - b[i]).abs();
                let pos = i / d;
                if diff > per_pos_max[pos] {
                    per_pos_max[pos] = diff;
                }
                if diff > max_abs {
                    max_abs = diff;
                }
                if a[i].to_bits() != b[i].to_bits() {
                    diffs += 1;
                }
            }
            let nz: Vec<String> = per_pos_max
                .iter()
                .enumerate()
                .filter(|(_, m)| **m > 0.0)
                .map(|(p, m)| format!("p{p}:{m:.1e}"))
                .collect();
            eprintln!(
                "DIAG PRE-vs-POST same-forward(full): max_abs={max_abs:.3e} diffs={diffs}/{elems} nonzero[{}]",
                nz.join(" ")
            );
        }
        // DECISIVE: does `fresh`'s first-FA cache change between immediate
        // post-write (full_attn capture, mid-forward) and post-forward
        // (fresh_snapshot, deep_clone after the forward)? If yes, the cache is
        // corrupted during the rest of the forward. Compare full_attn.POST-WRITE
        // (Vec) vs fresh_snapshot's first-FA-layer keys read now.
        if let Some((_, _, Some((imm_v, imm_shape)), _, _, _)) = full_attn.as_ref() {
            let AnyCache::Hybrid(fresh_layers) = &fresh_snapshot else {
                eprintln!("DIAG SELF: fresh_snapshot not Hybrid");
                return;
            };
            for fl in fresh_layers.iter() {
                let Some(higgs_models::LayerCache::KV(fk)) = fl else {
                    continue;
                };
                let Some(fka) = fk.keys() else { break };
                let Ok(fkaf) = fka.as_dtype(Dtype::Float32) else {
                    break;
                };
                let pa = fkaf.index((.., .., 0..split_i, ..));
                let _ = eval([&pa]);
                let now_v = pa.as_slice::<f32>().to_vec();
                let d = (*imm_shape.get(3).unwrap_or(&1)).max(1) as usize;
                let per_pos = split_at;
                let elems = per_pos * d;
                let mut max_abs = 0.0f32;
                let mut diffs = 0usize;
                let mut nz: Vec<String> = Vec::new();
                let mut per_pos_max = vec![0.0f32; per_pos];
                for i in 0..elems.min(imm_v.len()).min(now_v.len()) {
                    let diff = (imm_v[i] - now_v[i]).abs();
                    let pos = i / d;
                    if diff > per_pos_max[pos] {
                        per_pos_max[pos] = diff;
                    }
                    if diff > max_abs {
                        max_abs = diff;
                    }
                    if imm_v[i].to_bits() != now_v[i].to_bits() {
                        diffs += 1;
                    }
                }
                for (p, m) in per_pos_max.iter().enumerate() {
                    if *m > 0.0 {
                        nz.push(format!("p{p}:{m:.1e}"));
                    }
                }
                eprintln!(
                    "DIAG SELF fresh first-FA: immediate-vs-postforward max_abs={max_abs:.3e} diffs={diffs}/{elems} nonzero[{}]",
                    nz.join(" ")
                );
                break;
            }
        }
        let cmp = |label: &str, a: Option<&Array>, b: Option<&Array>| -> (bool, f32, usize) {
            let (Some(a), Some(b)) = (a, b) else {
                eprintln!("DIAG {label}: missing array");
                return (false, 0.0, 0);
            };
            // Cast to f32 (lossless for bf16->f32; no-op for f32). Comparing
            // the cast bits is still bit-exact: equal bf16 -> equal f32.
            let a = match a.as_dtype(Dtype::Float32) {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("DIAG {label}: as_dtype f32 failed");
                    return (false, 0.0, 0);
                }
            };
            let b = match b.as_dtype(Dtype::Float32) {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("DIAG {label}: as_dtype f32 failed");
                    return (false, 0.0, 0);
                }
            };
            let pa = a.index((.., .., 0..split_i, ..));
            let pb = b.index((.., .., 0..split_i, ..));
            if eval([&pa, &pb]).is_err() {
                eprintln!("DIAG {label}: eval failed");
                return (false, 0.0, 0);
            }
            let sa = pa.as_slice::<f32>();
            let sb = pb.as_slice::<f32>();
            if sa.len() != sb.len() {
                eprintln!(
                    "DIAG {label}: shape mismatch {} vs {} (split={split_i})",
                    sa.len(),
                    sb.len()
                );
                return (false, f32::INFINITY, sa.len().max(sb.len()));
            }
            let mut exact = true;
            let mut max_abs = 0.0f32;
            let mut diffs = 0usize;
            for (x, y) in sa.iter().zip(sb.iter()) {
                if x.to_bits() != y.to_bits() {
                    exact = false;
                    diffs += 1;
                }
                let d = (x - y).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            eprintln!(
                "DIAG {label}: bit_exact={exact} max_abs={max_abs:.3e} diff_elems={diffs}/{}",
                sa.len()
            );
            (exact, max_abs, diffs)
        };
        let (AnyCache::Hybrid(fresh_layers), AnyCache::Hybrid(snap_layers)) =
            (&fresh_snapshot, snap)
        else {
            eprintln!("DIAG store-drift: cache not Hybrid, skipping");
            return;
        };
        // Dump raw values: first-FA-layer keys, position 0, head 0, first 4 elems
        // for fresh_snapshot vs snap vs full_attn.POST (X). Rules out comparison
        // artifacts and shows whose data is actually wrong.
        for (fl, sl) in fresh_layers.iter().zip(snap_layers.iter()) {
            let (Some(LayerCache::KV(fk)), Some(LayerCache::KV(sk))) = (fl, sl) else {
                continue;
            };
            let dump = |label: &str, a: Option<&Array>| -> String {
                let Some(a) = a else {
                    return format!("{label}: none");
                };
                let Ok(af) = a.as_dtype(Dtype::Float32) else {
                    return format!("{label}: dtype err");
                };
                let p = af.index((.., .., 0..1, ..));
                let _ = eval([&p]);
                let s = p.as_slice::<f32>();
                format!(
                    "{label} offset={} [{}]",
                    a.shape()[2],
                    s.iter()
                        .take(4)
                        .map(|v| format!("{v:.3}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            eprintln!(
                "DIAG VALS first-FA p0: {} | {}",
                dump("fresh", fk.keys()),
                dump("snap", sk.keys())
            );
            break;
        }
        // Localize the diff per position (axis 2) for the FIRST KV layer keys,
        // collapsing across heads/head-dim, to see if it concentrates at the
        // body tail (windowing/offset leak) vs spread (tiling) vs misaligned.
        for (fl, sl) in fresh_layers.iter().zip(snap_layers.iter()) {
            let (Some(LayerCache::KV(fk)), Some(LayerCache::KV(sk))) = (fl, sl) else {
                continue;
            };
            let (Some(ak), Some(bk)) = (fk.keys(), sk.keys()) else {
                break;
            };
            let (Ok(af), Ok(bf)) = (ak.as_dtype(Dtype::Float32), bk.as_dtype(Dtype::Float32))
            else {
                break;
            };
            let pa = af.index((.., .., 0..split_i, ..));
            let pb = bf.index((.., .., 0..split_i, ..));
            let _ = eval([&pa, &pb]);
            let shape = pa.shape();
            let h = *shape.get(1).unwrap_or(&1);
            let split = *shape.get(2).unwrap_or(&1);
            let d = *shape.get(3).unwrap_or(&1).max(&1);
            let sa = pa.as_slice::<f32>();
            let sb = pb.as_slice::<f32>();
            let mut per_pos_max = vec![0.0f32; split as usize];
            let mut per_pos_diffs = vec![0u32; split as usize];
            for (i, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
                let pos = ((i as i32 / d) % split) as usize;
                let dd = (x - y).abs();
                if dd > per_pos_max[pos] {
                    per_pos_max[pos] = dd;
                }
                if x.to_bits() != y.to_bits() {
                    per_pos_diffs[pos] += 1;
                }
            }
            eprintln!(
                "DIAG LOC first-KV-keys: H={h} split={split} D={d} fresh.offset={} snap.offset={}",
                fresh_snapshot.resident_len(),
                snap.resident_len()
            );
            // Print compact per-position summary: only positions with nonzero diff.
            let nz: Vec<String> = per_pos_max
                .iter()
                .enumerate()
                .filter(|&(_, &m)| m > 0.0)
                .map(|(p, &m)| format!("pos{p}:{m:.2e}/{}", per_pos_diffs[p]))
                .collect();
            eprintln!(
                "DIAG LOC nonzero positions (pos:max_abs/diffs): {}",
                nz.join(" ")
            );
            break;
        }
        let mut all_exact = true;
        let mut global_max = 0.0f32;
        for (i, (fl, sl)) in fresh_layers.iter().zip(snap_layers.iter()).enumerate() {
            let (Some(LayerCache::KV(fk)), Some(LayerCache::KV(sk))) = (fl, sl) else {
                continue;
            };
            let (ke, km, _) = cmp(&format!("L{i:02}_keys"), fk.keys(), sk.keys());
            let (ve, vm, _) = cmp(&format!("L{i:02}_vals"), fk.values(), sk.values());
            all_exact &= ke & ve;
            global_max = global_max.max(km).max(vm);
        }
        eprintln!(
            "DIAG store-drift SUMMARY: all_bit_exact={all_exact} global_max_abs={global_max:.3e} (split_at={split_at}, full_len={})",
            prepared.actual_prompt_tokens.len()
        );
        // CONTROL A report: body-vs-body (run-to-run determinism). If this is
        // NOT bit_exact, the engine's forward is nondeterministic and the
        // store-drift above is partly/wholly nondeterminism, not a seq-len effect.
        let (AnyCache::Hybrid(ba), AnyCache::Hybrid(bb)) = (&body_a, &body_b) else {
            return;
        };
        let mut ctrl_exact = true;
        let mut ctrl_max = 0.0f32;
        let mut ctrl_diffs = 0usize;
        for (la, lb) in ba.iter().zip(bb.iter()) {
            let (Some(LayerCache::KV(ka)), Some(LayerCache::KV(kb))) = (la, lb) else {
                continue;
            };
            let (ke, km, kd) = cmp("CTRL", ka.keys(), kb.keys());
            let (ve, vm, vd) = cmp("CTRL", ka.values(), kb.values());
            ctrl_exact &= ke & ve;
            ctrl_max = ctrl_max.max(km).max(vm);
            ctrl_diffs += kd + vd;
        }
        eprintln!(
            "DIAG CONTROL-A (body-vs-body determinism): bit_exact={ctrl_exact} max_abs={ctrl_max:.3e} total_diffs={ctrl_diffs}"
        );
        // CONTROL-B (prepared.cache cleanliness): compare snap (body forward on
        // prepared.cache) vs body_a (body forward on a truly fresh cache). If
        // they differ, prepared.cache was NOT empty before the body forward —
        // residual tokens shifted the KV layout, and the "store drift" is an
        // unclean-cache artifact, not a seq-len tiling effect.
        let (AnyCache::Hybrid(snap_layers), AnyCache::Hybrid(ba_layers)) = (snap, &body_a) else {
            return;
        };
        let mut cb_exact = true;
        let mut cb_max = 0.0f32;
        let mut cb_diffs = 0usize;
        for (ls, la) in snap_layers.iter().zip(ba_layers.iter()) {
            let (Some(LayerCache::KV(ks)), Some(LayerCache::KV(ka))) = (ls, la) else {
                continue;
            };
            let (ke, km, kd) = cmp("CTRLB-K", ks.keys(), ka.keys());
            let (ve, vm, vd) = cmp("CTRLB-V", ks.values(), ka.values());
            cb_exact &= ke & ve;
            cb_max = cb_max.max(km).max(vm);
            cb_diffs += kd + vd;
        }
        eprintln!(
            "DIAG CONTROL-B (snap-vs-freshbody, prepared.cache clean?): bit_exact={cb_exact} max_abs={cb_max:.3e} total_diffs={cb_diffs}"
        );
        // VARY-SUFFIX SWEEP: forward body+K for K in {2,5,10} on fresh caches,
        // compare [..split_at] KV to body_a. If appending K tokens changes the
        // FIRST K body positions, the model has a structural non-causal op
        // (Route A chunking cannot fix that). If it changes the LAST ~K (near
        // the boundary) or spreads as tiny FP tiling, it's a tiling effect.
        let total = prepared.actual_prompt_tokens.len();
        for k in [2usize, 5, 10] {
            let end = (split_at + k).min(total);
            if end <= split_at {
                continue;
            }
            let arr = Array::from(&prepared.actual_prompt_tokens[..end]).index(NewAxis);
            let mut c = match prepared
                .model
                .make_cache_with_config(self.kv_cache_config)
                .map_err(EngineError::Mlx)
            {
                Ok(c) => c,
                Err(_) => continue,
            };
            let _ = prepared.model.forward_with_hidden(&arr, None, &mut c);
            let (AnyCache::Hybrid(cl), AnyCache::Hybrid(bl)) = (&c, &body_a) else {
                continue;
            };
            // Localize on the first KV layer keys.
            for (cf, bf) in cl.iter().zip(bl.iter()) {
                let (Some(LayerCache::KV(ck)), Some(LayerCache::KV(bk))) = (cf, bf) else {
                    continue;
                };
                let (Some(ak), Some(bk2)) = (ck.keys(), bk.keys()) else {
                    break;
                };
                let (Ok(af), Ok(bf2)) = (ak.as_dtype(Dtype::Float32), bk2.as_dtype(Dtype::Float32))
                else {
                    break;
                };
                let pa = af.index((.., .., 0..split_i, ..));
                let pb = bf2.index((.., .., 0..split_i, ..));
                let _ = eval([&pa, &pb]);
                let shape = pa.shape();
                let d = *shape.get(3).unwrap_or(&1).max(&1);
                let split = *shape.get(2).unwrap_or(&1);
                let sa = pa.as_slice::<f32>();
                let sb = pb.as_slice::<f32>();
                let mut per_pos_max = vec![0.0f32; split as usize];
                for (i, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
                    let pos = ((i as i32 / d) % split) as usize;
                    let dd = (x - y).abs();
                    if dd > per_pos_max[pos] {
                        per_pos_max[pos] = dd;
                    }
                }
                let nz: Vec<String> = per_pos_max
                    .iter()
                    .enumerate()
                    .filter(|&(_, &m)| m > 0.0)
                    .map(|(p, &m)| format!("pos{p}:{m:.1e}"))
                    .collect();
                eprintln!(
                    "DIAG SWEEP body+{k} (len={end}) vs body: first-KV-keys nonzero positions: {}",
                    nz.join(" ")
                );
                break;
            }
        }
    }

    /// Decode a single step: forward pass on the current token, apply penalties
    /// and optional constraint mask, then sample. Returns `(next_token, Option<LogprobArrays>)`.
    fn decode_step(
        current_token: &Array,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        params: &SamplingParams,
        generated_tokens: &[u32],
        logprob_top_n: Option<u32>,
        constraint: Option<&crate::constrained::ConstrainedGenerator>,
    ) -> Result<(Array, Option<LogprobArrays>), EngineError> {
        let decode_input = current_token.index((.., NewAxis));
        let logits = model
            .forward(&decode_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let sliced = logits.index((.., -1, ..));

        let penalized =
            apply_penalties(&sliced, generated_tokens, params).map_err(EngineError::Mlx)?;

        // Apply constraint mask if structured output is requested
        let constrained = if let Some(cg) = constraint {
            cg.apply_mask(&penalized).map_err(EngineError::Mlx)?
        } else {
            penalized
        };

        let next_token = sample(&constrained, params).map_err(EngineError::Mlx)?;

        let logprob_data = if let Some(top_n) = logprob_top_n {
            // Compute logprobs from the same distribution we sampled from.
            // Temperature is already accounted for inside `sample`, so we
            // replicate the scaling here for the logprob computation.
            let scaled = if params.temperature <= f32::EPSILON {
                constrained
            } else {
                constrained
                    .multiply(mlx_rs::array!(1.0 / params.temperature))
                    .map_err(EngineError::Mlx)?
            };
            Some(
                LogprobArrays::compute(&scaled, &next_token, Some(top_n))
                    .map_err(EngineError::Mlx)?,
            )
        } else {
            None
        };

        Ok((next_token, logprob_data))
    }

    /// Cache-resident multi-turn generation: keep the conversation's KV cache
    /// alive across tool hops and prefill ONLY the new suffix on the next turn
    /// (no re-prefill of history). Opt-in per conversation via `session_id`. On
    /// the first turn — or a conversation that diverged from the retained cache —
    /// it falls back to a clean full prefill (the [`take_continuable`] guard) and
    /// retains the resulting cache; on a continuation it reuses the live cache.
    /// Greedy decode with MTP speculation when the model ships a head (the
    /// stop-aware cycle keeps the retained cache 1:1 with the stashed tokens);
    /// plain sequential decode otherwise.
    ///
    /// # Best-effort: output is approximate, not bit-identical
    ///
    /// This per-session path is a **latency optimization, not an exact-replay
    /// guarantee**. A continued turn's output can differ slightly from a stateless
    /// full prefill of the same conversation, because (1) when TurboQuant is
    /// enabled, the retained KV is compressed for between-turn storage
    /// (`quantize_for_retention`), which is lossy (with `kv_cache = off` the
    /// retained KV stays dense and exact), and (2) the continuation prompt is
    /// reconciled in text space
    /// (decode the retained tokens, strip `<think>` blocks, re-match the prefix),
    /// which can diverge from a cleanly-rendered stateless prompt.
    ///
    /// For dense KV models, the normal radix prefix cache reconstructs cached KV
    /// exactly. For Hybrid models, both radix and session continuation are
    /// latency optimizations with bounded deterministic drift versus a cold
    /// one-shot full prefill; the session path is the right option when the user
    /// experience depends on avoiding repeated long-context prefill.
    ///
    /// [`take_continuable`]: Self::take_continuable
    pub fn generate_continued(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
    ) -> Result<SessionGeneration, EngineError> {
        self.generate_continued_with_thinking(
            session_id,
            prompt_tokens,
            max_tokens,
            params,
            self.enable_thinking,
        )
    }

    /// Cache-resident generation with request-scoped thinking control.
    ///
    /// The same flag must be used to render `prompt_tokens` and to select the
    /// decode path. In particular, an explicit no-thinking request may enter
    /// the exact paired dSpark domain even when the engine's legacy default is
    /// thinking-enabled.
    pub fn generate_continued_with_thinking(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        enable_thinking: bool,
    ) -> Result<SessionGeneration, EngineError> {
        let timing = std::env::var("HIGGS_DIAG_SESSION_TIMING").is_ok_and(|v| v == "1");
        let total_start = std::time::Instant::now();
        // Serialize all work for this conversation: hold the per-session lock for
        // the entire call so a second concurrent request for the same session_id
        // queues here and only proceeds once this one has stashed its result
        // (it then continues from that result, or full-prefills if it diverged).
        // Acquired BEFORE the model lock — the global order is session -> model,
        // so this can never deadlock. The map lock itself is released immediately;
        // only the per-session lock is held across the body.
        let session_lock = std::sync::Arc::clone(
            lock_or_recover(&self.session_locks)
                .entry(session_id)
                .or_default(),
        );
        let _session_guard = session_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Opportunistic idle eviction: free retained caches abandoned longer than
        // the configured TTL. Cheap (bounded by max_retained_sessions) and runs
        // on each cache-resident request, so memory is reclaimed without a
        // background task. 0 = disabled.
        let idle_secs = self.kv_cache_config.retained_idle_secs;
        if idle_secs > 0 {
            self.evict_idle_retained(std::time::Duration::from_secs(idle_secs));
        }

        let total = u32::try_from(prompt_tokens.len())
            .map_err(|_| EngineError::Generation("prompt too long".to_owned()))?;
        let has_mtp = {
            let model = self
                .model
                .lock()
                .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
            model.has_mtp()
        };
        let speculation_route = resolve_speculation_route(
            params.speculation,
            self.dflash.is_some() && paired_dflash_exact_domain(params, enable_thinking),
            has_mtp,
        );
        if speculation_route == SpeculationRoute::DFlash {
            return self.generate_continued_dflash_locked(
                session_id,
                prompt_tokens,
                max_tokens,
                params,
                timing,
                total_start,
            );
        }

        let (mut prepared, prefilled, continued) = if let Some((state, prior)) =
            self.take_continuable(session_id, prompt_tokens)
        {
            let cache = state.demote();
            debug_assert!(
                prior <= prompt_tokens.len(),
                "continuation prior {prior} exceeds prompt length {}",
                prompt_tokens.len()
            );
            let suffix: Vec<u32> = prompt_tokens.get(prior..).unwrap_or_default().to_vec();
            let prefilled = u32::try_from(suffix.len()).unwrap_or(u32::MAX);
            let prompt_array = Array::from(suffix.as_slice()).index(NewAxis);
            let model = self
                .model
                .lock()
                .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
            // Same single sanctioned MLX-gate acquisition as prepare_generation —
            // the continuation path builds PreparedGeneration by hand, so it must
            // take the gate too or its eval would fire the off-gate assert.
            let mlx_gate = higgs_models::mlx_exec::acquire();
            let prepared = PreparedGeneration {
                model,
                cache,
                actual_prompt_tokens: suffix,
                prompt_array,
                prompt_len: total,
                pixel_values: None,
                stored_clone: None,
                _mlx_gate: mlx_gate,
            };
            (prepared, prefilled, true)
        } else {
            let prepared = self.prepare_generation(prompt_tokens, None, None)?;
            let prefilled = u32::try_from(prepared.actual_prompt_tokens.len()).unwrap_or(u32::MAX);
            (prepared, prefilled, false)
        };

        if continued {
            self.cache_metrics
                .continuations
                .fetch_add(1, Ordering::Relaxed);
            self.cache_metrics.prefill_saved_tokens.fetch_add(
                u64::from(total.saturating_sub(prefilled)),
                Ordering::Relaxed,
            );
        }

        let prefill_start = std::time::Instant::now();
        // capture_hidden: the MTP head primes from the prefilled tokens'
        // hidden states (unprimed acceptance measured ~75% vs 85-97% primed).
        // Chunked long prefills return no hidden — priming is then skipped.
        let want_mtp = speculation_route == SpeculationRoute::MtpFamily
            && prepared.model.has_mtp()
            && max_tokens > 1;
        // dSpark sessions returned through the paired-cache path above. The
        // remaining selectors are deliberately sidecar-free MTP or AR.
        let (current_token, _, prefill_hidden) = self.run_prefill(
            prompt_tokens,
            self.gen_prompt_suffixes.for_request(enable_thinking),
            &mut prepared,
            params,
            None,
            None,
            want_mtp,
            false,
            None,
            None,
            None,
        )?;
        let prefill_elapsed = prefill_start.elapsed();
        let first_id: u32 = current_token.item();
        let mut generated: Vec<u32> = vec![first_id];
        let mut token_ledger = TokenLedger::new(prompt_tokens.len());
        token_ledger
            .emit_pending(first_id)
            .map_err(session_ledger_error)?;

        let decode_start = std::time::Instant::now();
        // MTP speculative decode for the session path when the model ships a
        // head. Verify targets are sampled at the request temperature when it
        // is nonzero (`verify_targets`), so speculation preserves the output
        // distribution — drafts only decide how many positions commit per
        // cycle. The stop-aware cycle (`mtp_cycle_bounded`) never advances the
        // backbone past a stop token, so the retained cache stays 1:1 with the
        // stashed token list after the EOS pop below — the invariant
        // sequential decode provided for free. Falls back to sequential
        // decode for dense models without a head.
        let mtp_cache = (want_mtp && !self.eos_token_ids.contains(&first_id))
            .then(|| prepared.model.make_mtp_cache())
            .flatten();
        let mut mtp_stats = crate::mtp::MtpStats::default();
        if let Some(mut session_mtp_cache) = mtp_cache {
            // Prime the head over the freshly-prefilled tokens, then mirror
            // the first emitted token — same warm-up `mtp_generate` does.
            if let Some(ref hidden) = prefill_hidden {
                crate::mtp::prime_mtp_cache(
                    &mut prepared.model,
                    &mut session_mtp_cache,
                    &prepared.actual_prompt_tokens,
                    hidden,
                )?;
            }
            // Bootstrap identical to `mtp_generate`: forward the already-
            // emitted first token, keep its hidden, sample the next confirmed
            // token (pending, NOT emitted — the first cycle emits it).
            let first_input = Array::from_slice(
                &[i32::try_from(first_id)
                    .map_err(|_| EngineError::Generation("first token overflow i32".to_owned()))?],
                &[1, 1],
            );
            let first_forward = token_ledger
                .begin_cache_only_forward()
                .map_err(session_ledger_error)?;
            let (hidden, logits) = prepared
                .model
                .forward_with_hidden(&first_input, None, &mut prepared.cache)
                .map_err(EngineError::Mlx)?;
            token_ledger
                .complete_cache_only_forward(first_forward)
                .map_err(session_ledger_error)?;
            if let Some(prev_hidden) = prefill_hidden
                .as_ref()
                .filter(|_| !prepared.actual_prompt_tokens.is_empty())
                .map(|prefill| {
                    Self::hidden_row_from_sequence(prefill, prepared.actual_prompt_tokens.len() - 1)
                })
                .transpose()?
            {
                crate::mtp::mirror_mtp_token(
                    &mut prepared.model,
                    &mut session_mtp_cache,
                    &prev_hidden,
                    first_id,
                )?;
            }
            let boot_logits = apply_penalties(&logits.index((.., -1, ..)), &generated, params)
                .map_err(EngineError::Mlx)?;
            let next_arr = sample(&boot_logits, params).map_err(EngineError::Mlx)?;
            let h = hidden.index((.., -1.., ..));
            eval([&next_arr, &h]).map_err(EngineError::Mlx)?;
            let mut current_hidden = h;
            let mut confirmed: u32 = next_arr.item();

            loop {
                let completion = u32::try_from(generated.len()).unwrap_or(u32::MAX);
                if completion >= max_tokens {
                    break; // pending token dropped, unforwarded — aligned.
                }
                if self.eos_token_ids.contains(&confirmed) {
                    // Pending stop token: emit without forwarding. The token
                    // ledger excludes it from the retained key below.
                    generated.push(confirmed);
                    token_ledger
                        .emit_pending(confirmed)
                        .map_err(session_ledger_error)?;
                    break;
                }

                // The session selector is already resolved before prefill, so
                // this is the ordinary MTP cycle with no hidden dSpark sidecar.
                let remaining = usize::try_from(max_tokens.saturating_sub(completion))
                    .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
                let draft_depth = self
                    .tuning
                    .mtp_draft_n_max()
                    .min(remaining.saturating_sub(1).max(1));
                let result = crate::mtp::mtp_cycle_bounded(
                    &mut prepared.model,
                    &mut prepared.cache,
                    &mut session_mtp_cache,
                    &current_hidden,
                    confirmed,
                    draft_depth,
                    &self.eos_token_ids,
                    Some(params),
                    &generated,
                )?;
                mtp_stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
                token_ledger
                    .extend_forwarded(result.tokens.iter().copied())
                    .map_err(session_ledger_error)?;
                generated.extend_from_slice(&result.tokens);
                current_hidden = result.hidden;
                confirmed = result.next_token_id;
            }
        } else if !self.eos_token_ids.contains(&first_id) && max_tokens > 1 {
            let mut cur = current_token;
            while u32::try_from(generated.len()).unwrap_or(u32::MAX) < max_tokens {
                let forwarded = token_ledger
                    .begin_cache_only_forward()
                    .map_err(session_ledger_error)?;
                let (next, _) = Self::decode_step(
                    &cur,
                    &mut prepared.model,
                    &mut prepared.cache,
                    params,
                    &generated,
                    None,
                    None,
                )?;
                token_ledger
                    .complete_cache_only_forward(forwarded)
                    .map_err(session_ledger_error)?;
                let next_id: u32 = next.item();
                generated.push(next_id);
                token_ledger
                    .emit_pending(next_id)
                    .map_err(session_ledger_error)?;
                if self.eos_token_ids.contains(&next_id) {
                    break;
                }
                cur = next;
            }
        }
        let decode_elapsed = decode_start.elapsed();
        if mtp_stats.cycles() > 0 {
            tracing::info!(
                session_id,
                cycles = mtp_stats.cycles(),
                drafted = mtp_stats.drafted(),
                accepted_drafts = mtp_stats.accepted_drafts(),
                emitted = mtp_stats.emitted(),
                accept_rate = format!("{:.1}%", mtp_stats.acceptance_rate_percent()),
                "session MTP decode"
            );
        }

        match token_ledger.retention_action(&self.eos_token_ids) {
            RetentionAction::Ready { .. } => {}
            RetentionAction::ExcludeEos { .. } => {
                token_ledger
                    .exclude_pending_eos(&self.eos_token_ids)
                    .map_err(session_ledger_error)?;
            }
            RetentionAction::CacheOnlyForward { token } => {
                let ticket = token_ledger
                    .begin_cache_only_forward()
                    .map_err(session_ledger_error)?;
                let token = i32::try_from(token).map_err(|_| {
                    EngineError::Generation("retained pending token overflow i32".to_owned())
                })?;
                let input = Array::from_slice(&[token], &[1, 1]);
                let _ = prepared
                    .model
                    .forward(&input, None, &mut prepared.cache)
                    .map_err(EngineError::Mlx)?;
                token_ledger
                    .complete_cache_only_forward(ticket)
                    .map_err(session_ledger_error)?;
            }
            RetentionAction::ForwardInFlight { token } => {
                return Err(EngineError::Generation(format!(
                    "session token {token} still has a cache-only forward in flight"
                )));
            }
        }
        debug_assert_eq!(token_ledger.emitted_tokens(), generated);
        let retained_generated = token_ledger
            .retainable_tokens()
            .map_err(session_ledger_error)?
            .to_vec();

        // Retain the live cache + the exact tokens it now holds (prompt +
        // generated) so the next hop continues from here.
        //
        // Compress (TurboQuant) AND evaluate the cache while the model lock is
        // STILL HELD. MLX's Metal command buffer is process-global and aborts on
        // concurrent eval across threads, so every MLX/GPU operation must be
        // serialized by the model Mutex (the engine's de-facto MLX-execution
        // lock). Doing this before `drop(model)` keeps this request's GPU work
        // from racing the next request's forward pass; `stash_retained` then only
        // publishes the already-evaluated cache.
        let PreparedGeneration {
            mut cache,
            model,
            _mlx_gate: mlx_gate,
            ..
        } = prepared;
        let retain_start = std::time::Instant::now();
        // Only compress retained KV when TurboQuant is actually enabled. With
        // mode=Off (the default) the user opted out of quantization — and the
        // compress/dequant round trip is pure loss there: ~2s CPU to pack at
        // turn end, then EVERY full-attention layer re-dequantizes the whole
        // cache on CPU during the next continuation's prefill (measured ~4s of
        // a 4.1s warm-turn TTFT at 4k tokens on Qwen3.6-35B). Dense retention
        // is also exact, removing one documented lossiness source. Resident
        // memory stays bounded by max_retained_sessions / max_session_tokens /
        // the idle TTL, which apply either way.
        // Over the dense token cap, fall back to TurboQuant retention instead
        // of dropping: a 39.5k-token session that lost retention re-prefilled
        // for 149.8s on the next turn, versus a bounded ~4-6x-smaller TQ
        // resident and a CPU dequant on resume (tens of seconds at that size,
        // and GPU dequant work is in flight). Exactness was already
        // best-effort on this path.
        let cap = self.kv_cache_config.max_session_tokens;
        let total_turn_tokens = prompt_tokens.len().saturating_add(retained_generated.len());
        let boundary_valid = i32::try_from(total_turn_tokens).map_or_else(
            |_| {
                tracing::warn!(
                    session_id,
                    retained_tokens = total_turn_tokens,
                    "Retained session boundary exceeds i32; dropping session retention"
                );
                false
            },
            |expected| {
                cache.validate_absolute_boundary(expected).map_or_else(
                    |error| {
                        tracing::warn!(
                            session_id,
                            error = %error,
                            expected,
                            "Retained target cache is misaligned; dropping session retention"
                        );
                        false
                    },
                    |()| true,
                )
            },
        );
        let over_dense_cap = cap > 0 && total_turn_tokens > cap;
        let mut cap_exempt = false;
        if boundary_valid && (self.kv_cache_config.is_turboquant() || over_dense_cap) {
            match cache.quantize_for_retention(self.kv_cache_config) {
                Ok(layers) if layers > 0 => {
                    cap_exempt = over_dense_cap;
                    tracing::debug!(
                        session_id,
                        compressed_layers = layers,
                        over_dense_cap,
                        "Compressed retained KV to TurboQuant for between-turn retention"
                    );
                }
                Ok(_) => {}
                // Leave the cache dense on failure — correctness over footprint.
                Err(e) => tracing::warn!(
                    session_id,
                    error = %e,
                    "Failed to TurboQuant-compress retained KV; retaining dense"
                ),
            }
        }
        let retained_state = if boundary_valid {
            let mut full = prompt_tokens.to_vec();
            full.extend_from_slice(&retained_generated);
            match RetainedState::target_only(cache, full, &mlx_gate) {
                Ok(state) => Some(state),
                Err(error) => {
                    tracing::warn!(
                        session_id,
                        error = %error,
                        "Failed to evaluate and key retained target state; dropping session retention"
                    );
                    None
                }
            }
        } else {
            drop(cache);
            None
        };
        let retain_elapsed = retain_start.elapsed();
        // Release model lock and MLX gate together, only after the last eval.
        drop(model);
        drop(mlx_gate);
        if let Some(state) = retained_state {
            self.stash_retained(session_id, state, cap_exempt);
        }
        let total_elapsed = total_start.elapsed();

        #[allow(clippy::print_stderr)] // env-gated diagnostic
        if timing {
            eprintln!(
                "DIAG session-timing: continued={continued} prompt={total} prefilled={prefilled} generated={} prefill={prefill_elapsed:.2?} decode={decode_elapsed:.2?} retain_eval={retain_elapsed:.2?} total={total_elapsed:.2?}",
                generated.len()
            );
        }

        tracing::info!(
            session_id,
            continued,
            prompt_tokens = total,
            prefilled_tokens = prefilled,
            prefill_saved = total.saturating_sub(prefilled),
            "cache-resident turn"
        );

        Ok(SessionGeneration {
            text: self.decode_tokens(&generated)?,
            completion_tokens: u32::try_from(generated.len()).unwrap_or(u32::MAX),
            prompt_tokens: total,
            prefilled_tokens: prefilled,
            continued,
        })
    }

    /// Convert one live session pair into its sole atomic publication.
    ///
    /// Deterministic tap ineligibility may demote to a validated target-only
    /// publication. Once effectful sealing begins, any error returns `None`.
    fn seal_session_dspark_publication(
        &self,
        session_id: u64,
        pair: LivePair,
        drafter: &mut DFlashDrafter,
        mlx_gate: &higgs_models::mlx_exec::MlxExecToken,
    ) -> Option<SessionDsparkPublication> {
        let retained_boundary = pair.token_len();
        match pair.seal_for_session(drafter, mlx_gate) {
            Ok(publication) => {
                if let Some(reason) = publication.demotion() {
                    tracing::warn!(
                        session_id,
                        ?reason,
                        expected = retained_boundary,
                        "dSpark retention is ineligible; publishing validated target only"
                    );
                }
                Some(publication)
            }
            Err(error) => {
                tracing::warn!(
                    session_id,
                    error = %error,
                    expected = retained_boundary,
                    "Failed to retain exact session dSpark state; publishing nothing"
                );
                None
            }
        }
    }

    /// Publish only a successfully sealed session outcome after all MLX/model
    /// guards have been released. An effectful seal failure is represented by
    /// `None`, so neither the target nor dSpark half can be reintroduced.
    fn publish_session_dspark_publication(
        &self,
        session_id: u64,
        publication: Option<SessionDsparkPublication>,
        cap_exempt: bool,
    ) {
        if let Some(publication) = publication {
            self.stash_retained(session_id, publication.into_state(), cap_exempt);
        }
    }

    /// Exact greedy/no-thinking dSpark session path.
    ///
    /// The caller owns the per-session lock for this entire method. A retained
    /// pair is move-reused only when both private halves match the exact prior
    /// token prefix; target-only state is intentionally discarded because its
    /// historical tap stream cannot reconstruct drafter continuity. Every
    /// successful turn publishes one sealed pair, or one explicit target-only
    /// downgrade when a deterministic tap precondition is unavailable. Once
    /// effectful sealing begins, any error publishes neither half.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn generate_continued_dflash_locked(
        &self,
        session_id: u64,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        timing: bool,
        total_start: std::time::Instant,
    ) -> Result<SessionGeneration, EngineError> {
        let dflash = self
            .dflash
            .as_ref()
            .ok_or_else(|| EngineError::Generation("DFlash state missing".to_owned()))?;
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation(
                "session DFlash requires a non-empty prompt".to_owned(),
            ));
        }
        let total = Self::prompt_len(prompt_tokens)?;

        if let Ok(mut values) = self.last_dflash_accepts.lock() {
            values.clear();
        }
        if let Ok(mut values) = self.last_dflash_draft_matches.lock() {
            values.clear();
        }
        if let Ok(mut values) = self.last_dflash_draft_counts.lock() {
            values.clear();
        }
        if let Ok(mut value) = self.last_dflash_timing.lock() {
            *value = None;
        }

        // Move a retained pair out of the session map before taking the model
        // lock. A target-only hit is not sufficient for dSpark because no
        // independent lookup may supply the missing historical tap frontier.
        let reusable = self.take_continuable(session_id, prompt_tokens).and_then(
            |(state, prior)| match state {
                RetainedState::Paired(pair) => Some((pair, prior)),
                RetainedState::TargetOnly(target) => {
                    drop(target);
                    None
                }
            },
        );

        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
        let mlx_gate = higgs_models::mlx_exec::acquire();
        let mut drafter = lock_or_recover(&dflash.drafter);

        let expected_taps = dflash.tap_layers.len();
        let resumed = reusable.and_then(|(pair, prior)| {
            let Some(prefix) = prompt_tokens.get(..prior) else {
                tracing::warn!(
                    session_id,
                    prior,
                    "Retained paired session boundary exceeds prompt; cold-prefilling"
                );
                return None;
            };
            match pair.resume(prefix, expected_taps) {
                Ok(pair) => Some((pair, prior)),
                Err(error) => {
                    tracing::warn!(
                        session_id,
                        error = %error,
                        prior,
                        "Retained paired session could not resume as one proven branch; cold-prefilling"
                    );
                    None
                }
            }
        });

        // Cache-state divergence must never fail the request outright: if the
        // resumed (continued) attempt errors, retry ONCE from a cold prefill
        // within the same request. The retained pair was already moved out of
        // the session map, so nothing poisoned survives in either outcome —
        // this only converts a residual 500 into a slower successful turn.
        let mut resumed = resumed;
        let mut retried_cold = false;
        let (generation, publication, cap_exempt) = loop {
            let (pair, prefill_from, continued) = if let Some((pair, prior)) = resumed.take() {
                (pair, prior, true)
            } else {
                let target = model
                    .make_cache_with_config(self.kv_cache_config)
                    .map_err(EngineError::Mlx)?;
                let pair = LivePair::cold(target, drafter.make_cache(), expected_taps)
                    .map_err(session_pair_error)?;
                (pair, 0, false)
            };
            if retried_cold {
                // The failed continued attempt may have recorded partial per-round
                // stats; the retry must report only its own rounds.
                if let Ok(mut values) = self.last_dflash_accepts.lock() {
                    values.clear();
                }
                if let Ok(mut values) = self.last_dflash_draft_matches.lock() {
                    values.clear();
                }
                if let Ok(mut values) = self.last_dflash_draft_counts.lock() {
                    values.clear();
                }
            }
            let attempt =
                (|| -> Result<(SessionGeneration, Option<SessionDsparkPublication>, bool), EngineError> {
                    let prefill_tokens = prompt_tokens.get(prefill_from..).ok_or_else(|| {
                        EngineError::Generation(
                            "session DFlash prefill boundary exceeds prompt".to_owned(),
                        )
                    })?;
                    if prefill_tokens.is_empty() {
                        return Err(EngineError::Generation(
                            "session DFlash continuation requires a non-empty suffix".to_owned(),
                        ));
                    }
                    let prefilled = u32::try_from(prefill_tokens.len()).unwrap_or(u32::MAX);

                    if continued {
                        self.cache_metrics
                            .continuations
                            .fetch_add(1, Ordering::Relaxed);
                        self.cache_metrics.prefill_saved_tokens.fetch_add(
                            u64::from(total.saturating_sub(prefilled)),
                            Ordering::Relaxed,
                        );
                    }

                    let prefill_start = std::time::Instant::now();
                    let (pair, prefill_logits) = pair
                        .prefill_known(prefill_tokens, |exact_suffix, target_cache, draft_cache| {
                            self.dflash_prefill_with_taps(
                                &mut model,
                                target_cache,
                                &mut drafter,
                                draft_cache,
                                exact_suffix,
                                &dflash.tap_layers,
                            )
                        })
                        .map_err(session_pair_error)?;
                    let first_token = sample(&prefill_logits.index((.., -1, ..)), params)
                        .map_err(EngineError::Mlx)?;
                    eval([&first_token]).map_err(EngineError::Mlx)?;
                    let prefill_elapsed = prefill_start.elapsed();

                    let first_id: u32 = first_token.item();
                    let mut decode = SessionDsparkDecodeState::begin(pair, first_id)?;
                    let max_tokens = usize::try_from(max_tokens).map_err(|_| {
                        EngineError::Generation("max_tokens overflow usize".to_owned())
                    })?;

                    let decode_start = std::time::Instant::now();
                    // This loop owns only session delivery/termination. Ticket creation,
                    // exact history, target+dSpark advancement, lease validation, and
                    // ledger commit all run through the same canonical transaction driver
                    // used by stateless decoding.
                    while decode.len() < max_tokens {
                        let pending_anchor = decode.pending_token().ok_or_else(|| {
                            EngineError::Generation(
                                "session DFlash ledger lost its pending token".to_owned(),
                            )
                        })?;
                        if self.eos_token_ids.contains(&pending_anchor) {
                            break;
                        }
                        let remaining = max_tokens.saturating_sub(decode.len());
                        if remaining == 0 {
                            break;
                        }
                        let (next_decode, committed) = decode.run_round(
                            |target_cache,
                             draft_cache,
                             tap_frontier,
                             anchor_ticket,
                             history_before_anchor| {
                                let round = self.canonical_dflash_round(
                                    &mut model,
                                    target_cache,
                                    &mut drafter,
                                    draft_cache,
                                    tap_frontier,
                                    anchor_ticket,
                                    history_before_anchor,
                                    remaining,
                                    remaining,
                                    params,
                                    &[],
                                    None,
                                    dflash,
                                )?;
                                round.into_lease_parts()
                            },
                        )?;
                        decode = next_decode;

                        if let Ok(mut values) = self.last_dflash_accepts.lock() {
                            values.push(u32::try_from(committed.successors.len()).unwrap_or(0));
                        }
                        if let Ok(mut values) = self.last_dflash_draft_matches.lock() {
                            values.push(u32::try_from(committed.draft_matches).unwrap_or(0));
                        }
                        if let Ok(mut values) = self.last_dflash_draft_counts.lock() {
                            values.push(u32::try_from(committed.draft_tokens.len()).unwrap_or(0));
                        }
                    }
                    let decode_elapsed = decode_start.elapsed();
                    let retain_start = std::time::Instant::now();

                    // Align the final visible token with both caches. EOS remains visible
                    // but is intentionally absent from the retained key; a non-EOS length
                    // tail receives one target-only forward whose taps are consumed while
                    // sealing the drafter snapshot.
                    let (mut pair, generated) = decode.finish(
                        &self.eos_token_ids,
                        |target_cache, _draft_cache, tap_frontier, token| {
                            let input_token = i32::try_from(token).map_err(|_| {
                                EngineError::Generation(
                                    "session DFlash retained token overflow i32".to_owned(),
                                )
                            })?;
                            let input = Array::from_slice(&[input_token], &[1, 1]);
                            let (_hidden, taps) = model
                                .forward_raw_with_taps(
                                    &input,
                                    None,
                                    target_cache,
                                    &dflash.tap_layers,
                                )
                                .map_err(EngineError::Mlx)?;
                            let next_target_boundary = tap_frontier
                                .target_boundary()
                                .checked_add(1)
                                .ok_or_else(|| {
                                    EngineError::Generation(
                                        "session target boundary overflow".to_owned(),
                                    )
                                })?;
                            tap_frontier.append(next_target_boundary, taps)
                        },
                    )?;
                    let retained_boundary = pair.token_len();

                    let cap = self.kv_cache_config.max_session_tokens;
                    let over_dense_cap = cap > 0 && retained_boundary > cap;
                    let mut cap_exempt = false;
                    if self.kv_cache_config.is_turboquant() || over_dense_cap {
                        match pair.quantize_target_for_retention(self.kv_cache_config, &mlx_gate) {
                            Ok(layers) if layers > 0 => {
                                cap_exempt = over_dense_cap;
                                tracing::debug!(
                                    session_id,
                                    compressed_layers = layers,
                                    over_dense_cap,
                                    "Compressed paired target KV for between-turn retention"
                                );
                            }
                            Ok(_) => {}
                            Err(error) => tracing::warn!(
                                session_id,
                                error = %error,
                                "Failed to TurboQuant-compress paired target KV; retaining dense"
                            ),
                        }
                    }

                    let publication = self.seal_session_dspark_publication(
                        session_id,
                        pair,
                        &mut drafter,
                        &mlx_gate,
                    );
                    let retain_elapsed = retain_start.elapsed();

                    let total_elapsed = total_start.elapsed();
                    if let Ok(mut value) = self.last_dflash_timing.lock() {
                        *value = Some(GenerationPhaseTiming {
                            prefill: prefill_elapsed,
                            decode: decode_elapsed,
                            request: total_elapsed,
                            decode_tokens: u32::try_from(generated.len().saturating_sub(1))
                                .unwrap_or(u32::MAX),
                        });
                    }
                    #[allow(clippy::print_stderr)]
                    if timing {
                        eprintln!(
                            "DIAG session-dspark-timing: continued={continued} prompt={total} prefilled={prefilled} generated={} prefill={prefill_elapsed:.2?} decode={decode_elapsed:.2?} retain_eval={retain_elapsed:.2?} total={total_elapsed:.2?}",
                            generated.len()
                        );
                    }
                    tracing::info!(
                        session_id,
                        continued,
                        prompt_tokens = total,
                        prefilled_tokens = prefilled,
                        prefill_saved = total.saturating_sub(prefilled),
                        "paired dSpark cache-resident turn"
                    );

                    Ok((
                        SessionGeneration {
                            text: self.decode_tokens(&generated)?,
                            completion_tokens: u32::try_from(generated.len()).unwrap_or(u32::MAX),
                            prompt_tokens: total,
                            prefilled_tokens: prefilled,
                            continued,
                        },
                        publication,
                        cap_exempt,
                    ))
                })();
            match attempt {
                Ok(parts) => break parts,
                Err(error) if continued && !retried_cold => {
                    retried_cold = true;
                    tracing::warn!(
                        session_id,
                        error = %error,
                        "continued dSpark turn failed; retrying this request from a cold prefill"
                    );
                }
                Err(error) => return Err(error),
            }
        };

        drop(drafter);
        drop(model);
        drop(mlx_gate);
        self.publish_session_dspark_publication(session_id, publication, cap_exempt);
        Ok(generation)
    }

    /// Greedy decode with a token-age KV-prune policy applied after every step.
    ///
    /// This is the prune-rate sweep driver: a deliberately plain sequential loop
    /// (no MTP, prompt-lookup, constraints, or logprobs) so the only variable is
    /// the prune policy. Reuses the production `run_prefill` / `decode_step`, so
    /// sampling and detokenization match a normal request. `rope` carries the
    /// model's RoPE params (`base = rope_theta`, `dims = head_dim`, `scale = 1.0`,
    /// `traditional = false` for Qwen3).
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::doc_markdown
    )]
    pub fn generate_with_prune(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        policy: &crate::prune::PrunePolicy,
        rope: higgs_models::cache::RopeShift,
    ) -> Result<PrunedGeneration, EngineError> {
        let mut prepared = self.prepare_generation(prompt_tokens, None, None)?;
        let prompt_len = usize::try_from(prepared.prompt_len)
            .map_err(|_| EngineError::Generation("prompt_len overflow".to_owned()))?;
        let (current_token, _, _) = self.run_prefill(
            prompt_tokens,
            self.gen_prompt_suffixes.for_request(self.enable_thinking),
            &mut prepared,
            params,
            None,
            None,
            false,
            prefix_cache_enabled(),
            None,
            None,
            None,
        )?;

        let first_id: u32 = current_token.item();
        let mut tokens: Vec<u32> = vec![first_id];
        let mut peak_resident = prepared.cache.resident_len();

        if self.eos_token_ids.contains(&first_id) || max_tokens <= 1 {
            return Ok(PrunedGeneration {
                text: self.decode_tokens(&tokens)?,
                completion_tokens: 1,
                peak_resident_kv: u32::try_from(peak_resident).unwrap_or(0),
                decode_seconds: 0.0,
                pruned_steps: 0,
            });
        }

        // Structural policy: a per-resident-token "protected" mask (sinks +
        // fact-bearing tokens) kept in lockstep with the cache. Empty for the
        // age-based policy so its decode perf is unchanged.
        let protect = policy.protect_facts;
        let sink_usize = usize::try_from(policy.sink.max(0)).unwrap_or(0);
        let resident0 = usize::try_from(prepared.cache.resident_len()).unwrap_or(0);
        let mut protected: Vec<bool> = if protect {
            let apt = &prepared.actual_prompt_tokens;
            (0..resident0)
                .map(|i| i < sink_usize || apt.get(i).is_some_and(|&id| self.token_is_fact(id)))
                .collect()
        } else {
            Vec::new()
        };

        let mut cur = current_token;
        let mut cur_id = first_id;
        let mut pruned_steps = 0_u32;
        let start = std::time::Instant::now();
        while u32::try_from(tokens.len()).unwrap_or(u32::MAX) < max_tokens {
            let (next, _) = Self::decode_step(
                &cur,
                &mut prepared.model,
                &mut prepared.cache,
                params,
                &tokens,
                None,
                None,
            )?;
            // The token just forwarded (`cur`) is now resident; keep the mask aligned.
            if protect {
                protected.push(self.token_is_fact(cur_id));
            }
            // Logical length as if nothing were ever pruned: prompt + tokens
            // forwarded so far (the token just forwarded is the last in `tokens`).
            let full_len = i32::try_from(prompt_len + tokens.len()).unwrap_or(i32::MAX);
            let pruned_now = if protect {
                crate::prune::apply_structural_prune(
                    &mut prepared.cache,
                    &mut protected,
                    full_len,
                    policy,
                    rope,
                )? > 0
            } else {
                crate::prune::apply_prune(&mut prepared.cache, full_len, policy, rope)?
            };
            if pruned_now {
                pruned_steps += 1;
            }
            peak_resident = peak_resident.max(prepared.cache.resident_len());

            let next_id: u32 = next.item();
            tokens.push(next_id);
            if self.eos_token_ids.contains(&next_id) {
                break;
            }
            cur = next;
            cur_id = next_id;
        }
        let decode_seconds = start.elapsed().as_secs_f32();

        Ok(PrunedGeneration {
            text: self.decode_tokens(&tokens)?,
            completion_tokens: u32::try_from(tokens.len()).unwrap_or(u32::MAX),
            peak_resident_kv: u32::try_from(peak_resident).unwrap_or(0),
            decode_seconds,
            pruned_steps,
        })
    }

    /// Render a single user message to prompt tokens.
    fn render_user(&self, content: String, enable_thinking: bool) -> Result<Vec<u32>, EngineError> {
        let msg = ChatMessage {
            role: "user".to_owned(),
            content,
            tool_calls: None,
        };
        self.prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, enable_thinking)
    }

    /// Long-horizon generation with **model-driven context self-maintenance**.
    ///
    /// Instead of pruning KV, the model curates its own working context: each
    /// segment generates up to `seg_max_tokens`; if it runs out without finishing
    /// (no EOS), the model is asked to write a concise progress summary, and the
    /// next segment continues from `[task + summary]` with a fresh, bounded cache.
    /// Resident KV stays bounded by task + segment budget regardless of how long
    /// the reasoning runs — the win we measured against KV-pruning.
    ///
    /// This is the generic, engine-driven version (the caller does not pre-chunk
    /// the task): it relies on the model summarizing both its state and its
    /// position well enough to resume — verified in `self_maintained_engine`.
    #[allow(clippy::option_if_let_else)] // the match reads clearer than map_or_else
    pub fn generate_self_maintained(
        &self,
        task: &str,
        params: &SamplingParams,
        rope: higgs_models::cache::RopeShift,
        cfg: &SelfMaintainCfg,
    ) -> Result<SelfMaintainedOutput, EngineError> {
        let disabled = crate::prune::PrunePolicy::disabled();
        let mut carried: Option<String> = None;
        let mut peak = 0_u32;
        let mut total = 0_u32;
        let mut summaries = Vec::new();

        for seg in 0..cfg.max_segments.max(1) {
            let prompt = match &carried {
                None => format!(
                    "{task}\n\nSolve this step by step. When you reach the final result, end with a line 'Answer: <answer>'."
                ),
                Some(progress) => format!(
                    "{task}\n\nYou have already started and made progress. Progress so far:\n{progress}\n\nContinue from this exact state — do not restart. When you reach the final result, end with a line 'Answer: <answer>'."
                ),
            };
            let toks = self.render_user(prompt, cfg.enable_thinking)?;
            let out =
                self.generate_with_prune(&toks, cfg.seg_max_tokens, params, &disabled, rope)?;
            peak = peak.max(out.peak_resident_kv);
            total = total.saturating_add(out.completion_tokens);

            // A segment that stopped before its budget hit EOS → the model finished.
            if out.completion_tokens < cfg.seg_max_tokens {
                return Ok(SelfMaintainedOutput {
                    text: out.text,
                    segments: seg + 1,
                    peak_resident_kv: peak,
                    total_tokens: total,
                    summaries,
                });
            }

            // Truncated → checkpoint: have the model summarize its own progress.
            let sum_prompt = format!(
                "{task}\n\nWork in progress (possibly incomplete):\n{}\n\nWrite a concise progress note capturing exactly what you have completed so far and the current state needed to continue: which steps are done, the key running values, and what remains. Do NOT give the final answer yet.",
                out.text
            );
            let sum_toks = self.render_user(sum_prompt, cfg.enable_thinking)?;
            let summary = self.generate_with_prune(
                &sum_toks,
                cfg.summary_max_tokens,
                params,
                &disabled,
                rope,
            )?;
            peak = peak.max(summary.peak_resident_kv);
            total = total.saturating_add(summary.completion_tokens);
            summaries.push(summary.text.clone());
            carried = Some(summary.text);
        }

        // Exhausted the segment budget without an explicit finish.
        Ok(SelfMaintainedOutput {
            text: carried.unwrap_or_default(),
            segments: cfg.max_segments.max(1),
            peak_resident_kv: peak,
            total_tokens: total,
            summaries,
        })
    }

    /// Whether a token decodes to text containing a digit — a cheap, schema-free
    /// proxy for "fact-bearing" (conclusion) tokens in arithmetic reasoning. The
    /// structural prune policy protects these from eviction.
    fn token_is_fact(&self, id: u32) -> bool {
        self.tokenizer
            .decode(std::slice::from_ref(&id), false)
            .is_ok_and(|s| s.bytes().any(|b| b.is_ascii_digit()))
    }

    /// Decode the token buffer and return the text, mapping tokenizer errors.
    ///
    /// Decodes WITHOUT skipping special tokens so content-bearing markup
    /// survives — notably models (e.g. `MiniCPM5`) that encode their tool-call
    /// structure (`<function>`, `<param>`, …) as special tokens, which the
    /// tool parser needs to see. Control tokens (EOS) are filtered out first so
    /// they never leak into visible text. Plain text contains no special
    /// tokens and decodes identically either way, so normal responses are
    /// unaffected.
    fn decode_tokens(&self, tokens: &[u32]) -> Result<String, EngineError> {
        let decode = |ids: &[u32]| {
            self.tokenizer
                .decode(ids, false)
                .map_err(|e| EngineError::Tokenization(e.to_string()))
        };
        // Fast path: no control token present, decode the slice as-is.
        if !tokens.iter().any(|id| self.decode_skip_ids.contains(id)) {
            return decode(tokens);
        }
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !self.decode_skip_ids.contains(id))
            .collect();
        decode(&filtered)
    }

    /// The model's hidden dimension (embedding output size).
    pub fn hidden_size(&self) -> i32 {
        let model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.hidden_size()
    }

    /// Compute embeddings for a sequence of token IDs.
    ///
    /// Runs a single forward pass through the model to get hidden states,
    /// mean-pools across the sequence dimension, and L2-normalizes.
    #[allow(clippy::significant_drop_tightening)]
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::Generation("Input is empty".to_owned()));
        }

        with_new_default_stream(Stream::new(), || {
            let input = Array::from(token_ids).index(NewAxis);
            let mut model = self
                .model
                .lock()
                .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;
            // Gate MLX eval for the embed forward (held until end of this block).
            let _mlx_gate = higgs_models::mlx_exec::acquire();
            let mut cache = model
                .make_cache_with_config(self.kv_cache_config)
                .map_err(EngineError::Mlx)?;

            // Forward pass to get hidden states [1, seq_len, hidden_size]
            let hidden = model
                .forward_hidden(&input, None, &mut cache)
                .map_err(EngineError::Mlx)?;

            // Mean-pool across seq_len (axis 1), producing [1, hidden_size]
            let pooled = hidden.mean_axes(&[1], false).map_err(EngineError::Mlx)?;

            // Cast to f32 before extracting values (model may use bfloat16)
            let pooled_f32 = pooled.as_dtype(Dtype::Float32).map_err(EngineError::Mlx)?;
            eval([&pooled_f32]).map_err(EngineError::Mlx)?;

            // L2-normalize on CPU
            let values = pooled_f32.as_slice::<f32>().to_vec();
            let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                Ok(values.iter().map(|x| x / norm).collect())
            } else {
                Ok(values)
            }
        })
    }

    /// Convert a token count to u32, with an overflow error.
    fn completion_len(tokens: &[u32]) -> Result<u32, EngineError> {
        tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Too many tokens generated".to_owned()))
    }

    fn hidden_row_from_sequence(hidden: &Array, row_index: usize) -> Result<Array, EngineError> {
        let row_i32 = i32::try_from(row_index)
            .map_err(|_| EngineError::Generation("hidden row index too large".to_owned()))?;
        Ok(hidden.index((.., row_i32..row_i32 + 1, ..)))
    }

    // =========================================================================
    // Session Management (Batched Generation)
    // =========================================================================

    /// Create a new session for batched generation.
    ///
    /// Returns the session ID.
    pub fn create_session(
        &self,
        _prompt_tokens: &[u32],
        _max_tokens: usize,
    ) -> Result<u64, EngineError> {
        Err(EngineError::Generation(
            "session generation is not implemented for SimpleEngine yet".to_owned(),
        ))
    }

    /// Get session state.
    pub fn get_session(&self, session_id: u64) -> Option<Session> {
        let sessions = lock_or_recover(&self.sessions);
        sessions.get(&session_id).cloned()
    }

    /// Remove a session and free its resources.
    pub fn remove_session(&self, session_id: u64) -> Result<(), EngineError> {
        let mut scheduler = lock_or_recover(&self.scheduler);
        let mut sessions = lock_or_recover(&self.sessions);
        sessions.remove(&session_id);
        scheduler.remove(session_id);
        if let Some(paged_cache_mutex) = &self.paged_cache {
            let mut paged_cache = lock_or_recover(paged_cache_mutex);
            paged_cache
                .remove_session(session_id)
                .map_err(|e| EngineError::Generation(format!("Failed to remove session: {e}")))?;
        }

        Ok(())
    }

    /// Check if session is finished.
    pub fn is_session_finished(&self, session_id: u64) -> bool {
        let sessions = lock_or_recover(&self.sessions);
        sessions.get(&session_id).is_none_or(|s| s.finished)
    }

    /// Step one token for all active sessions (batched generation).
    ///
    /// Returns outputs for each session that produced a token.
    ///
    /// Note: Current implementation processes sessions sequentially.
    /// True batched generation (parallel decode across sessions) is TODO.
    pub fn step(
        &self,
        _params: &SamplingParams,
    ) -> Result<Vec<(u64, GenerationOutput)>, EngineError> {
        Err(EngineError::Generation(
            "session stepping is not implemented for SimpleEngine yet".to_owned(),
        ))
    }

    /// Generate a complete response for a session (batched mode).
    ///
    /// This is a helper for batched generation that generates all tokens
    /// for a session in one call, using the paged cache.
    pub fn generate_session(
        &self,
        session_id: u64,
        _params: &SamplingParams,
        _stop_sequences: &[String],
        _logprobs: bool,
        _top_logprobs: Option<u32>,
    ) -> Result<GenerationOutput, EngineError> {
        Err(EngineError::Generation(format!(
            "session generation is not implemented for SimpleEngine yet (session {session_id})"
        )))
    }

    /// Generate a complete response from a token prompt.
    ///
    /// For multimodal requests, pass `pixel_values` with preprocessed image
    /// data and ensure `prompt_tokens` contains `IMAGE_TOKEN_INDEX` at image
    /// positions.
    #[allow(clippy::significant_drop_tightening, clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            self.enable_thinking,
            constraint,
            pixel_values,
            checkpoint_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<GenerationOutput, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                finish_reason: "length".to_owned(),
                prompt_tokens: Self::prompt_len(prompt_tokens)?,
                completion_tokens: 0,
                token_logprobs: None,
            });
        }

        // Set a task-local default stream so every MLX operation reuses it
        // instead of creating a new Stream (5 FFI calls) per operation.
        with_new_default_stream(Stream::new(), || {
            self.generate_inner(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                pixel_values,
                checkpoint_id,
            )
        })
    }

    #[allow(
        clippy::significant_drop_tightening,
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::float_cmp
    )]
    fn generate_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        mut constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<GenerationOutput, EngineError> {
        let thinking_budget = params.thinking_budget.unwrap_or(DEFAULT_THINKING_BUDGET);
        // DFlash speculative decoding: use the draft-verify loop when a drafter
        // is loaded, the request allows it (`speculation` = auto/dflash), no
        // constraints active, and no multimodal input.
        let sampled_dspark = self.sampled_dspark_requires_ar(params);
        if sampled_dspark {
            tracing::debug!(
                "sampled dSpark request uses ordinary AR to preserve the shared RNG transition"
            );
        }
        let speculation_route = resolve_speculation_route(
            params.speculation,
            self.dflash.is_some()
                && !sampled_dspark
                && constraint.is_none()
                && pixel_values.is_none()
                && !logprobs,
            stateless_mtp_family_eligible(params, constraint.is_some(), logprobs),
        );
        if speculation_route == SpeculationRoute::DFlash {
            return self.generate_dflash_inner(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                enable_thinking,
                thinking_budget,
            );
        }

        let ar_request_t0 = std::time::Instant::now();
        if let Ok(mut timing) = self.last_ar_timing.lock() {
            *timing = None;
        }
        let logprob_top_n = logprobs.then(|| top_logprobs.unwrap_or(0));

        let mut prepared = self.prepare_generation(prompt_tokens, pixel_values, checkpoint_id)?;
        let prompt_len = prepared.prompt_len;
        #[allow(clippy::float_cmp)]
        let capture_mtp_prefill = mtp_prefill_priming_enabled()
            && self.tuning.enable_mtp()
            && prepared.model.has_mtp()
            && prepared.pixel_values.is_none()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0;

        let (current_token, first_logprob_data, prefill_hidden) = self.run_prefill(
            prompt_tokens,
            self.gen_prompt_suffixes.for_request(enable_thinking),
            &mut prepared,
            params,
            logprob_top_n,
            constraint.as_ref(),
            capture_mtp_prefill,
            prefix_cache_enabled(),
            checkpoint_id,
            None,
            None,
        )?;

        // Capture T1 (already eval'd inside run_prefill).
        let first_token_id: u32 = current_token.item();
        let ar_prefill_elapsed = ar_request_t0.elapsed();
        // Advance the constraint past the first sampled token before decode.
        if let Some(ref mut cg) = constraint {
            cg.advance(first_token_id);
        }
        let mut tokens: Vec<u32> = vec![first_token_id];
        let mut all_logprobs: Option<Vec<higgs_models::TokenLogprobInfo>> = logprobs.then(Vec::new);
        if let (Some(all_lp), Some(lp_data)) = (&mut all_logprobs, &first_logprob_data) {
            all_lp.push(lp_data.materialize(first_token_id));
        }
        let has_stop_sequences = !stop_sequences.is_empty();

        // Handle T1 termination before entering the pipeline.
        if self.eos_token_ids.contains(&first_token_id) {
            self.record_ar_generation_timing(
                ar_prefill_elapsed,
                std::time::Duration::ZERO,
                ar_request_t0.elapsed(),
                1,
                "stop",
            );
            return Ok(GenerationOutput {
                text: self.decode_tokens(&tokens)?,
                finish_reason: "stop".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprobs: all_logprobs,
            });
        }
        if has_stop_sequences {
            let text = self.decode_tokens(&tokens)?;
            if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                self.record_ar_generation_timing(
                    ar_prefill_elapsed,
                    std::time::Duration::ZERO,
                    ar_request_t0.elapsed(),
                    1,
                    "stop",
                );
                return Ok(GenerationOutput {
                    text: truncated,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: 1,
                    token_logprobs: all_logprobs,
                });
            }
        }
        if max_tokens <= 1 {
            self.record_ar_generation_timing(
                ar_prefill_elapsed,
                std::time::Duration::ZERO,
                ar_request_t0.elapsed(),
                1,
                "length",
            );
            return Ok(GenerationOutput {
                text: self.decode_tokens(&tokens)?,
                finish_reason: "length".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprobs: all_logprobs,
            });
        }

        // Architecture-neutral speculative decode: prompt-lookup drafting plus
        // batched verifier logits. Explicitly opt-in while benchmark data is
        // collected because the normal path has a pipelined single-token loop.
        #[allow(clippy::float_cmp)]
        if prompt_lookup_enabled()
            && speculation_route == SpeculationRoute::MtpFamily
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0
        {
            return self.prompt_lookup_generate(
                &mut prepared.model,
                &mut prepared.cache,
                first_token_id,
                max_tokens,
                prompt_len,
                &mut tokens,
                stop_sequences,
                enable_thinking,
                thinking_budget,
            );
        }

        // MTP speculative decode: enabled by the resolved MLX runtime tuning.
        // Only for greedy (temperature == 0), no constraints, no logprobs.
        #[allow(clippy::float_cmp)]
        if self.tuning.enable_mtp()
            && speculation_route == SpeculationRoute::MtpFamily
            && prepared.model.has_mtp()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0
        {
            let actual_prompt_tokens = prepared.actual_prompt_tokens.clone();
            return self.mtp_generate(
                &mut prepared.model,
                &mut prepared.cache,
                &actual_prompt_tokens,
                prefill_hidden.as_ref(),
                first_token_id,
                max_tokens,
                prompt_len,
                &mut tokens,
                stop_sequences,
                enable_thinking,
                thinking_budget,
            );
        }

        // Pipelined decode: build step N+2's graph while GPU computes step N+1.
        // When constrained generation is active, pipelining would apply the FSM mask
        // one step behind (since we need the sampled token value to advance the FSM
        // before constraining the next step). Fall back to sequential decode instead.
        let ar_decode_t0 = std::time::Instant::now();
        let (mut next_token, mut next_logprob_data) = Self::decode_step(
            &current_token,
            &mut prepared.model,
            &mut prepared.cache,
            params,
            &tokens,
            logprob_top_n,
            constraint.as_ref(),
        )?;
        {
            let mut eval_targets: Vec<&Array> = vec![&next_token];
            if let Some(ref lp) = next_logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            if constraint.is_some() {
                eval(eval_targets).map_err(EngineError::Mlx)?;
            } else {
                async_eval(eval_targets).map_err(EngineError::Mlx)?;
            }
        }

        let mut total_forward_ns: u128 = 0;
        let mut total_eval_ns: u128 = 0;
        let mut total_item_ns: u128 = 0;
        let mut total_other_ns: u128 = 0;
        let mut step_count: u32 = 0;

        // Thinking budget: force </think> after N tokens if model hasn't closed it.
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        // Seed thinking state from the first token (already emitted above).
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let t0 = std::time::Instant::now();
            // The currently queued token is guaranteed to terminate a
            // length-bounded request when adding it reaches `max_tokens`.
            // Do not launch the pipelined successor in that case: it can never
            // be observed, and an async final forward would otherwise spill
            // GPU work past the request's wall-clock boundary into the next
            // benchmark sample.
            let will_finish_by_length =
                Self::completion_len(&tokens)?.saturating_add(1) >= max_tokens;

            // When constrained, extract the sampled token and advance the FSM
            // before building the next step, so the mask is always applied at the
            // correct FSM state.
            let constrained_token_id: Option<u32> = constraint.is_some().then(|| {
                let id: u32 = next_token.item();
                if let Some(ref mut cg) = constraint {
                    cg.advance(id);
                }
                id
            });

            let following = if will_finish_by_length {
                None
            } else {
                Some(Self::decode_step(
                    &next_token,
                    &mut prepared.model,
                    &mut prepared.cache,
                    params,
                    &tokens,
                    logprob_top_n,
                    constraint.as_ref(),
                )?)
            };
            let t1 = std::time::Instant::now();
            if let Some((following_token, following_logprob_data)) = &following {
                let mut eval_targets: Vec<&Array> = vec![following_token];
                if let Some(lp) = following_logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
                if constraint.is_some() {
                    eval(eval_targets).map_err(EngineError::Mlx)?;
                } else {
                    async_eval(eval_targets).map_err(EngineError::Mlx)?;
                }
            }
            let t2 = std::time::Instant::now();

            // In the unconstrained pipeline, extract the token here (after building following).
            let mut token_id: u32 = constrained_token_id.unwrap_or_else(|| next_token.item());

            // Thinking budget: force </think> after N tokens if model hasn't closed it.
            // NOTE: when the budget fires, token_id is overwritten but the KV cache
            // already reflects the originally-sampled token.  The next forward pass
            // feeds close_id as input while the cache holds a different token at this
            // position — a one-entry discontinuity.  Re-running forward to fix the
            // cache is expensive for negligible quality impact after 256+ tokens.
            if let Some(close_id) = think_close_token {
                if !seen_think_close {
                    if token_id == close_id {
                        seen_think_close = true;
                    } else {
                        thinking_tokens += 1;
                        if thinking_tokens >= thinking_budget {
                            token_id = close_id;
                            seen_think_close = true;
                            tracing::info!(
                                budget = thinking_budget,
                                "Thinking budget reached, forcing </think>"
                            );
                        }
                    }
                }
            }

            // Materialize logprobs for the token we just extracted
            if let (Some(all_lp), Some(lp_data)) = (&mut all_logprobs, &next_logprob_data) {
                all_lp.push(lp_data.materialize(token_id));
            }

            let t3 = std::time::Instant::now();

            tokens.push(token_id);
            let completion_len = Self::completion_len(&tokens)?;
            let t4 = std::time::Instant::now();

            total_forward_ns += (t1 - t0).as_nanos();
            total_eval_ns += (t2 - t1).as_nanos();
            total_item_ns += (t3 - t2).as_nanos();
            total_other_ns += (t4 - t3).as_nanos();
            step_count += 1;
            let record_ar_finish = |finish_reason: &str| {
                let decode_elapsed = ar_decode_t0.elapsed();
                let request_elapsed = ar_request_t0.elapsed();
                self.record_ar_generation_timing(
                    ar_prefill_elapsed,
                    decode_elapsed,
                    request_elapsed,
                    completion_len,
                    finish_reason,
                );
            };

            // Check if constraint is in final state
            if constraint
                .as_ref()
                .is_some_and(crate::constrained::ConstrainedGenerator::is_finished)
            {
                record_ar_finish("stop");
                Self::log_decode_timing(
                    step_count,
                    total_forward_ns,
                    total_eval_ns,
                    total_item_ns,
                    total_other_ns,
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                });
            }

            if self.eos_token_ids.contains(&token_id) {
                record_ar_finish("stop");
                Self::log_decode_timing(
                    step_count,
                    total_forward_ns,
                    total_eval_ns,
                    total_item_ns,
                    total_other_ns,
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                });
            }

            if has_stop_sequences {
                let text = self.decode_tokens(&tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    record_ar_finish("stop");
                    Self::log_decode_timing(
                        step_count,
                        total_forward_ns,
                        total_eval_ns,
                        total_item_ns,
                        total_other_ns,
                    );
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                        token_logprobs: all_logprobs,
                    });
                }
            }

            if completion_len >= max_tokens {
                record_ar_finish("length");
                Self::log_decode_timing(
                    step_count,
                    total_forward_ns,
                    total_eval_ns,
                    total_item_ns,
                    total_other_ns,
                );
                return Ok(GenerationOutput {
                    text: self.decode_tokens(&tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: all_logprobs,
                });
            }

            let Some((following_token, following_logprob_data)) = following else {
                return Err(EngineError::Generation(
                    "length-final AR token did not terminate generation".to_owned(),
                ));
            };
            // If thinking budget was just reached, override the pipelined token
            // so the next decode step gets </think> as input.
            if seen_think_close && thinking_tokens == thinking_budget {
                if let Some(close_id) = think_close_token {
                    next_token = Array::from_slice(&[close_id], &[1]);
                }
                thinking_tokens += 1; // prevent re-triggering
            } else {
                next_token = following_token;
            }
            next_logprob_data = following_logprob_data;
        }
    }

    /// Append per-layer tap hiddens (`[1, T, H]` each) to the drafter's
    /// unconsumed backlog along the sequence axis. Used by the MTP floor to
    /// keep the drafter's context contiguous across floored stretches; the
    /// next spec round's drafter forward ingests the whole backlog at once.
    fn append_taps(
        current: &mut Vec<Array>,
        new: &[Array],
    ) -> Result<(), mlx_rs::error::Exception> {
        if current.is_empty() {
            *current = new.to_vec();
            return Ok(());
        }
        if current.len() != new.len() {
            return Err(mlx_rs::error::Exception::custom(format!(
                "tap layer count mismatch: backlog {} vs step {}",
                current.len(),
                new.len()
            )));
        }
        for (cur, add) in current.iter_mut().zip(new) {
            *cur = mlx_rs::ops::concatenate_axis(&[&*cur, add], 1)?;
        }
        Ok(())
    }

    /// Prefill the target and dSpark context with bounded tap residency.
    ///
    /// Completed target chunks are projected into the drafter's per-layer
    /// context cache immediately, then materialized and dropped. Only the last
    /// chunk's taps survive for the first draft query. The target vocabulary
    /// head likewise runs only on the final hidden row, matching ordinary
    /// prefill instead of allocating `[prompt, vocab]` logits.
    #[allow(clippy::too_many_arguments)]
    fn dflash_prefill_raw_with_taps(
        &self,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        drafter: &mut DFlashDrafter,
        draft_cache: &mut DFlashCache,
        prompt_tokens: &[u32],
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation(
                "DFlash prefill requires at least one token".to_owned(),
            ));
        }
        let prompt = Array::from(prompt_tokens).index(NewAxis);
        let seq_len =
            prompt.shape().get(1).copied().ok_or_else(|| {
                EngineError::Generation("DFlash prompt has no token axis".to_owned())
            })?;
        let chunk_size = self.tuning.chunked_prefill_chunk_size();
        let chunked = seq_len > self.tuning.chunked_prefill_threshold()
            && chunk_size > 0
            && chunk_size < seq_len;
        let mut offset = 0_i32;

        while chunked && offset + chunk_size < seq_len {
            let chunk = prompt.index((.., offset..offset + chunk_size));
            let (hidden, taps) = model
                .forward_raw_with_taps(&chunk, None, cache, tap_layers)
                .map_err(EngineError::Mlx)?;
            if taps.len() != tap_layers.len()
                || taps
                    .iter()
                    .any(|tap| tap.shape().get(1).copied() != Some(chunk_size))
            {
                return Err(EngineError::Generation(format!(
                    "DFlash prefill chunk returned {} taps for {} configured layers",
                    taps.len(),
                    tap_layers.len()
                )));
            }
            cache.eval().map_err(EngineError::Mlx)?;
            let mut targets = Vec::with_capacity(taps.len() + 1);
            targets.push(&hidden);
            targets.extend(taps.iter());
            eval(targets).map_err(EngineError::Mlx)?;
            drafter
                .prime_taps(&taps, draft_cache)
                .map_err(EngineError::Mlx)?;
            offset += chunk_size;
            higgs_models::progress::report_prefill_progress(offset, seq_len);
        }

        let final_chunk = prompt.index((.., offset..));
        let final_rows = seq_len - offset;
        let (hidden, taps) = model
            .forward_raw_with_taps(&final_chunk, None, cache, tap_layers)
            .map_err(EngineError::Mlx)?;
        if taps.len() != tap_layers.len()
            || taps
                .iter()
                .any(|tap| tap.shape().get(1).copied() != Some(final_rows))
        {
            return Err(EngineError::Generation(format!(
                "DFlash final prefill returned {} taps for {} configured layers",
                taps.len(),
                tap_layers.len()
            )));
        }
        cache.eval().map_err(EngineError::Mlx)?;
        higgs_models::progress::report_prefill_progress(seq_len, seq_len);
        Ok((hidden, taps))
    }

    #[allow(clippy::too_many_arguments)]
    fn dflash_advance_with_taps(
        &self,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        drafter: &mut DFlashDrafter,
        draft_cache: &mut DFlashCache,
        prompt_tokens: &[u32],
        tap_layers: &[usize],
    ) -> Result<Vec<Array>, EngineError> {
        let (hidden, taps) = self.dflash_prefill_raw_with_taps(
            model,
            cache,
            drafter,
            draft_cache,
            prompt_tokens,
            tap_layers,
        )?;
        let mut targets = Vec::with_capacity(taps.len() + 1);
        targets.push(&hidden);
        targets.extend(taps.iter());
        eval(targets).map_err(EngineError::Mlx)?;
        Ok(taps)
    }

    #[allow(clippy::too_many_arguments)]
    fn dflash_prefill_with_taps(
        &self,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        drafter: &mut DFlashDrafter,
        draft_cache: &mut DFlashCache,
        prompt_tokens: &[u32],
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), EngineError> {
        let (hidden, taps) = self.dflash_prefill_raw_with_taps(
            model,
            cache,
            drafter,
            draft_cache,
            prompt_tokens,
            tap_layers,
        )?;
        let logits = model
            .project_raw_hidden_last(&hidden)
            .map_err(EngineError::Mlx)?;
        let mut targets = Vec::with_capacity(taps.len() + 2);
        targets.push(&hidden);
        targets.push(&logits);
        targets.extend(taps.iter());
        eval(targets).map_err(EngineError::Mlx)?;
        Ok((logits, taps))
    }

    fn dflash_cold_prefill(
        &self,
        model: &mut AnyModel,
        drafter: &mut DFlashDrafter,
        prompt_tokens: &[u32],
        dflash: &DFlashState,
    ) -> Result<DflashPrefillOutcome, EngineError> {
        let cache = model
            .make_cache_with_config(self.kv_cache_config)
            .map_err(EngineError::Mlx)?;
        let draft_cache = drafter.make_cache();
        if dflash.is_dspark {
            let pair = LivePair::cold(cache, draft_cache, dflash.tap_layers.len())
                .map_err(radix_pair_error)?;
            let (pair, logits) = pair
                .prefill_known(prompt_tokens, |exact_suffix, target_cache, draft_cache| {
                    self.dflash_prefill_with_taps(
                        model,
                        target_cache,
                        drafter,
                        draft_cache,
                        exact_suffix,
                        &dflash.tap_layers,
                    )
                })
                .map_err(radix_pair_error)?;
            let parts = pair.into_stateless_parts().map_err(radix_pair_error)?;
            let expected = parts.frontier.target_boundary();
            return DflashPrefillOutcome::validated(
                parts.target,
                parts.dflash,
                logits,
                parts.frontier.taps,
                0,
                expected,
                dflash.tap_layers.len(),
            );
        }

        let mut cache = cache;
        let mut draft_cache = draft_cache;
        let expected = i32::try_from(prompt_tokens.len()).map_err(|_| {
            EngineError::Generation("DFlash prompt boundary overflow i32".to_owned())
        })?;
        let (logits, taps) = self.dflash_prefill_with_taps(
            model,
            &mut cache,
            drafter,
            &mut draft_cache,
            prompt_tokens,
            &dflash.tap_layers,
        )?;
        DflashPrefillOutcome::validated(
            cache,
            draft_cache,
            logits,
            taps,
            0,
            expected,
            dflash.tap_layers.len(),
        )
    }

    /// Materialize or build one exact paired conversation-body prefix, then
    /// prefill the request-specific generation tail into its live branch.
    ///
    /// `paired_plan` was selected while the prefix mutex was held, but owns all
    /// resources needed here after that mutex is released. Pair preparation,
    /// target cloning, dSpark sealing and snapshot forking happen under the MLX
    /// gate; the prefix mutex is reacquired only for CPU-only publication.
    #[allow(clippy::too_many_arguments)]
    fn dflash_prepare_prefill(
        &self,
        model: &mut AnyModel,
        drafter: &mut DFlashDrafter,
        mlx_gate: &higgs_models::mlx_exec::MlxExecToken,
        prompt_tokens: &[u32],
        partition: Option<DflashPromptPartition<'_>>,
        paired_plan: Option<DflashPairedPrefixPlan>,
        deferred_touch: &mut DeferredPairedTouch<'_>,
        dflash: &DFlashState,
    ) -> Result<DflashPrefillOutcome, EngineError> {
        let (Some(partition), Some(paired_plan)) = (partition, paired_plan) else {
            return self.dflash_cold_prefill(model, drafter, prompt_tokens, dflash);
        };

        let DflashPairedPrefixPlan {
            lookup,
            publication_ticket,
        } = paired_plan;
        let expected_taps = dflash.tap_layers.len();
        let reusable = lookup.and_then(|plan| match plan.materialize(expected_taps) {
            Ok(matched) => {
                let (pair, touch) = matched.into_pair_and_touch();
                let prefix_len = pair.token_len();
                deferred_touch.arm(touch);
                Some((pair, prefix_len))
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "Failed to materialize paired dSpark radix state; cold-prefilling"
                );
                None
            }
        });
        let (mut pair, reused_prefix_len) = if let Some((pair, prefix_len)) = reusable {
            (pair, prefix_len)
        } else {
            let target = model
                .make_cache_with_config(self.kv_cache_config)
                .map_err(EngineError::Mlx)?;
            let pair = LivePair::cold(target, drafter.make_cache(), expected_taps)
                .map_err(radix_pair_error)?;
            (pair, 0)
        };

        let publish_boundary =
            publication_ticket.store_boundary(partition.body_end, pair.target_is_hybrid());
        if publish_boundary == 0 || reused_prefix_len > publish_boundary {
            drop(pair);
            return self.dflash_cold_prefill(model, drafter, prompt_tokens, dflash);
        }

        if reused_prefix_len < publish_boundary {
            let body_delta = partition
                .prompt
                .get(reused_prefix_len..publish_boundary)
                .ok_or_else(|| {
                    EngineError::Generation("paired dSpark body boundary exceeds prompt".to_owned())
                })?;
            pair = pair
                .prefill_known(body_delta, |exact_suffix, target_cache, draft_cache| {
                    let body_taps = self.dflash_advance_with_taps(
                        model,
                        target_cache,
                        drafter,
                        draft_cache,
                        exact_suffix,
                        &dflash.tap_layers,
                    )?;
                    Ok::<_, EngineError>(((), body_taps))
                })
                .map_err(radix_pair_error)?
                .0;

            let checkpoint = pair.checkpoint_for_radix(drafter, mlx_gate, |checkpoint| {
                DiskPrefixCache::prepare_memory_paired_prefix(publication_ticket, checkpoint)
            });
            let (continued, prepared) = match checkpoint {
                Ok(result) => result,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        expected = publish_boundary,
                        "Failed to checkpoint a paired radix dSpark boundary; cold-prefilling"
                    );
                    return self.dflash_cold_prefill(model, drafter, prompt_tokens, dflash);
                }
            };
            pair = continued;
            match prepared {
                Ok(prepared) => {
                    let publish =
                        lock_or_recover(&self.prefix_cache).commit_memory_paired_prefix(prepared);
                    if let Err(error) = publish {
                        tracing::warn!(
                            error = %error,
                            "Failed to publish prepared paired dSpark radix state"
                        );
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "Failed to freeze paired dSpark radix state; continuing with the validated live pair"
                    );
                }
            }
        }

        let tail = partition
            .from_boundary(publish_boundary)
            .filter(|tokens| !tokens.is_empty())
            .ok_or_else(|| {
                EngineError::Generation(
                    "paired dSpark full hit has no cached logits or generation suffix".to_owned(),
                )
            })?;
        let (pair, logits) = pair
            .prefill_known(tail, |exact_suffix, target_cache, draft_cache| {
                self.dflash_prefill_with_taps(
                    model,
                    target_cache,
                    drafter,
                    draft_cache,
                    exact_suffix,
                    &dflash.tap_layers,
                )
            })
            .map_err(radix_pair_error)?;
        let parts = pair.into_stateless_parts().map_err(radix_pair_error)?;
        let expected = parts.frontier.target_boundary();
        DflashPrefillOutcome::validated(
            parts.target,
            parts.dflash,
            logits,
            parts.frontier.taps,
            reused_prefix_len,
            expected,
            expected_taps,
        )
    }

    /// Return the shortest candidate prefix that completes a textual stop.
    ///
    /// `DFlash` commits multiple tokens per round, so stop detection belongs in
    /// the verification transaction rather than only in the streaming sink.
    /// Capping the prefix before cache commit keeps output length, taps, and
    /// target state at the same token boundary.
    fn dflash_stop_prefix_len(
        &self,
        existing: &[u32],
        candidates: &[u32],
        stop_sequences: &[String],
    ) -> Result<Option<usize>, EngineError> {
        dflash_new_stop_prefix_len(existing, candidates, stop_sequences, |tokens| {
            self.decode_tokens(tokens)
        })
    }

    /// One target-authoritative S=1 DFlash/dSpark round shared by stateless and
    /// retained-session decoding.
    ///
    /// The move-only anchor ticket identifies the visible token absent from
    /// target KV. This method forwards it, then forwards each matching draft one
    /// at a time. The returned final successor is therefore the sole
    /// unforwarded token. The old tap frontier is consumed into the drafter and
    /// replaced with a new boundary-tagged frontier for exactly the target rows
    /// advanced here.
    #[allow(clippy::too_many_arguments)]
    fn canonical_dflash_round(
        &self,
        model: &mut AnyModel,
        target_cache: &mut AnyCache,
        drafter: &mut DFlashDrafter,
        draft_cache: &mut DFlashCache,
        tap_frontier: DflashTapFrontier,
        anchor_ticket: SpeculativeTicket,
        history_before_anchor: &[u32],
        hard_output_limit: usize,
        canonical_step_limit: usize,
        params: &SamplingParams,
        stop_sequences: &[String],
        terminal_token: Option<u32>,
        dflash: &DFlashState,
    ) -> Result<CanonicalDflashRound, EngineError> {
        if hard_output_limit == 0 || canonical_step_limit == 0 {
            return Err(EngineError::Generation(
                "canonical DFlash round requires output capacity".to_owned(),
            ));
        }
        tap_frontier.validate_live_draft(draft_cache)?;
        let expected_target_boundary = tap_frontier.target_boundary();
        target_cache
            .validate_absolute_boundary(expected_target_boundary)
            .map_err(EngineError::Mlx)?;

        let pending_anchor = anchor_ticket.token();
        if self.eos_token_ids.contains(&pending_anchor) {
            return Err(EngineError::Generation(
                "canonical DFlash round cannot forward a pending EOS token".to_owned(),
            ));
        }
        let anchor = i32::try_from(pending_anchor)
            .map_err(|_| EngineError::Generation("DFlash anchor overflow i32".to_owned()))?;
        let block_size = dflash.block_size;
        let block_size_us = usize::try_from(block_size)
            .map_err(|_| EngineError::Generation("block_size overflow for usize".to_owned()))?;
        let mut block_tokens = vec![dflash.mask_token_id; block_size_us];
        if let Some(slot) = block_tokens.first_mut() {
            *slot = anchor;
        }
        let block_ids = Array::from_slice(&block_tokens, &[1, block_size]);
        let noise_embedding = model
            .embed_token_ids(&block_ids)
            .map_err(EngineError::Mlx)?;
        let draft_transaction = drafter
            .stage_forward(&noise_embedding, &tap_frontier.taps, draft_cache)
            .map_err(EngineError::Mlx)?;
        let staged_position = draft_transaction.position().map_err(EngineError::Mlx)?;
        if staged_position != expected_target_boundary {
            return Err(EngineError::Generation(format!(
                "dSpark context transaction diverged: drafter={staged_position} target={expected_target_boundary}"
            )));
        }

        let round_draft_cap =
            dflash_tail_draft_cap(dflash.draft_cap, hard_output_limit, dflash.is_dspark);
        let proposal = dflash_propose_tokens(
            model,
            drafter,
            draft_transaction.hidden(),
            anchor,
            round_draft_cap,
            dflash.dspark_target_head,
        )?;
        let draft_tokens = proposal.host_tokens()?;
        let draft_inputs = draft_tokens
            .iter()
            .map(|&token| i32::try_from(token))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                EngineError::Generation("draft token overflow for sequential verify".to_owned())
            })?;

        let step_limit = hard_output_limit.min(canonical_step_limit);
        let mut target_tokens = Vec::with_capacity(step_limit);
        let mut committed_taps: Vec<Array> = Vec::new();
        let mut chain_history =
            Vec::with_capacity(history_before_anchor.len().saturating_add(step_limit + 1));
        chain_history.extend_from_slice(history_before_anchor);
        chain_history.push(pending_anchor);
        let stop_context = chain_history.clone();

        for (position, input_token) in std::iter::once(anchor).chain(draft_inputs).enumerate() {
            if target_tokens.len() >= step_limit {
                break;
            }
            let single = Array::from_slice(&[input_token], &[1, 1]);
            let (step_logits, step_taps) = model
                .forward_with_taps(&single, None, target_cache, &dflash.tap_layers)
                .map_err(EngineError::Mlx)?;
            let target =
                *crate::mtp::verify_targets(&step_logits, Some(params), Some(&chain_history))?
                    .first()
                    .ok_or_else(|| {
                        EngineError::Generation(
                            "canonical DFlash verifier produced no target".to_owned(),
                        )
                    })?;
            Self::append_taps(&mut committed_taps, &step_taps).map_err(EngineError::Mlx)?;
            target_tokens.push(target);
            chain_history.push(target);

            let textual_stop = self
                .dflash_stop_prefix_len(&stop_context, &target_tokens, stop_sequences)?
                .is_some();
            if dflash_canonical_target_is_terminal(
                position,
                target,
                &draft_tokens,
                &self.eos_token_ids,
                textual_stop,
                terminal_token,
            ) {
                break;
            }
        }

        let draft_matches = draft_tokens
            .iter()
            .zip(&target_tokens)
            .take_while(|(draft, target)| *draft == *target)
            .count();
        let verify =
            DFlashVerifyRound::committed(target_tokens, committed_taps, dflash.tap_layers.len());
        let advanced_rows = verify.validate_output_boundary()?;
        let expected_after = expected_target_boundary
            .checked_add(advanced_rows)
            .ok_or_else(|| EngineError::Generation("target boundary overflow".to_owned()))?;
        target_cache
            .validate_absolute_boundary(expected_after)
            .map_err(EngineError::Mlx)?;
        let (successors, round_taps) = verify.commit_target_state(model, target_cache)?;
        let next_frontier = DflashTapFrontier::new(
            expected_target_boundary,
            expected_after,
            round_taps,
            dflash.tap_layers.len(),
        )?;
        draft_transaction
            .commit(draft_cache)
            .map_err(EngineError::Mlx)?;
        if draft_cache.position() != expected_target_boundary {
            return Err(EngineError::Generation(format!(
                "canonical DFlash committed drafter boundary {}, expected {expected_target_boundary}",
                draft_cache.position()
            )));
        }

        Ok(CanonicalDflashRound {
            anchor_ticket,
            successors,
            tap_frontier: next_frontier,
            draft_tokens,
            draft_matches,
        })
    }

    /// Select owned paired-radix handles under a short, isolated prefix lock.
    fn select_dflash_paired_prefix(
        &self,
        partition: DflashPromptPartition<'_>,
    ) -> DflashPairedPrefixPlan {
        debug_assert!(
            !higgs_models::mlx_exec::held(),
            "paired dSpark radix selection must precede the MLX execution gate"
        );
        self.cache_metrics
            .radix_lookups
            .fetch_add(1, Ordering::Relaxed);
        self.cache_metrics
            .paired_radix_lookups
            .fetch_add(1, Ordering::Relaxed);

        // Keeping the mutex guard local to this helper makes it structurally
        // impossible for dflash_decode to retain the prefix lock while it
        // later acquires the model lock or process-wide MLX gate.
        let mut prefix_cache = lock_or_recover(&self.prefix_cache);
        let lookup = match prefix_cache.plan_memory_paired_prefix(partition.body()) {
            Ok(plan) => plan,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "Paired dSpark radix selection failed; cold-prefilling"
                );
                None
            }
        };
        let publication_ticket = prefix_cache.paired_prepare_ticket();
        DflashPairedPrefixPlan {
            lookup,
            publication_ticket,
        }
    }

    /// `DFlash` block-diffusion speculative decode loop.
    ///
    /// Each round: drafter proposes a `block_size` block from the target's tap
    /// hidden states, the target verifies the block in a single tape-recording
    /// forward, `accept_prefix` takes the longest greedy-matching prefix, and a
    /// GDN tape replay rolls partial-accept state back bit-exactly. Headline
    /// metric for tuning is `accept_len` (mean accepted tokens per round).
    ///
    /// The gate/EMA/thermal logic is written once here; the [`DflashSink`]
    /// decides how produced tokens are delivered — buffered into a
    /// [`GenerationOutput`] ([`DflashBufferedSink`]) or streamed
    /// chunk-by-chunk ([`DflashStreamSink`]). Token production is identical
    /// across sinks, so a streamed response is byte-for-byte equal to the
    /// buffered one (asserted by `dflash_streaming_matches_nonstreaming`).
    #[allow(
        clippy::cast_precision_loss,
        clippy::as_conversions,
        clippy::float_cmp,
        clippy::too_many_lines
    )]
    fn dflash_decode<S: DflashSink>(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        enable_thinking: bool,
        thinking_budget: u32,
        mut sink: S,
    ) -> Result<S::Output, EngineError> {
        let request_t0 = std::time::Instant::now();
        let dflash = self
            .dflash
            .as_ref()
            .ok_or_else(|| EngineError::Generation("DFlash state missing".to_owned()))?;
        let prompt_len = Self::prompt_len(prompt_tokens)?;
        if let Ok(mut v) = self.last_dflash_accepts.lock() {
            v.clear();
        }
        if let Ok(mut v) = self.last_dflash_draft_matches.lock() {
            v.clear();
        }
        if let Ok(mut v) = self.last_dflash_draft_counts.lock() {
            v.clear();
        }
        if let Ok(mut timing) = self.last_dflash_timing.lock() {
            *timing = None;
        }

        let partition = (dflash.is_dspark && prefix_cache_enabled())
            .then(|| {
                DflashPromptPartition::new(
                    prompt_tokens,
                    self.gen_prompt_suffixes.for_request(enable_thinking),
                )
            })
            .flatten();

        // Selection returns only owned handles. The helper's local prefix guard
        // is gone before this function can acquire model/MLX state.
        let paired_plan = partition.map(|partition| self.select_dflash_paired_prefix(partition));
        // Declared before model/MLX/drafter guards so Drop runs after them on
        // every success and error path.
        let mut deferred_touch = DeferredPairedTouch::new(&self.prefix_cache);

        let mut model = lock_or_recover(&self.model);
        // `generate_inner` dispatches DFlash before `prepare_generation`, whose
        // `PreparedGeneration` normally owns this token for plain AR. Acquire
        // the same process-global MLX gate here and retain it across shared
        // prefill, draft, verify, rollback, and sink finalization. Without it,
        // the first sanctioned eval panics in debug and concurrent release
        // requests can race MLX's process-global Metal command buffer.
        let mlx_gate = higgs_models::mlx_exec::acquire();
        debug_assert!(higgs_models::mlx_exec::held());
        let mut drafter = lock_or_recover(&dflash.drafter);
        let is_dspark = drafter.config.is_dspark();

        let prefill = self.dflash_prepare_prefill(
            &mut model,
            &mut drafter,
            &mlx_gate,
            prompt_tokens,
            partition,
            paired_plan,
            &mut deferred_touch,
            dflash,
        )?;
        let DflashPrefillOutcome {
            mut cache,
            mut draft_cache,
            logits: prefill_logits,
            taps,
            reused_prefix_len,
        } = prefill;
        if reused_prefix_len > 0 {
            let reused = u64::try_from(reused_prefix_len).unwrap_or(u64::MAX);
            self.cache_metrics
                .radix_hits
                .fetch_add(1, Ordering::Relaxed);
            self.cache_metrics
                .paired_radix_hits
                .fetch_add(1, Ordering::Relaxed);
            self.cache_metrics
                .prefill_saved_tokens
                .fetch_add(reused, Ordering::Relaxed);
            tracing::debug!(
                reused_prefix_len,
                prompt_len = prompt_tokens.len(),
                "Paired dSpark radix hit"
            );
        }

        // Sample first token.
        let last_logits = prefill_logits.index((.., -1, ..));
        let first_token = sample(&last_logits, params).map_err(EngineError::Mlx)?;
        eval([&first_token]).map_err(EngineError::Mlx)?;

        let first_token_id: u32 = first_token.item();
        let prefill_elapsed = request_t0.elapsed();
        let mut tokens: Vec<u32> = vec![first_token_id];
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        // Deliver the first token, then terminate early if it is EOS or the
        // request asked for a single token.
        let first_completion = Self::completion_len(&tokens)?;
        let first_forced = if self.eos_token_ids.contains(&first_token_id) {
            Some("stop")
        } else if max_tokens <= 1 {
            Some("length")
        } else {
            None
        };
        let first_stop = sink.emit(
            self,
            &tokens,
            stop_sequences,
            prompt_len,
            first_completion,
            first_forced,
        )?;
        if first_forced.is_some() || first_stop {
            if let Ok(mut timing) = self.last_dflash_timing.lock() {
                *timing = Some(GenerationPhaseTiming {
                    prefill: prefill_elapsed,
                    decode: std::time::Duration::ZERO,
                    request: request_t0.elapsed(),
                    decode_tokens: 0,
                });
            }
            return sink.finish(
                self,
                &tokens,
                first_forced.unwrap_or("stop"),
                prompt_len,
                first_completion,
            );
        }

        let mut last_token = i32::try_from(first_token_id)
            .map_err(|_| EngineError::Generation("first_token_id overflow for i32".to_owned()))?;
        let mut start = i32::try_from(prompt_len)
            .map_err(|_| EngineError::Generation("prompt_len overflow for i32".to_owned()))?;
        let (mut current_taps, mut dspark_frontier) = if is_dspark {
            (
                Vec::new(),
                Some(DflashTapFrontier::new(
                    draft_cache.position(),
                    start,
                    taps,
                    dflash.tap_layers.len(),
                )?),
            )
        } else {
            (taps, None)
        };
        let mut dspark_ledger = if is_dspark {
            let prompt_boundary = usize::try_from(prompt_len)
                .map_err(|_| EngineError::Generation("prompt boundary overflow".to_owned()))?;
            let mut ledger = TokenLedger::new(prompt_boundary);
            ledger
                .emit_pending(first_token_id)
                .map_err(session_ledger_error)?;
            Some(ledger)
        } else {
            None
        };
        // Adaptive (entropy-gated) block size. Acceptance length is the entropy
        // proxy: predictable/low-entropy regions accept long blocks (big win,
        // byte-exact), uncertain/high-entropy regions reject them. So grow the
        // block toward `block_max` when a block is fully accepted and shrink
        // toward `block_min` (≈ plain decode) when drafts are rejected, which
        // avoids paying for wide target blocks that cannot amortize their
        // work. Disable with HIGGS_DFLASH_ADAPTIVE=0; floor via
        // HIGGS_DFLASH_MIN_BLOCK.
        let block_max = dflash.block_size;
        let block_min = std::env::var("HIGGS_DFLASH_MIN_BLOCK")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&v| v >= 1)
            .map_or(2, |v| v.min(block_max));
        // dSpark's bidirectional trunk and log-SNR schedule are trained for a
        // fixed block. Its output/verify work may be capped independently, but
        // resizing the trunk changes the trained distribution and shape-fails
        // the fixed conditioning tensor.
        // Batched dSpark currently has a greedy, no-thinking proof boundary.
        // Sampling would consume RNG for discarded verifier rows, penalties
        // would need a transactional history, and forced thinking tokens would
        // replace a target after its successor had already advanced the cache.
        // Fail closed to the commit-only S=1 oracle for those requests even if
        // block mode was explicitly selected for profiling.
        let request_verify_mode =
            dflash
                .verify_mode
                .for_request(is_dspark, params, think_close_token.is_some());
        let canonical_verify = request_verify_mode == DFlashVerifyMode::CanonicalS1;
        if is_dspark && dflash.verify_mode == DFlashVerifyMode::BatchedTape && canonical_verify {
            tracing::info!(
                sampled = params.temperature != 0.0,
                penalized = params.has_penalties(),
                thinking = think_close_token.is_some(),
                "dSpark block verifier is outside its exact policy boundary; using canonical S=1"
            );
        }
        let adaptive =
            !is_dspark && std::env::var("HIGGS_DFLASH_ADAPTIVE").map_or(true, |v| v != "0");
        let mut block_size = block_max;
        // Smoothed utilization: a single hard token inside otherwise-predictable
        // text (code dip ~0.38 ≈ prose plateau ~0.36) shouldn't collapse the
        // block — only a *persistently* low utilization should. EMA separates
        // "occasional dip" from "sustained high entropy".
        let mut ema_util = 1.0_f64;
        let mask_id = dflash.mask_token_id;
        let t_start = std::time::Instant::now();
        // Acceptance-length aggregation: mean accepted tokens per round is the
        // headline metric for DFlash tuning (see P5).
        let mut total_accepted: u64 = 0;
        let mut rounds: u64 = 0;

        // Realized-speedup gate (auto-calibrated, no per-model knob). Measure
        // T_ar (wall per AR token) live; each spec round compute the realized
        // speedup ratio = n_accepted * T_ar / step_wall. If the EMA of that
        // ratio falls below 1.0, spec is actually slower than plain AR for this
        // text (MoE's fast AR raises the bar; a fixed entropy threshold can't
        // see that), so floor to AR and re-probe periodically. Start in AR for
        // one step to calibrate T_ar, then enter spec. Disable:
        // HIGGS_DFLASH_GATE=0.
        // Thinking-budget token replacement is incompatible with the speed
        // gate's pending-token convention; no-thinking dSpark can use the same
        // realized-speed floor as Modal DFlash now that every AR-floor tap is
        // accumulated into a contiguous drafter-context backlog below.
        let gate = think_close_token.is_none()
            && std::env::var("HIGGS_DFLASH_GATE").map_or(true, |v| v != "0");
        let mut t_ar_ema: Option<f64> = None;
        let mut ratio_ema = 2.0_f64;
        let mut in_ar = true;
        let mut ar_run: u32 = 0;
        // Re-probe spec sparsely with exponential backoff. In a sustained
        // high-entropy region every probe is a wasted slow spec round (~4
        // decode-units for ~1-2 tokens), so a fixed cadence taxed the floor by
        // ~probe_cost/cadence (≈11% at 32 — the measured 0.89× prose gap; the
        // per-step taps were measured free, ~0% — they are lazy clones).
        // Each losing probe doubles the interval (amortizing the cost toward 0 on
        // sustained prose); a winning probe resets to base so a regime shift back
        // to spec-friendly text is caught within `AR_PROBE_BASE` tokens.
        const AR_PROBE_BASE: u32 = 32;
        // ponytail: cap the backoff at 512 — residual probe tax <1% (≈0.99×),
        // and recovery never lags a regime change by more than ~512 tokens.
        const AR_PROBE_MAX: u32 = 512;
        let mut probe_every = AR_PROBE_BASE;

        // The first decode after prefill is kernel-cold on Metal (~8× steady
        // state: measured t_ar=187ms vs ~23ms warm). Seeding the gate's AR
        // reference from it froze t_ar high, so the realized-speedup ratio was
        // always >1 and the gate NEVER floored — a dead no-op. Fix: spend a short
        // warm-up window in AR, discard the cold sample, seed t_ar from the warm
        // steps, then hand off to spec. These are real greedy tokens, not waste.
        // ponytail: no periodic re-calibration — t_ar only updates on floored
        // steps, so during a spec-winning streak it holds its last warm value
        // (can't drift high to mask a loss); a real degradation re-floors and
        // refreshes it. Add periodic recal only if AR-baseline thermal drift
        // during long spec streaks ever proves to matter.
        const T_AR_CALIB: u32 = 3;
        let mut calibrated = false;

        // Spec-side cold-start grace (symmetric to T_AR_CALIB). The first spec
        // rounds after every (re-)entry are cold on two axes the warm t_ar is
        // not: the verify/drafter/lm_head Metal kernels are kernel-cold
        // (step_wall ~2x inflated until they JIT-warm) and the drafter KV cache
        // is empty (accept length climbs from ~3 to its steady ~6 as it fills).
        // Judging those rounds floored winning workloads (9B code: 1.53x->0.83x)
        // and then death-spiralled (floored -> spec re-cools -> re-probe also
        // looks bad). So don't update ratio_ema or floor until the burst warms.
        const SPEC_WARMUP: u32 = 3;
        let mut spec_warmup_left: u32 = 0;

        // Periodic t_ar re-calibration. t_ar is only measured during AR steps, so
        // a long spec-winning streak never refreshes it — and t_ar drifts with
        // thermal (measured 25ms cool vs 101ms throttled on the same model). A
        // stale t_ar skews `ratio = n_acc*t_ar/step_wall` and the gate can miss a
        // regime where spec started losing. So every RECAL_EVERY judged winning
        // rounds, force ONE AR step to refresh t_ar at the current thermal state.
        // ~1 AR token per ~240 generated (<0.5% overhead); unlike a probe it does
        // not re-arm the warm-up grace or reset ratio_ema (1 AR step barely cools
        // the drafter, and we want to keep the winning EMA).
        const T_AR_RECAL_EVERY: u32 = 48;
        let mut spec_since_recal: u32 = 0;
        let mut recal = false;

        // MTP floor. When the target ships a native MTP head (Qwen3.6-A3B) and
        // decoding is greedy, floored rounds run MTP speculative cycles instead
        // of plain AR steps. Plain AR is the wrong floor on such models: MTP
        // decodes ~1.5x faster than AR on Qwen3.6-35B-A3B (measured 48 vs 34
        // tok/s), so gating DFlash against AR keeps DFlash active in regions
        // where MTP would win (measured: DFlash 39 tok/s on thinking prose).
        // With the MTP floor, t_ar_ema measures the MTP per-token rate, so the
        // gate floors DFlash to MTP unless DFlash genuinely beats it. The head
        // hidden/cache are re-seeded at each floor entry — the backbone
        // advances during spec rounds, so a prior window's state is stale.
        // The current MTP floor cycle is greedy-only and has no penalty-history
        // input. Do not let a performance floor change request semantics: hot
        // sampling and penalized greedy requests use the ordinary AR floor.
        // dSpark's exact proof boundary is the ordinary S=1 target transition.
        // A native MTP sidecar has its own S>1 verify/rollback schedule and
        // cannot be introduced as a performance floor until it satisfies the
        // same transaction gates.
        let mtp_floor = gate
            && !is_dspark
            && model.has_mtp()
            && params.temperature == 0.0
            && !params.has_penalties();
        let mut mtp_floor_hidden: Option<Array> = None;
        let mut mtp_floor_cache: Option<higgs_models::MtpCache> = None;
        let mut spec_entered_once = false;
        // MTP-cycle convention: `last_token` is sampled but NOT yet emitted
        // (the next cycle emits it as its confirmed token). The dflash AR/spec
        // convention is the opposite (`last_token` already emitted, pending a
        // forward). This flag tracks which convention `last_token` is in;
        // leaving the cycle chain flushes the pending token exactly once.
        // Getting this wrong duplicates a token at floor entry and drops one
        // at floor exit (measured: "1,, 2" / "4 55" on deterministic output).
        let mut mtp_pending = false;

        loop {
            let round_t0 = std::time::Instant::now();
            if gate && in_ar {
                // AR floor: plain S=1 decode (byte-exact). Measures T_ar live.
                //
                // Taps are only consumed by the drafter when we re-enter spec next
                // round, i.e. on the step that hands control back ("leaving"): the
                // end of the initial calibration window, or a probe cooldown. Use
                // the optimized tap-less `forward` for every other floored step.
                // The two paths share `forward_raw_hidden` (identical layer loop,
                // mask, and KV/GDN cache mutation) and the same `project_logits`,
                // so logits + cache are byte-identical; the only thing dropped is a
                // tap clone the next floored step would overwrite unused.
                let calibrating = !calibrated;
                // recal: a short window to refresh a stale t_ar mid-streak. With
                // the MTP floor it must span entry+cycle+leaving (3 steps) so it
                // actually measures an MTP cycle — a 1-step window is
                // leaving-only plain AR, which would recalibrate the floor to
                // the AR rate and let a losing spec streak keep running.
                let window = if recal {
                    if mtp_floor { 3 } else { 1 }
                } else if calibrating {
                    // MTP-floor calibration needs 2 extra steps: the entry step
                    // seeds the hidden (no cycle yet) and the first cycle is
                    // kernel-cold on the batched verify path — neither is a
                    // valid floor sample.
                    if mtp_floor {
                        T_AR_CALIB + 2
                    } else {
                        T_AR_CALIB
                    }
                } else {
                    probe_every
                };
                let leaving = ar_run + 1 >= window;
                let need_taps = leaving;
                // Tokens emitted by this floored step (MTP cycles emit several);
                // t_ar_ema below is per-token, so it stays comparable across
                // plain-AR and MTP-floor steps.
                let mut floor_tokens: u32 = 1;
                let mut was_mtp_cycle = false;
                if let (false, Some(hidden)) = (need_taps, mtp_floor_hidden.take()) {
                    was_mtp_cycle = true;
                    // MTP-floor cycle: draft with the native head, batch-verify,
                    // rollback rejected drafts. Head KV was (re)built at this
                    // floor entry; `hidden` is the backbone hidden of the token
                    // preceding `last_token` from the previous entry step/cycle.
                    let mtp_cache = match mtp_floor_cache.as_mut() {
                        Some(c) => c,
                        None => {
                            mtp_floor_cache.insert(model.make_mtp_cache().ok_or_else(|| {
                                EngineError::Generation("MTP cache creation failed".to_owned())
                            })?)
                        }
                    };
                    let confirmed = u32::try_from(last_token).map_err(|_| {
                        EngineError::Generation("confirmed token overflow u32".to_owned())
                    })?;
                    // Tapped cycle: collect the emitted positions' tap hiddens
                    // and append them to the drafter's backlog, keeping its
                    // context cache contiguous across the floored stretch. A
                    // context hole blinds the drafter at the next probe (it
                    // cannot continue "…, 87, 88," without the recent tokens)
                    // — measured accept collapse ~6 → ~1.2 on deterministic
                    // text, which made every probe a false negative.
                    let (result, cycle_taps) = crate::mtp::mtp_cycle_tapped(
                        &mut model,
                        &mut cache,
                        mtp_cache,
                        &hidden,
                        confirmed,
                        self.tuning.mtp_draft_n_max(),
                        &dflash.tap_layers,
                    )?;
                    Self::append_taps(&mut current_taps, &cycle_taps).map_err(EngineError::Mlx)?;
                    // `result.tokens` begins with the (previously unemitted)
                    // pending confirmed token — the cycle emits it.
                    floor_tokens = 0;
                    for &tok in &result.tokens {
                        tokens.push(tok);
                        floor_tokens += 1;
                        if self.eos_token_ids.contains(&tok) {
                            break;
                        }
                    }
                    // The backbone cache advanced by the full accepted batch
                    // regardless of EOS truncation above.
                    start += i32::try_from(result.tokens.len())
                        .map_err(|_| EngineError::Generation("cycle len overflow".to_owned()))?;
                    last_token = i32::try_from(result.next_token_id).map_err(|_| {
                        EngineError::Generation("mtp next token overflow i32".to_owned())
                    })?;
                    mtp_pending = true;
                    mtp_floor_hidden = Some(result.hidden);
                    if std::env::var("HIGGS_DFLASH_TRACE").is_ok() {
                        tracing::info!(
                            drafted = result.drafted,
                            accepted = result.accepted_drafts,
                            emitted = floor_tokens,
                            dt_ms = format!("{:.1}", round_t0.elapsed().as_secs_f64() * 1e3),
                            "MTP-floor cycle"
                        );
                    }
                } else {
                    // Leaving the cycle chain: flush the pending confirmed token
                    // (cycles sample it but never emit it — see `mtp_pending`).
                    // It is emitted BEFORE this step forwards it, keeping text
                    // order; if it is EOS, skip the forward and let the loop
                    // bottom finish (a forward would sample a post-EOS token).
                    let mut pending_was_eos = false;
                    if mtp_pending {
                        let pend = u32::try_from(last_token).map_err(|_| {
                            EngineError::Generation("pending token overflow u32".to_owned())
                        })?;
                        tokens.push(pend);
                        mtp_pending = false;
                        pending_was_eos = self.eos_token_ids.contains(&pend);
                    }
                    if pending_was_eos {
                        floor_tokens = 1;
                    } else {
                        let dspark_forward = dspark_ledger
                            .as_mut()
                            .map(|ledger| {
                                ledger
                                    .begin_cache_only_forward()
                                    .map_err(session_ledger_error)
                            })
                            .transpose()?;
                        let single = Array::from_slice(&[last_token], &[1, 1]);
                        let ar_logits = if need_taps && mtp_floor {
                            // MTP-floor leaving step: APPEND this position's taps to
                            // the drafter backlog (the drafter has not consumed the
                            // floored stretch yet), unlike the plain-AR leaving step
                            // which replaces already-consumed taps.
                            let (logits, ar_taps) = model
                                .forward_with_taps(&single, None, &mut cache, &dflash.tap_layers)
                                .map_err(EngineError::Mlx)?;
                            Self::append_taps(&mut current_taps, &ar_taps)
                                .map_err(EngineError::Mlx)?;
                            logits
                        } else if need_taps || is_dspark {
                            let (logits, ar_taps) = model
                                .forward_with_taps(&single, None, &mut cache, &dflash.tap_layers)
                                .map_err(EngineError::Mlx)?;
                            if is_dspark {
                                // dSpark's cross-attention cache must see every
                                // target position skipped while speculation is
                                // floored. These lazy tap clones are appended to
                                // the pending verify taps and consumed together
                                // on the next probe, so cache length and `start`
                                // remain contiguous across AR↔spec transitions.
                                let next_boundary = start.checked_add(1).ok_or_else(|| {
                                    EngineError::Generation(
                                        "dSpark AR target boundary overflow".to_owned(),
                                    )
                                })?;
                                let frontier = dspark_frontier.take().ok_or_else(|| {
                                    EngineError::Generation(
                                        "dSpark AR floor lost its tap frontier".to_owned(),
                                    )
                                })?;
                                dspark_frontier = Some(frontier.append(next_boundary, ar_taps)?);
                            } else {
                                // Append, never replace: the backlog may still
                                // hold un-consumed taps (the chunked-prefill
                                // tail plus every floored AR step below). A
                                // replace here pinned the drafter cache at the
                                // last chunk boundary and every sampled request
                                // with a >1-chunk prompt died at its first spec
                                // probe with "DFlash context transaction
                                // diverged: drafter=<chunk+1>".
                                Self::append_taps(&mut current_taps, &ar_taps)
                                    .map_err(EngineError::Mlx)?;
                            }
                            logits
                        } else if mtp_floor {
                            // MTP-floor entry step: a plain AR step that also keeps
                            // the backbone hidden (so the next floored step can run
                            // an MTP cycle — mirrors `mtp_generate`'s bootstrap) and
                            // this position's taps for the drafter backlog.
                            let (hidden, logits, step_taps) = model
                                .forward_with_hidden_taps(
                                    &single,
                                    None,
                                    &mut cache,
                                    &dflash.tap_layers,
                                )
                                .map_err(EngineError::Mlx)?;
                            mtp_floor_hidden = Some(hidden.index((.., -1.., ..)));
                            Self::append_taps(&mut current_taps, &step_taps)
                                .map_err(EngineError::Mlx)?;
                            logits
                        } else {
                            // Plain-AR floor step: collect this position's tap
                            // like the MTP floor does. A context hole blinds
                            // the drafter at the next probe and breaks the
                            // `drafter position + backlog == start` transaction
                            // invariant. The cost is one lazy tap clone per
                            // floored step.
                            let (logits, ar_taps) = model
                                .forward_with_taps(&single, None, &mut cache, &dflash.tap_layers)
                                .map_err(EngineError::Mlx)?;
                            Self::append_taps(&mut current_taps, &ar_taps)
                                .map_err(EngineError::Mlx)?;
                            logits
                        };
                        if let Some(ticket) = dspark_forward {
                            dspark_ledger
                                .as_mut()
                                .ok_or_else(|| {
                                    EngineError::Generation(
                                        "dSpark AR floor lost its token ledger".to_owned(),
                                    )
                                })?
                                .complete_cache_only_forward(ticket)
                                .map_err(session_ledger_error)?;
                        }
                        let ar_row =
                            apply_penalties(&ar_logits.index((.., -1, ..)), &tokens, params)
                                .map_err(EngineError::Mlx)?;
                        let ar_next = sample(&ar_row, params).map_err(EngineError::Mlx)?;
                        eval([&ar_next]).map_err(EngineError::Mlx)?;
                        let ar_id: u32 = ar_next.item();
                        if mtp_floor && !need_taps {
                            // MTP-floor entry step: the sampled token becomes the
                            // pending confirmed for the next cycle — NOT emitted
                            // here (the cycle emits it; pushing it here duplicates
                            // it in the output).
                            last_token = i32::try_from(ar_id).map_err(|_| {
                                EngineError::Generation("ar token overflow".to_owned())
                            })?;
                            mtp_pending = true;
                            floor_tokens = 0;
                        } else {
                            tokens.push(ar_id);
                            if let Some(ledger) = dspark_ledger.as_mut() {
                                ledger.emit_pending(ar_id).map_err(session_ledger_error)?;
                            }
                            last_token = i32::try_from(ar_id).map_err(|_| {
                                EngineError::Generation("ar token overflow".to_owned())
                            })?;
                        }
                        start += 1;
                    }
                }
                let dt = round_t0.elapsed().as_secs_f64();
                // Floor-rate sample validity. Plain floor: skip only the
                // kernel-cold first decode of the initial calibration window.
                // MTP floor: the floor rate is the MTP-cycle rate, so sample
                // ONLY cycles (entry/leaving are 1-token AR steps ~1.5x slower
                // — blending them in overestimates the floor and lets a losing
                // spec streak survive the gate), and skip the first-ever cycle
                // (kernel-cold batched verify, measured ~2x steady state).
                let update_ema = if mtp_floor {
                    was_mtp_cycle && !(calibrating && ar_run <= 1)
                } else {
                    !(calibrating && ar_run == 0)
                };
                if update_ema {
                    let dt_tok = dt / f64::from(floor_tokens.max(1));
                    t_ar_ema = Some(t_ar_ema.map_or(dt_tok, |e| 0.7f64.mul_add(e, 0.3 * dt_tok)));
                }
                ar_run += 1;
                // Hand back to spec once the warm calibration window completes, or
                // after a probe cooldown re-tests whether spec is worthwhile again.
                //
                // Exception — start floored on MTP-floor targets: the floor is
                // the strong default there (~1.5x AR on Qwen3.6-A3B), so spec
                // must prove itself at the first probe instead of getting the
                // benefit of the doubt. This drops the front-loaded spec grace +
                // losing probes on floor-winning workloads (measured ~20% of a
                // short code run); a spec-winning workload engages at the first
                // probe, ~probe_every floored steps in. The head cache/hidden
                // stay valid across this handoff: the whole calibration window
                // ran through the floor, so backbone and MTP head are aligned.
                let start_floored = mtp_floor && calibrating;
                if leaving && start_floored {
                    ar_run = 0;
                    calibrated = true;
                } else if leaving {
                    in_ar = false;
                    ar_run = 0;
                    calibrated = true;
                    // The spec rounds advance the backbone without the MTP head,
                    // so the head KV is stale for the next floor window; the
                    // hidden was already drained by the leaving step's take().
                    mtp_floor_cache = None;
                    if recal {
                        // Just refreshed t_ar mid-streak: resume spec with the
                        // winning EMA intact and no warm-up (the drafter barely
                        // cooled over one AR step). need_taps was true, so
                        // current_taps is fresh for the next draft.
                        recal = false;
                    } else {
                        // Grant the fresh spec burst its cold-start grace and
                        // re-seed the EMA to the optimistic prior. Re-seeding (not
                        // blending) is load-bearing: otherwise a re-probe inherits
                        // the stale sub-1.0 ratio_ema and re-floors in one warm
                        // round, undoing the grace. Mirrors the t_ar_ema map_or seed.
                        //
                        // With the MTP floor the cost asymmetry flips: flooring a
                        // winning workload briefly costs little (the floor is MTP,
                        // ~1.5x AR), while each unjudged warm-up round on a losing
                        // workload costs a full spec round (~150-200ms for ~2
                        // tokens) per probe. So re-probes get a 1-round grace; the
                        // full grace applies only to the first spec entry (and
                        // always on plain-AR floors, where mis-flooring is costly).
                        spec_warmup_left = if mtp_floor && spec_entered_once {
                            1
                        } else {
                            SPEC_WARMUP
                        };
                        spec_entered_once = true;
                        ratio_ema = 2.0;
                    }
                }
            } else {
                let completion_len = Self::completion_len(&tokens)?;
                let remaining = usize::try_from(max_tokens.saturating_sub(completion_len))
                    .map_err(|_| EngineError::Generation("remaining tokens overflow".to_owned()))?;
                let exact_sequential = canonical_verify || (is_dspark && remaining == 1);
                let n_accepted = if exact_sequential {
                    let sequential_limit = think_close_token.map_or(remaining, |_| {
                        if seen_think_close {
                            remaining
                        } else {
                            let until_forced =
                                thinking_budget.saturating_sub(thinking_tokens).max(1);
                            remaining.min(usize::try_from(until_forced).unwrap_or(usize::MAX))
                        }
                    });
                    let ledger = dspark_ledger.as_mut().ok_or_else(|| {
                        EngineError::Generation(
                            "canonical dSpark decode lost its token ledger".to_owned(),
                        )
                    })?;
                    if ledger.emitted_tokens() != tokens.as_slice() {
                        return Err(EngineError::Generation(
                            "canonical dSpark ledger/output sequence diverged before round"
                                .to_owned(),
                        ));
                    }
                    let mut committed_frontier = None;
                    let committed = drive_canonical_dspark_round(
                        ledger,
                        |anchor_ticket, history_before_anchor| {
                            let frontier = dspark_frontier.take().ok_or_else(|| {
                                EngineError::Generation(
                                    "canonical dSpark decode lost its tap frontier".to_owned(),
                                )
                            })?;
                            let mut round = self.canonical_dflash_round(
                                &mut model,
                                &mut cache,
                                &mut drafter,
                                &mut draft_cache,
                                frontier,
                                anchor_ticket,
                                history_before_anchor,
                                remaining,
                                sequential_limit,
                                params,
                                stop_sequences,
                                think_close_token,
                                dflash,
                            )?;

                            if let Some(close_id) = think_close_token
                                && !seen_think_close
                            {
                                let mut force_pending_at = None;
                                for (index, &token) in round.successors.iter().enumerate() {
                                    if token == close_id {
                                        seen_think_close = true;
                                        break;
                                    }
                                    thinking_tokens = thinking_tokens.saturating_add(1);
                                    if thinking_tokens >= thinking_budget {
                                        seen_think_close = true;
                                        force_pending_at = Some(index);
                                        tracing::info!(
                                            budget = thinking_budget,
                                            "Thinking budget reached, forcing </think>"
                                        );
                                        break;
                                    }
                                }
                                if let Some(index) = force_pending_at {
                                    if index + 1 != round.successors.len() {
                                        return Err(EngineError::Generation(format!(
                                            "thinking policy selected canonical successor {index}, but {} successors were committed",
                                            round.successors.len()
                                        )));
                                    }
                                    round.replace_pending_successor(close_id)?;
                                }
                            }

                            let (frontier, commit) = round.into_lease_parts()?;
                            committed_frontier = Some(frontier);
                            Ok(commit)
                        },
                    )?;
                    tokens.extend_from_slice(&committed.successors);
                    if ledger.emitted_tokens() != tokens.as_slice() {
                        return Err(EngineError::Generation(
                            "canonical dSpark ledger/output sequence diverged after round commit"
                                .to_owned(),
                        ));
                    }
                    let committed_frontier = committed_frontier.ok_or_else(|| {
                        EngineError::Generation(
                            "canonical dSpark round completed without its replacement frontier"
                                .to_owned(),
                        )
                    })?;
                    let accepted_count = committed.successors.len();
                    let accepted_rows = i32::try_from(accepted_count).map_err(|_| {
                        EngineError::Generation(
                            "canonical dSpark accepted count overflow".to_owned(),
                        )
                    })?;
                    if let Ok(mut values) = self.last_dflash_accepts.lock() {
                        values.push(u32::try_from(accepted_count).unwrap_or(0));
                    }
                    if let Ok(mut values) = self.last_dflash_draft_matches.lock() {
                        values.push(u32::try_from(committed.draft_matches).unwrap_or(0));
                    }
                    if let Ok(mut values) = self.last_dflash_draft_counts.lock() {
                        values.push(u32::try_from(committed.draft_tokens.len()).unwrap_or(0));
                    }
                    rounds += 1;
                    total_accepted += accepted_count as u64;
                    if std::env::var("HIGGS_DFLASH_TRACE").is_ok()
                        && (tokens.len() <= 32 || std::env::var("HIGGS_DFLASH_TRACE_ALL").is_ok())
                    {
                        tracing::info!(
                            drafts = ?committed.draft_tokens,
                            verify_argmax = ?committed.successors,
                            n_accepted = accepted_rows,
                            accepted = ?committed.successors,
                            "canonical DFlash iter trace"
                        );
                    }
                    last_token = i32::try_from(*committed.successors.last().ok_or_else(|| {
                        EngineError::Generation(
                            "canonical dSpark round returned no successor".to_owned(),
                        )
                    })?)
                    .map_err(|_| {
                        EngineError::Generation("accepted token overflow i32".to_owned())
                    })?;
                    start = committed_frontier.target_boundary();
                    dspark_frontier = Some(committed_frontier);
                    accepted_rows
                } else {
                    // Experimental S>1 throughput path. It remains isolated
                    // behind the explicit block verifier policy; the default
                    // dSpark path above is the shared S=1 transaction.
                    // The noise block is FIXED at the drafter's trained block
                    // size — validate_noise rejects any other shape (observed
                    // 500: "drafter noise must be [1, 16, ...], got [1, 8, ...]"
                    // after the adaptive gate shrank the block). Adaptivity is
                    // applied to the PROPOSAL cap below instead, which is where
                    // the target-verify savings actually come from.
                    let block_size_us = usize::try_from(dflash.block_size).map_err(|_| {
                        EngineError::Generation("block_size overflow for usize".to_owned())
                    })?;
                    let mut block_tokens = vec![mask_id; block_size_us];
                    if let Some(slot) = block_tokens.get_mut(0) {
                        *slot = last_token;
                    }
                    let block_ids = Array::from_slice(&block_tokens, &[1, dflash.block_size]);
                    let noise_embedding = model
                        .embed_token_ids(&block_ids)
                        .map_err(EngineError::Mlx)?;
                    let draft_transaction = if let Some(frontier) = dspark_frontier.as_ref() {
                        frontier.validate_live_draft(&draft_cache)?;
                        drafter.stage_forward(&noise_embedding, &frontier.taps, &draft_cache)
                    } else {
                        drafter.stage_forward(&noise_embedding, &current_taps, &draft_cache)
                    }
                    .map_err(EngineError::Mlx)?;
                    let staged_position = draft_transaction.position().map_err(EngineError::Mlx)?;
                    if staged_position != start {
                        return Err(EngineError::Generation(format!(
                            "DFlash context transaction diverged: drafter={staged_position} target={start}"
                        )));
                    }
                    let round_draft_cap =
                        dflash_tail_draft_cap(dflash.draft_cap, remaining, is_dspark)
                            .min(block_size.max(1));
                    let proposal = dflash_propose_tokens(
                        &model,
                        &drafter,
                        draft_transaction.hidden(),
                        last_token,
                        round_draft_cap,
                        dflash.dspark_target_head,
                    )?;
                    let dspark_anchor_ticket = dspark_ledger
                        .as_mut()
                        .map(|ledger| {
                            ledger
                                .begin_speculative_round()
                                .map_err(session_ledger_error)
                        })
                        .transpose()?;
                    // Verify [anchor, draft_0, ...] in one target forward and
                    // use the GDN tape to select the committed prefix.
                    let verify_input = dflash_verify_input(last_token, &proposal)?;
                    let verify_len = *verify_input.shape().get(1).ok_or_else(|| {
                        EngineError::Generation("verify input missing T axis".to_owned())
                    })?;
                    // A verifier transaction must advance every model layer.
                    // The former early-exit experiment could not distinguish a
                    // skipped attention layer from an advanced layer with no
                    // GDN tape, so partial commit could trim retained history.
                    let early_exit = None;
                    let row_schedule = if is_dspark {
                        higgs_models::qwen3_next::DFlashRowSchedule::CanonicalS1
                    } else {
                        higgs_models::qwen3_next::DFlashRowSchedule::NativeBatch
                    };
                    let (verify_logits, verify_taps, layer_tapes) = model
                        .forward_with_taps_tape_scheduled(
                            &verify_input,
                            None,
                            &mut cache,
                            &dflash.tap_layers,
                            early_exit,
                            row_schedule,
                        )
                        .map_err(EngineError::Mlx)?;
                    // Keep draft tokens device-resident until the target verify
                    // graph reaches its single logits barrier. Pulling them
                    // earlier serializes proposal and verification and can add
                    // a full GPU/host synchronization to every round.
                    let (verify_flat, draft_u32) = dflash_resolve_target_then_draft(
                        || crate::mtp::verify_targets(&verify_logits, Some(params), Some(&tokens)),
                        || proposal.host_tokens(),
                    )?;
                    let mut accepted = accept_prefix(&draft_u32, &verify_flat);
                    let draft_matches = accepted.len().saturating_sub(1);
                    // The cache convention retains one verify-input token
                    // per emitted token, so the same truncation drives tape
                    // rollback below.
                    accepted.truncate(remaining);
                    if let Some(eos_at) = accepted
                        .iter()
                        .position(|token| self.eos_token_ids.contains(token))
                    {
                        accepted.truncate(eos_at + 1);
                    }
                    if let Some(stop_len) =
                        self.dflash_stop_prefix_len(&tokens, &accepted, stop_sequences)?
                    {
                        accepted.truncate(stop_len);
                    }
                    let (mut verify_round, draft_u32, draft_matches) = (
                        DFlashVerifyRound::TentativeTape {
                            target_rows: verify_len,
                            target_tokens: verify_flat,
                            output_tokens: accepted,
                            taps: verify_taps,
                            layer_tapes,
                            pending_replaced: false,
                            expected_taps: dflash.tap_layers.len(),
                        },
                        draft_u32,
                        draft_matches,
                    );

                    draft_transaction
                        .commit(&mut draft_cache)
                        .map_err(EngineError::Mlx)?;
                    if is_dspark && draft_cache.position() != start {
                        return Err(EngineError::Generation(format!(
                            "batched dSpark committed drafter boundary {}, expected {start}",
                            draft_cache.position()
                        )));
                    }

                    if let Some(close_id) = think_close_token {
                        let mut forced_index = None;
                        if !seen_think_close {
                            for (index, &token) in verify_round.output_tokens().iter().enumerate() {
                                if token == close_id {
                                    seen_think_close = true;
                                    break;
                                }
                                thinking_tokens = thinking_tokens.saturating_add(1);
                                if thinking_tokens >= thinking_budget {
                                    seen_think_close = true;
                                    forced_index = Some(index);
                                    tracing::info!(
                                        budget = thinking_budget,
                                        "Thinking budget reached, forcing </think>"
                                    );
                                    break;
                                }
                            }
                        }
                        if let Some(index) = forced_index {
                            // The forced close becomes the pending token. Later
                            // verifier rows were conditioned on the replaced target
                            // and must not be committed.
                            verify_round.replace_pending_output(index, close_id)?;
                        }
                    }
                    if let Ok(mut v) = self.last_dflash_accepts.lock() {
                        v.push(u32::try_from(verify_round.output_tokens().len()).unwrap_or(0));
                    }
                    if let Ok(mut v) = self.last_dflash_draft_matches.lock() {
                        v.push(u32::try_from(draft_matches).unwrap_or(0));
                    }
                    if let Ok(mut v) = self.last_dflash_draft_counts.lock() {
                        v.push(u32::try_from(draft_u32.len()).unwrap_or(0));
                    }
                    let n_accepted = verify_round.validate_output_boundary()?;
                    rounds += 1;
                    total_accepted += verify_round.output_tokens().len() as u64;

                    if std::env::var("HIGGS_DFLASH_TRACE").is_ok()
                        && (tokens.len() <= 32 || std::env::var("HIGGS_DFLASH_TRACE_ALL").is_ok())
                    {
                        tracing::info!(
                            drafts = ?draft_u32,
                            verify_argmax = ?verify_round.target_tokens(),
                            n_accepted,
                            accepted = ?verify_round.output_tokens(),
                            "DFlash iter trace"
                        );
                    }

                    // h. Partial accept — GDN-only replay from tape. The verify
                    //    advanced state for ALL positions; on partial rejection we
                    //    restore each GDN layer's snapshot, replay the SSM kernel for
                    //    the n_accepted positions, and trim KV layers by the rejected
                    //    count. Issued lazily (no eval) so it folds into the next
                    //    verify's host barrier.
                    // `verify_len`, not `block_size`: with confidence truncation the
                    // chain can be shorter than the block, and a fully-accepted
                    // truncated chain needs no rollback.
                    let (accepted, verify_taps) =
                        verify_round.commit_target_state(&model, &mut cache)?;
                    let kept_taps: Vec<Array> = verify_taps
                        .into_iter()
                        .map(|tap| tap.index((.., ..n_accepted, ..)))
                        .collect();

                    // i. Update state.
                    if let Some(ticket) = dspark_anchor_ticket {
                        let expected_after = start.checked_add(n_accepted).ok_or_else(|| {
                            EngineError::Generation(
                                "batched dSpark target boundary overflow".to_owned(),
                            )
                        })?;
                        let old_frontier = dspark_frontier.take().ok_or_else(|| {
                            EngineError::Generation(
                                "batched dSpark lost its tap frontier".to_owned(),
                            )
                        })?;
                        if old_frontier.target_boundary() != start {
                            return Err(EngineError::Generation(format!(
                                "batched dSpark frontier ends at {}, target starts at {start}",
                                old_frontier.target_boundary()
                            )));
                        }
                        let next_frontier = DflashTapFrontier::new(
                            start,
                            expected_after,
                            kept_taps,
                            dflash.tap_layers.len(),
                        )?;
                        let ledger = dspark_ledger.as_mut().ok_or_else(|| {
                            EngineError::Generation(
                                "batched dSpark lost its token ledger".to_owned(),
                            )
                        })?;
                        ledger
                            .complete_speculative_round(ticket, &accepted)
                            .map_err(session_ledger_error)?;
                        tokens.extend_from_slice(&accepted);
                        if ledger.emitted_tokens() != tokens.as_slice() {
                            return Err(EngineError::Generation(
                                "batched dSpark ledger/output sequence diverged".to_owned(),
                            ));
                        }
                        dspark_frontier = Some(next_frontier);
                    } else {
                        current_taps = kept_taps;
                        tokens.extend_from_slice(&accepted);
                    }
                    last_token = i32::try_from(*accepted.last().ok_or_else(|| {
                        EngineError::Generation("accept_prefix returned empty vec".to_owned())
                    })?)
                    .map_err(|_| {
                        EngineError::Generation("accepted token overflow i32".to_owned())
                    })?;
                    start = start.checked_add(n_accepted).ok_or_else(|| {
                        EngineError::Generation("DFlash target boundary overflow".to_owned())
                    })?;
                    n_accepted
                };

                // Adapt the next round's block size by block UTILIZATION
                // (n_accepted / block_size) — the entropy proxy. On low-entropy text
                // acceptance scales with the block (util stays high → grow toward
                // block_max for the biggest win); on high-entropy text acceptance
                // plateaus (util drops → shrink so the verify isn't wasted, and the
                // S>1 verify stays off near-ties). Gating on util, not raw accept,
                // keeps big blocks where they pay off. (all block_size uses above.)
                if adaptive {
                    let util = f64::from(n_accepted) / f64::from(block_size);
                    ema_util = 0.5f64.mul_add(ema_util, 0.5 * util);
                    block_size = if ema_util >= 0.75 {
                        (block_size * 2).min(block_max) // sustained high → grow
                    } else if ema_util <= 0.5 {
                        (block_size / 2).max(block_min) // sustained low → shrink
                    } else {
                        block_size
                    };
                }

                // Realized-speedup gate: floor to AR next round if spec has become
                // slower than plain AR (or to calibrate T_ar if we have no sample).
                if gate {
                    let step_wall = round_t0.elapsed().as_secs_f64().max(1e-9);
                    if spec_warmup_left > 0 {
                        // Cold spec warm-up: stay in spec, don't judge or floor,
                        // don't poison ratio_ema with the cold step_wall / low
                        // cold-cache accept. The kernels warm and the drafter
                        // cache fills over these rounds.
                        spec_warmup_left -= 1;
                    } else if let Some(t_ar) = t_ar_ema {
                        let ratio = f64::from(n_accepted) * t_ar / step_wall;
                        ratio_ema = 0.6f64.mul_add(ratio_ema, 0.4 * ratio);
                        if std::env::var("HIGGS_DFLASH_TRACE").is_ok() && rounds <= 8 {
                            tracing::info!(
                                round = rounds,
                                n_accepted,
                                t_ar_ms = format!("{:.1}", t_ar * 1e3),
                                step_wall_ms = format!("{:.1}", step_wall * 1e3),
                                ratio = format!("{ratio:.2}"),
                                ratio_ema = format!("{ratio_ema:.2}"),
                                "GATE"
                            );
                        }
                        if ratio_ema < 1.0 {
                            // Losing probe: floor and back off so the next probe
                            // amortizes over a longer AR run (sustained prose).
                            in_ar = true;
                            ar_run = 0;
                            probe_every = probe_every.saturating_mul(2).min(AR_PROBE_MAX);
                        } else {
                            // Winning: stay in spec, but reset the cadence so a
                            // later degradation is re-probed promptly, not lazily.
                            probe_every = AR_PROBE_BASE;
                            // Refresh t_ar periodically so a long winning streak
                            // can't run on a thermally-stale AR reference.
                            spec_since_recal += 1;
                            if spec_since_recal >= T_AR_RECAL_EVERY {
                                in_ar = true;
                                ar_run = 0;
                                recal = true;
                                spec_since_recal = 0;
                            }
                        }
                    } else {
                        // No T_ar sample yet (initial calibration) — floor at base
                        // cadence; this isn't a losing probe, so don't back off.
                        in_ar = true;
                        ar_run = 0;
                        probe_every = AR_PROBE_BASE;
                    }
                }
            } // end spec-round branch

            // j. Deliver this round's tokens and check termination via the sink.
            let completion_len = Self::completion_len(&tokens)?;
            let eos = tokens.iter().any(|t| self.eos_token_ids.contains(t));
            let forced = if eos {
                Some("stop")
            } else if completion_len >= max_tokens {
                Some("length")
            } else {
                None
            };
            let stop_hit = sink.emit(
                self,
                &tokens,
                stop_sequences,
                prompt_len,
                completion_len,
                forced,
            )?;
            if forced.is_some() || stop_hit {
                let accept_len = if rounds > 0 {
                    total_accepted as f64 / rounds as f64
                } else {
                    0.0
                };
                let decode_elapsed = t_start.elapsed();
                let request_elapsed = request_t0.elapsed();
                let secs = decode_elapsed.as_secs_f64();
                let tok_per_sec = if secs > 0.0 {
                    total_accepted as f64 / secs
                } else {
                    0.0
                };
                // `stop_hit` without a forced reason means a stop sequence fired.
                let finish_reason = forced.unwrap_or("stop");
                let decode_tokens = u32::try_from(total_accepted).unwrap_or(u32::MAX);
                if let Ok(mut timing) = self.last_dflash_timing.lock() {
                    *timing = Some(GenerationPhaseTiming {
                        prefill: prefill_elapsed,
                        decode: decode_elapsed,
                        request: request_elapsed,
                        decode_tokens,
                    });
                }
                tracing::info!(
                    tokens = tokens.len(),
                    accept_len = format!("{accept_len:.2}"),
                    spec_rounds = rounds,
                    probe_every = probe_every,
                    decode_tokens = total_accepted,
                    tok_per_sec = format!("{tok_per_sec:.1}"),
                    prefill_ms = format!("{:.1}", prefill_elapsed.as_secs_f64() * 1e3),
                    decode_ms = format!("{:.1}", secs * 1e3),
                    request_ms = format!("{:.1}", request_elapsed.as_secs_f64() * 1e3),
                    finish_reason,
                    "DFlash generation complete"
                );
                return sink.finish(self, &tokens, finish_reason, prompt_len, completion_len);
            }
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    fn log_decode_timing(
        steps: u32,
        forward_ns: u128,
        eval_ns: u128,
        item_ns: u128,
        other_ns: u128,
    ) {
        if steps > 0 {
            let s = f64::from(steps);
            tracing::info!(
                steps,
                forward_ms = format!("{:.2}", forward_ns as f64 / s / 1e6),
                eval_ms = format!("{:.2}", eval_ns as f64 / s / 1e6),
                item_ms = format!("{:.2}", item_ns as f64 / s / 1e6),
                other_ms = format!("{:.2}", other_ns as f64 / s / 1e6),
                total_ms = format!(
                    "{:.2}",
                    (forward_ns + eval_ns + item_ns + other_ns) as f64 / s / 1e6
                ),
                "Decode loop timing (per step avg)"
            );
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn record_ar_generation_timing(
        &self,
        prefill: std::time::Duration,
        decode: std::time::Duration,
        request: std::time::Duration,
        completion_tokens: u32,
        finish_reason: &str,
    ) {
        let decode_tokens = completion_tokens.saturating_sub(1);
        let tok_per_sec = if decode.is_zero() {
            0.0
        } else {
            f64::from(decode_tokens) / decode.as_secs_f64()
        };
        if let Ok(mut timing) = self.last_ar_timing.lock() {
            *timing = Some(GenerationPhaseTiming {
                prefill,
                decode,
                request,
                decode_tokens,
            });
        }
        tracing::info!(
            completion_tokens,
            decode_tokens,
            tok_per_sec = format!("{tok_per_sec:.1}"),
            prefill_ms = format!("{:.1}", prefill.as_secs_f64() * 1e3),
            decode_ms = format!("{:.1}", decode.as_secs_f64() * 1e3),
            request_ms = format!("{:.1}", request.as_secs_f64() * 1e3),
            finish_reason,
            "AR generation complete"
        );
    }

    #[allow(clippy::cast_precision_loss)]
    fn log_mtp_decode_stats(
        stats: &crate::mtp::MtpStats,
        elapsed: std::time::Duration,
        reason: &str,
    ) {
        tracing::info!(
            reason,
            cycles = stats.cycles(),
            drafted = stats.drafted(),
            accepted_drafts = stats.accepted_drafts(),
            emitted = stats.emitted(),
            accept_rate = format!("{:.1}%", stats.acceptance_rate_percent()),
            tok_per_s = format!("{:.1}", f64::from(stats.emitted()) / elapsed.as_secs_f64()),
            "MTP decode complete"
        );
    }

    #[allow(clippy::cast_precision_loss)]
    fn log_prompt_lookup_decode_stats(
        stats: &crate::mtp::MtpStats,
        elapsed: std::time::Duration,
        reason: &str,
    ) {
        tracing::info!(
            reason,
            cycles = stats.cycles(),
            drafted = stats.drafted(),
            accepted_drafts = stats.accepted_drafts(),
            emitted = stats.emitted(),
            accept_rate = format!("{:.1}%", stats.acceptance_rate_percent()),
            tok_per_s = format!("{:.1}", f64::from(stats.emitted()) / elapsed.as_secs_f64()),
            "Prompt-lookup decode complete"
        );
    }

    /// Architecture-neutral prompt-lookup speculative decode loop.
    ///
    /// The draft provider copies likely next tokens from repeated prompt/history
    /// spans, then verifies the whole candidate window in one model pass.
    #[allow(
        clippy::too_many_arguments,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn prompt_lookup_generate(
        &self,
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        first_token_id: u32,
        max_tokens: u32,
        prompt_len: u32,
        tokens: &mut Vec<u32>,
        stop_sequences: &[String],
        enable_thinking: bool,
        thinking_budget: u32,
    ) -> Result<GenerationOutput, EngineError> {
        let has_stop_sequences = !stop_sequences.is_empty();
        let unchecked_lookup = unchecked_prompt_lookup_enabled();
        let first_token_i32 = i32::try_from(first_token_id)
            .map_err(|_| EngineError::Generation("token id exceeds i32 range".to_owned()))?;
        let first_input = Array::from_slice(&[first_token_i32], &[1, 1]);
        let logits = model
            .forward_all_logits(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next_arr =
            mlx_rs::argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        eval([&next_arr]).map_err(EngineError::Mlx)?;

        let mut confirmed_token_id: u32 = next_arr.item();
        let base_config = prompt_lookup_config();
        let mut stats = crate::mtp::MtpStats::default();
        let t_start = std::time::Instant::now();

        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let completion_len = Self::completion_len(tokens)?;
            if completion_len >= max_tokens {
                let elapsed = t_start.elapsed();
                Self::log_prompt_lookup_decode_stats(&stats, elapsed, "length");
                return Ok(GenerationOutput {
                    text: self.decode_tokens(tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: None,
                });
            }

            let remaining = usize::try_from(max_tokens.saturating_sub(completion_len))
                .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
            let config = crate::mtp::PromptLookupConfig {
                max_drafts: base_config.max_drafts.min(remaining.saturating_sub(1)),
                ..base_config
            };

            let result = if unchecked_lookup {
                crate::mtp::unchecked_prompt_lookup_cycle(
                    model,
                    cache,
                    tokens,
                    confirmed_token_id,
                    config,
                )?
            } else {
                crate::mtp::prompt_lookup_cycle(model, cache, tokens, confirmed_token_id, config)?
            };
            stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);

            for &tok in &result.tokens {
                if let Some(close_id) = think_close_token {
                    if !seen_think_close {
                        if tok == close_id {
                            seen_think_close = true;
                        } else {
                            thinking_tokens += 1;
                            if thinking_tokens >= thinking_budget {
                                tokens.push(close_id);
                                seen_think_close = true;
                                tracing::info!(
                                    budget = thinking_budget,
                                    "Prompt lookup: thinking budget reached, forcing </think>"
                                );
                                if self.eos_token_ids.contains(&close_id) {
                                    let elapsed = t_start.elapsed();
                                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "stop");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "stop".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: Self::completion_len(tokens)?,
                                        token_logprobs: None,
                                    });
                                }

                                if has_stop_sequences {
                                    let text = self.decode_tokens(tokens)?;
                                    if let Some(truncated) =
                                        check_stop_sequences(&text, stop_sequences)
                                    {
                                        let elapsed = t_start.elapsed();
                                        Self::log_prompt_lookup_decode_stats(
                                            &stats, elapsed, "stop",
                                        );
                                        return Ok(GenerationOutput {
                                            text: truncated,
                                            finish_reason: "stop".to_owned(),
                                            prompt_tokens: prompt_len,
                                            completion_tokens: Self::completion_len(tokens)?,
                                            token_logprobs: None,
                                        });
                                    }
                                }

                                let forced_completion_len = Self::completion_len(tokens)?;
                                if forced_completion_len >= max_tokens {
                                    let elapsed = t_start.elapsed();
                                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "length");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "length".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: forced_completion_len,
                                        token_logprobs: None,
                                    });
                                }
                                break;
                            }
                        }
                    }
                }

                tokens.push(tok);

                if self.eos_token_ids.contains(&tok) {
                    let elapsed = t_start.elapsed();
                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: self.decode_tokens(tokens)?,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            if has_stop_sequences {
                let text = self.decode_tokens(tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    let elapsed = t_start.elapsed();
                    Self::log_prompt_lookup_decode_stats(&stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            let final_completion_len = Self::completion_len(tokens)?;
            if final_completion_len >= max_tokens {
                let elapsed = t_start.elapsed();
                Self::log_prompt_lookup_decode_stats(&stats, elapsed, "length");
                return Ok(GenerationOutput {
                    text: self.decode_tokens(tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: final_completion_len,
                    token_logprobs: None,
                });
            }

            confirmed_token_id = result.next_token_id;
        }
    }

    /// MTP speculative decode loop.
    ///
    /// Runs the backbone to get the initial hidden state, then loops calling
    /// `mtp_cycle()` which drafts multiple tokens per cycle for speculative speedup.
    #[allow(
        clippy::too_many_arguments,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn mtp_generate(
        &self,
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        actual_prompt_tokens: &[u32],
        prefill_hidden: Option<&Array>,
        first_token_id: u32,
        max_tokens: u32,
        prompt_len: u32,
        tokens: &mut Vec<u32>,
        stop_sequences: &[String],
        enable_thinking: bool,
        thinking_budget: u32,
    ) -> Result<GenerationOutput, EngineError> {
        let has_stop_sequences = !stop_sequences.is_empty();

        // Create MTP cache for the MTP head's attention layer(s).
        let mut mtp_cache = model
            .make_mtp_cache()
            .ok_or_else(|| EngineError::Generation("MTP cache creation failed".into()))?;
        if let Some(hidden) = prefill_hidden {
            crate::mtp::prime_mtp_cache(model, &mut mtp_cache, actual_prompt_tokens, hidden)?;
        }

        let first_input = Array::from_slice(&[first_token_id as i32], &[1, 1]);
        let (hidden, logits) = model
            .forward_with_hidden(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next_arr =
            mlx_rs::argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        let h = hidden.index((.., -1.., ..));
        if let Some(previous_hidden) = prefill_hidden
            .filter(|_| !actual_prompt_tokens.is_empty())
            .map(|prefill| Self::hidden_row_from_sequence(prefill, actual_prompt_tokens.len() - 1))
            .transpose()?
        {
            crate::mtp::mirror_mtp_token(model, &mut mtp_cache, &previous_hidden, first_token_id)?;
        }
        eval([&next_arr, &h]).map_err(EngineError::Mlx)?;

        let mut current_hidden = h;
        let mut confirmed_token_id: u32 = next_arr.item();
        let mut mtp_stats = crate::mtp::MtpStats::default();
        let mut adaptive_depth = mtp_adaptive_draft_enabled()
            .then(|| adaptive_draft_depth_for_cap(self.tuning.mtp_draft_n_max()));
        let hybrid_prompt_lookup = mtp_prompt_lookup_enabled();
        let hybrid_prompt_lookup_config = prompt_lookup_config();
        let t_start = std::time::Instant::now();

        // Thinking budget: force </think> after N tokens if model hasn't closed it.
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        // Seed thinking state from the first token (already emitted by caller).
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let cycle_completion_len = Self::completion_len(tokens)?;
            let remaining = usize::try_from(max_tokens.saturating_sub(cycle_completion_len))
                .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
            let draft_depth = adaptive_depth
                .as_ref()
                .map_or_else(
                    || self.tuning.mtp_draft_n_max(),
                    crate::mtp::AdaptiveDraftDepth::current,
                )
                .min(remaining.saturating_sub(1).max(1));
            let prompt_config = crate::mtp::PromptLookupConfig {
                max_drafts: hybrid_prompt_lookup_config
                    .max_drafts
                    .min(remaining.saturating_sub(1)),
                ..hybrid_prompt_lookup_config
            };
            let result = if hybrid_prompt_lookup && prompt_config.max_drafts > 0 {
                crate::mtp::mtp_prompt_lookup_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    tokens,
                    confirmed_token_id,
                    prompt_config,
                )?
                .map_or_else(
                    || {
                        crate::mtp::mtp_cycle(
                            model,
                            cache,
                            &mut mtp_cache,
                            &current_hidden,
                            confirmed_token_id,
                            draft_depth,
                        )
                    },
                    Ok,
                )?
            } else {
                crate::mtp::mtp_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    confirmed_token_id,
                    draft_depth,
                )?
            };

            mtp_stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
            if let Some(depth) = &mut adaptive_depth {
                depth.observe(result.accepted_drafts, result.drafted);
            }

            for &tok in &result.tokens {
                // Thinking budget enforcement
                if let Some(close_id) = think_close_token {
                    if !seen_think_close {
                        if tok == close_id {
                            seen_think_close = true;
                        } else {
                            thinking_tokens += 1;
                            if thinking_tokens >= thinking_budget {
                                tokens.push(close_id);
                                seen_think_close = true;
                                tracing::info!(
                                    budget = thinking_budget,
                                    "MTP: thinking budget reached, forcing </think>"
                                );
                                if self.eos_token_ids.contains(&close_id) {
                                    let elapsed = t_start.elapsed();
                                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "stop".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: Self::completion_len(tokens)?,
                                        token_logprobs: None,
                                    });
                                }

                                if has_stop_sequences {
                                    let text = self.decode_tokens(tokens)?;
                                    if let Some(truncated) =
                                        check_stop_sequences(&text, stop_sequences)
                                    {
                                        let elapsed = t_start.elapsed();
                                        Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                                        return Ok(GenerationOutput {
                                            text: truncated,
                                            finish_reason: "stop".to_owned(),
                                            prompt_tokens: prompt_len,
                                            completion_tokens: Self::completion_len(tokens)?,
                                            token_logprobs: None,
                                        });
                                    }
                                }

                                let completion_len = Self::completion_len(tokens)?;
                                if completion_len >= max_tokens {
                                    let elapsed = t_start.elapsed();
                                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "length");
                                    return Ok(GenerationOutput {
                                        text: self.decode_tokens(tokens)?,
                                        finish_reason: "length".to_owned(),
                                        prompt_tokens: prompt_len,
                                        completion_tokens: completion_len,
                                        token_logprobs: None,
                                    });
                                }

                                // Skip remaining tokens from this cycle
                                break;
                            }
                        }
                    }
                }

                tokens.push(tok);

                if self.eos_token_ids.contains(&tok) {
                    let elapsed = t_start.elapsed();
                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: self.decode_tokens(tokens)?,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            if has_stop_sequences {
                let text = self.decode_tokens(tokens)?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    let elapsed = t_start.elapsed();
                    Self::log_mtp_decode_stats(&mtp_stats, elapsed, "stop");
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: Self::completion_len(tokens)?,
                        token_logprobs: None,
                    });
                }
            }

            let completion_len = Self::completion_len(tokens)?;
            if completion_len >= max_tokens {
                let elapsed = t_start.elapsed();
                Self::log_mtp_decode_stats(&mtp_stats, elapsed, "length");
                return Ok(GenerationOutput {
                    text: self.decode_tokens(tokens)?,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprobs: None,
                });
            }

            current_hidden = result.hidden;
            confirmed_token_id = result.next_token_id;
        }
    }

    /// MTP speculative decode loop — streaming variant.
    ///
    /// Same logic as `mtp_generate`, but sends each accepted token (or pair)
    /// via the streaming channel instead of accumulating into a buffer.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn mtp_generate_streaming(
        &self,
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        actual_prompt_tokens: &[u32],
        prefill_hidden: Option<&Array>,
        first_token_id: u32,
        max_tokens: u32,
        prompt_len: u32,
        tokens: &mut Vec<u32>,
        stop_sequences: &[String],
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        mut detok: IncrementalDetok,
        enable_thinking: bool,
        thinking_budget: u32,
    ) -> Result<(), EngineError> {
        let has_stop_sequences = !stop_sequences.is_empty();

        let mut mtp_cache = model
            .make_mtp_cache()
            .ok_or_else(|| EngineError::Generation("MTP cache creation failed".into()))?;
        if let Some(hidden) = prefill_hidden {
            crate::mtp::prime_mtp_cache(model, &mut mtp_cache, actual_prompt_tokens, hidden)?;
        }

        let first_input = Array::from_slice(&[first_token_id as i32], &[1, 1]);
        let (hidden, logits) = model
            .forward_with_hidden(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next_arr =
            mlx_rs::argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        let h = hidden.index((.., -1.., ..));
        if let Some(previous_hidden) = prefill_hidden
            .filter(|_| !actual_prompt_tokens.is_empty())
            .map(|prefill| Self::hidden_row_from_sequence(prefill, actual_prompt_tokens.len() - 1))
            .transpose()?
        {
            crate::mtp::mirror_mtp_token(model, &mut mtp_cache, &previous_hidden, first_token_id)?;
        }
        eval([&next_arr, &h]).map_err(EngineError::Mlx)?;

        let mut current_hidden = h;
        let mut confirmed_token_id: u32 = next_arr.item();
        let mut mtp_stats = crate::mtp::MtpStats::default();
        let mut adaptive_depth = mtp_adaptive_draft_enabled()
            .then(|| adaptive_draft_depth_for_cap(self.tuning.mtp_draft_n_max()));
        let hybrid_prompt_lookup = mtp_prompt_lookup_enabled();
        let hybrid_prompt_lookup_config = prompt_lookup_config();
        let t_start = std::time::Instant::now();

        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        loop {
            let cycle_completion_len = Self::completion_len(tokens)?;
            let remaining = usize::try_from(max_tokens.saturating_sub(cycle_completion_len))
                .map_err(|_| EngineError::Generation("max_tokens overflow".to_owned()))?;
            let draft_depth = adaptive_depth
                .as_ref()
                .map_or_else(
                    || self.tuning.mtp_draft_n_max(),
                    crate::mtp::AdaptiveDraftDepth::current,
                )
                .min(remaining.saturating_sub(1).max(1));
            let prompt_config = crate::mtp::PromptLookupConfig {
                max_drafts: hybrid_prompt_lookup_config
                    .max_drafts
                    .min(remaining.saturating_sub(1)),
                ..hybrid_prompt_lookup_config
            };
            let result = if hybrid_prompt_lookup && prompt_config.max_drafts > 0 {
                crate::mtp::mtp_prompt_lookup_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    tokens,
                    confirmed_token_id,
                    prompt_config,
                )?
                .map_or_else(
                    || {
                        crate::mtp::mtp_cycle(
                            model,
                            cache,
                            &mut mtp_cache,
                            &current_hidden,
                            confirmed_token_id,
                            draft_depth,
                        )
                    },
                    Ok,
                )?
            } else {
                crate::mtp::mtp_cycle(
                    model,
                    cache,
                    &mut mtp_cache,
                    &current_hidden,
                    confirmed_token_id,
                    draft_depth,
                )?
            };

            mtp_stats.record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
            if let Some(depth) = &mut adaptive_depth {
                depth.observe(result.accepted_drafts, result.drafted);
            }

            for &tok in &result.tokens {
                // Thinking budget enforcement
                if let Some(close_id) = think_close_token {
                    if !seen_think_close {
                        if tok == close_id {
                            seen_think_close = true;
                        } else {
                            thinking_tokens += 1;
                            if thinking_tokens >= thinking_budget {
                                tokens.push(close_id);
                                seen_think_close = true;
                                tracing::info!(
                                    budget = thinking_budget,
                                    "MTP streaming: thinking budget reached, forcing </think>"
                                );

                                let is_eos = self.eos_token_ids.contains(&close_id);
                                let completion_len = Self::completion_len(tokens)?;
                                let is_max = completion_len >= max_tokens;

                                let new_text = detok.append(&self.tokenizer, tokens)?;
                                let emitted_before = detok.text.len() - new_text.len();
                                let (mut final_new_text, hit_stop_seq) = if has_stop_sequences {
                                    find_stop_in_tail(&detok.text, new_text.len(), stop_sequences)
                                        .map_or((new_text, false), |pos| {
                                            let emit = detok
                                                .text
                                                .get(emitted_before..pos)
                                                .unwrap_or_default()
                                                .to_owned();
                                            (emit, true)
                                        })
                                } else {
                                    (new_text, false)
                                };
                                let step_finished = is_eos || is_max || hit_stop_seq;
                                if step_finished && !hit_stop_seq {
                                    final_new_text.push_str(&detok.flush(&self.tokenizer, tokens)?);
                                }
                                let finish_reason = if is_eos || hit_stop_seq {
                                    Some("stop".to_owned())
                                } else if is_max {
                                    Some("length".to_owned())
                                } else {
                                    None
                                };

                                if step_finished {
                                    let elapsed = t_start.elapsed();
                                    Self::log_mtp_decode_stats(
                                        &mtp_stats,
                                        elapsed,
                                        finish_reason.as_deref().unwrap_or("client"),
                                    );
                                }

                                if sender
                                    .blocking_send(StreamingOutput {
                                        new_text: final_new_text,
                                        finished: step_finished,
                                        finish_reason,
                                        prompt_tokens: prompt_len,
                                        completion_tokens: completion_len,
                                        token_logprob: None,
                                        prefill_progress: None,
                                    })
                                    .is_err()
                                {
                                    return Ok(());
                                }

                                if step_finished {
                                    return Ok(());
                                }

                                break;
                            }
                        }
                    }
                }

                tokens.push(tok);

                let is_eos = self.eos_token_ids.contains(&tok);
                let completion_len = Self::completion_len(tokens)?;
                let is_max = completion_len >= max_tokens;

                let new_text = detok.append(&self.tokenizer, tokens)?;
                let emitted_before = detok.text.len() - new_text.len();

                let (mut final_new_text, hit_stop_seq) = if has_stop_sequences {
                    find_stop_in_tail(&detok.text, new_text.len(), stop_sequences).map_or(
                        (new_text, false),
                        |pos| {
                            let emit = detok
                                .text
                                .get(emitted_before..pos)
                                .unwrap_or_default()
                                .to_owned();
                            (emit, true)
                        },
                    )
                } else {
                    (new_text, false)
                };

                let step_finished = is_eos || is_max || hit_stop_seq;
                if step_finished && !hit_stop_seq {
                    final_new_text.push_str(&detok.flush(&self.tokenizer, tokens)?);
                }
                let finish_reason = if is_eos || hit_stop_seq {
                    Some("stop".to_owned())
                } else if is_max {
                    Some("length".to_owned())
                } else {
                    None
                };

                if step_finished {
                    let elapsed = t_start.elapsed();
                    Self::log_mtp_decode_stats(
                        &mtp_stats,
                        elapsed,
                        finish_reason.as_deref().unwrap_or("client"),
                    );
                }

                if sender
                    .blocking_send(StreamingOutput {
                        new_text: final_new_text,
                        finished: step_finished,
                        finish_reason,
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                        token_logprob: None,
                        prefill_progress: None,
                    })
                    .is_err()
                {
                    return Ok(());
                }

                if step_finished {
                    return Ok(());
                }
            }

            current_hidden = result.hidden;
            confirmed_token_id = result.next_token_id;
        }
    }

    /// Generate tokens one at a time, sending each via the provided channel.
    ///
    /// If the receiver is dropped (client disconnected), generation stops early.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::significant_drop_tightening
    )]
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            sender,
            self.enable_thinking,
            // Non-thinking convenience entry (used by /v1/completions) never
            // streams prefill progress.
            false,
            constraint,
            pixel_values,
            checkpoint_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        return_progress: bool,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            let prompt_len = Self::prompt_len(prompt_tokens)?;
            let _ = sender.blocking_send(StreamingOutput {
                new_text: String::new(),
                finished: true,
                finish_reason: Some("length".to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprob: None,
                prefill_progress: None,
            });
            return Ok(());
        }

        with_new_default_stream(Stream::new(), || {
            self.generate_streaming_inner(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                pixel_values,
                checkpoint_id,
            )
        })
    }

    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::significant_drop_tightening
    )]
    fn generate_streaming_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        return_progress: bool,
        mut constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
        checkpoint_id: Option<&str>,
    ) -> Result<(), EngineError> {
        let thinking_budget = params.thinking_budget.unwrap_or(DEFAULT_THINKING_BUDGET);
        // DFlash streaming: branch BEFORE the normal prefill — the DFlash loop
        // runs its own tap-prefill, so dispatching here (mirroring the
        // non-streaming site) avoids a double prefill. Logprobs requests fall
        // through to the MTP/AR path, which produces per-token logprobs.
        let sampled_dspark = self.sampled_dspark_requires_ar(params);
        if sampled_dspark {
            tracing::debug!(
                "sampled streaming dSpark request uses ordinary AR to preserve the shared RNG transition"
            );
        }
        let speculation_route = resolve_speculation_route(
            params.speculation,
            self.dflash.is_some()
                && !sampled_dspark
                && constraint.is_none()
                && pixel_values.is_none()
                && !logprobs,
            stateless_mtp_family_eligible(params, constraint.is_some(), logprobs),
        );
        if speculation_route == SpeculationRoute::DFlash {
            return self.generate_dflash_streaming(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                sender,
                enable_thinking,
                thinking_budget,
            );
        }

        let logprob_top_n = logprobs.then(|| top_logprobs.unwrap_or(0));

        let mut prepared = self.prepare_generation(prompt_tokens, pixel_values, checkpoint_id)?;
        let prompt_len = prepared.prompt_len;
        #[allow(clippy::float_cmp)]
        let capture_mtp_prefill = mtp_prefill_priming_enabled()
            && self.tuning.enable_mtp()
            && prepared.model.has_mtp()
            && prepared.pixel_values.is_none()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0;

        // Stream prefill progress only when the caller opted in (OpenAI
        // `return_progress: true`). Direct higgs-engine callers and BatchEngine
        // keep the progress-free streaming contract: no sink, no extra events.
        //
        // The chunked-prefill loops report suffix-relative completions through a
        // thread-local sink; map them to absolute prompt position by adding the
        // prefix-cache hit. Events ride the normal streaming channel as
        // progress-only outputs. try_send: never stall prefill on a slow
        // consumer — dropped progress events are harmless. The returned guard is
        // held for the prefill's duration and uninstalls the sink on drop.
        let prefill_sink = return_progress.then(|| {
            let actual_tokens =
                u32::try_from(prepared.actual_prompt_tokens.len()).unwrap_or(u32::MAX);
            let cached = prompt_len.saturating_sub(actual_tokens);
            let make_progress_output = move |suffix_done: u32| StreamingOutput {
                new_text: String::new(),
                finished: false,
                finish_reason: None,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprob: None,
                prefill_progress: Some(crate::engine::PrefillProgress {
                    processed: (cached + suffix_done).min(prompt_len),
                    cached,
                    total: prompt_len,
                }),
            };
            // Initial event: tells the client the total (and cache hit) before
            // the first ~1024-token chunk completes.
            let _ = sender.try_send(make_progress_output(0));
            let progress_sender = sender.clone();
            higgs_models::progress::install_prefill_progress_sink(Box::new(move |done, _total| {
                let _ = progress_sender.try_send(make_progress_output(
                    u32::try_from(done.max(0)).unwrap_or(0),
                ));
            }))
        });

        let (current_token, first_logprob_data, prefill_hidden) = self.run_prefill(
            prompt_tokens,
            self.gen_prompt_suffixes.for_request(enable_thinking),
            &mut prepared,
            params,
            logprob_top_n,
            constraint.as_ref(),
            capture_mtp_prefill,
            prefix_cache_enabled(),
            checkpoint_id,
            None,
            None,
        )?;
        // Prefill done — decode must not report progress. Dropping the Option
        // uninstalls the sink when present; a no-op when progress was off.
        drop(prefill_sink);

        let mut all_tokens: Vec<u32> = Vec::new();
        let first_token_id: u32 = current_token.item();
        // Advance the constraint past the first sampled token before decode.
        if let Some(ref mut cg) = constraint {
            cg.advance(first_token_id);
        }
        all_tokens.push(first_token_id);

        let mut detok = IncrementalDetok::new(
            String::new(),
            0,
            std::sync::Arc::clone(&self.decode_skip_ids),
        );
        let first_new_text = detok.append(&self.tokenizer, &all_tokens)?;
        let first_emitted_before = detok.text.len() - first_new_text.len();
        let (mut first_text, first_hit_stop) = if stop_sequences.is_empty() {
            (first_new_text, false)
        } else {
            find_stop_in_tail(&detok.text, first_new_text.len(), stop_sequences).map_or(
                (first_new_text, false),
                |pos| {
                    let emit = detok
                        .text
                        .get(first_emitted_before..pos)
                        .unwrap_or_default()
                        .to_owned();
                    (emit, true)
                },
            )
        };

        let first_is_eos = self.eos_token_ids.contains(&first_token_id);
        let finished = first_is_eos || first_hit_stop || 1 >= max_tokens;

        if finished && !first_hit_stop {
            first_text.push_str(&detok.flush(&self.tokenizer, &all_tokens)?);
        }

        let first_logprob = first_logprob_data
            .as_ref()
            .map(|lp| lp.materialize(first_token_id));

        if sender
            .blocking_send(StreamingOutput {
                new_text: first_text,
                finished,
                finish_reason: if first_is_eos || first_hit_stop {
                    Some("stop".to_owned())
                } else if 1 >= max_tokens {
                    Some("length".to_owned())
                } else {
                    None
                },
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: first_logprob,
                prefill_progress: None,
            })
            .is_err()
        {
            return Ok(());
        }

        if finished {
            return Ok(());
        }

        // MTP speculative decode (streaming): greedy, no constraints, no logprobs.
        #[allow(clippy::float_cmp)]
        if self.tuning.enable_mtp()
            && speculation_route == SpeculationRoute::MtpFamily
            && prepared.model.has_mtp()
            && constraint.is_none()
            && !logprobs
            && params.temperature == 0.0
        {
            let actual_prompt_tokens = prepared.actual_prompt_tokens.clone();
            return self.mtp_generate_streaming(
                &mut prepared.model,
                &mut prepared.cache,
                &actual_prompt_tokens,
                prefill_hidden.as_ref(),
                first_token_id,
                max_tokens,
                prompt_len,
                &mut all_tokens,
                stop_sequences,
                sender,
                detok,
                enable_thinking,
                thinking_budget,
            );
        }

        // Thinking budget (streaming): force </think> after N tokens.
        let think_close_token = if enable_thinking {
            self.think_close_token
        } else {
            None
        };
        // Seed thinking state from the first token (already emitted above).
        let mut thinking_tokens: u32 = u32::from(think_close_token.is_some());
        let mut seen_think_close =
            think_close_token.is_some_and(|close_id| first_token_id == close_id);

        // Pipelined decode loop: build step N+2 while GPU computes step N+1
        let (mut next_token, mut next_logprob_data) = Self::decode_step(
            &current_token,
            &mut prepared.model,
            &mut prepared.cache,
            params,
            &all_tokens,
            logprob_top_n,
            constraint.as_ref(),
        )?;
        {
            let mut eval_targets: Vec<&Array> = vec![&next_token];
            if let Some(ref lp) = next_logprob_data {
                eval_targets.extend(lp.eval_targets());
            }
            async_eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        loop {
            // When constrained, extract the sampled token and advance the FSM
            // before decode_step, so the mask is always applied at the correct
            // FSM state (mirrors the non-streaming pattern in generate_inner).
            let constrained_token_id: Option<u32> = constraint.is_some().then(|| {
                let id: u32 = next_token.item();
                if let Some(ref mut cg) = constraint {
                    cg.advance(id);
                }
                id
            });

            let (following, following_logprob_data) = Self::decode_step(
                &next_token,
                &mut prepared.model,
                &mut prepared.cache,
                params,
                &all_tokens,
                logprob_top_n,
                constraint.as_ref(),
            )?;
            {
                let mut eval_targets: Vec<&Array> = vec![&following];
                if let Some(ref lp) = following_logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
                if constraint.is_some() {
                    eval(eval_targets).map_err(EngineError::Mlx)?;
                } else {
                    async_eval(eval_targets).map_err(EngineError::Mlx)?;
                }
            }

            let mut token_id: u32 = constrained_token_id.unwrap_or_else(|| next_token.item());

            // Thinking budget: force </think> after N tokens if model hasn't closed it.
            // NOTE: same KV-cache discontinuity caveat as the non-streaming path.
            if let Some(close_id) = think_close_token {
                if !seen_think_close {
                    if token_id == close_id {
                        seen_think_close = true;
                    } else {
                        thinking_tokens += 1;
                        if thinking_tokens >= thinking_budget {
                            token_id = close_id;
                            seen_think_close = true;
                            tracing::info!(
                                budget = thinking_budget,
                                "Thinking budget reached, forcing </think>"
                            );
                        }
                    }
                }
            }

            let token_logprob = next_logprob_data
                .as_ref()
                .map(|lp_data| lp_data.materialize(token_id));

            all_tokens.push(token_id);

            let completion_len = Self::completion_len(&all_tokens)?;

            let new_text = detok.append(&self.tokenizer, &all_tokens)?;
            let emitted_before = detok.text.len() - new_text.len();

            let (mut final_new_text, hit_stop_seq) = if stop_sequences.is_empty() {
                (new_text, false)
            } else {
                find_stop_in_tail(&detok.text, new_text.len(), stop_sequences).map_or(
                    (new_text, false),
                    |pos| {
                        let emit = detok
                            .text
                            .get(emitted_before..pos)
                            .unwrap_or_default()
                            .to_owned();
                        (emit, true)
                    },
                )
            };

            let is_eos = self.eos_token_ids.contains(&token_id);
            let is_max = completion_len >= max_tokens;
            let constraint_done = constraint
                .as_ref()
                .is_some_and(crate::constrained::ConstrainedGenerator::is_finished);
            let step_finished = is_eos || is_max || hit_stop_seq || constraint_done;

            if step_finished && !hit_stop_seq {
                final_new_text.push_str(&detok.flush(&self.tokenizer, &all_tokens)?);
            }

            let finish_reason = if is_eos || hit_stop_seq || constraint_done {
                Some("stop".to_owned())
            } else if is_max {
                Some("length".to_owned())
            } else {
                None
            };

            if sender
                .blocking_send(StreamingOutput {
                    new_text: final_new_text,
                    finished: step_finished,
                    finish_reason,
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                    token_logprob,
                    prefill_progress: None,
                })
                .is_err()
            {
                return Ok(());
            }

            if step_finished {
                break;
            }

            // If thinking budget was just reached, override the pipelined token
            // so the next decode step gets </think> as input.
            if seen_think_close && thinking_tokens == thinking_budget {
                if let Some(close_id) = think_close_token {
                    next_token = Array::from_slice(&[close_id], &[1]);
                }
                thinking_tokens += 1; // prevent re-triggering
            } else {
                next_token = following;
            }
            next_logprob_data = following_logprob_data;
        }

        Ok(())
    }
}

/// Cap on the trailing token window re-decoded per streaming step.
/// Generous for any multi-token UTF-8 sequence; prevents a stream of
/// undecodable tokens from growing the window without bound.
const MAX_DETOK_WINDOW: usize = 64;

/// Incremental streaming detokenizer.
///
/// Re-decoding the full completion on every generated token is O(n^2) in
/// completion length. Instead, decode only `tokens[prefix_offset..]` and emit
/// the difference against `tokens[prefix_offset..read_offset]`; both windows
/// start at the same token, so tokenizer boundary effects cancel out. Text
/// ending in an incomplete UTF-8 sequence (trailing replacement char) is held
/// back until a later token completes it.
/// Token IDs that must never surface in decoded output text.
///
/// Seeded with the model's EOS ids, plus every added *control* special token:
/// the `<|…|>` chat-control delimiters and the classic `<s>`/`</s>`/`<pad>`/
/// `<unk>`/`<mask>` sentinels. Content-bearing special tokens (`<think>`,
/// `<tool_call>`, `MiniCPM`'s `<function>`/`<param>`) deliberately do NOT match,
/// so they survive decoding and reach the tool-call / reasoning parsers. Shared
/// by the non-streaming decode path ([`SimpleEngine::decode_tokens`]) and the
/// streaming detokenizer ([`IncrementalDetok`]) so the two strip identically.
pub(crate) fn content_preserving_skip_ids(
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> std::collections::HashSet<u32> {
    let mut ids: std::collections::HashSet<u32> = eos_token_ids.iter().copied().collect();
    for (id, added) in tokenizer.get_added_tokens_decoder() {
        let content = added.content.as_str();
        let is_control = (content.starts_with("<|") && content.ends_with("|>"))
            || matches!(content, "<s>" | "</s>" | "<pad>" | "<unk>" | "<mask>");
        if is_control {
            ids.insert(id);
        }
    }
    ids
}

pub(crate) struct IncrementalDetok {
    /// Start of the decode window; advanced whenever text is emitted.
    prefix_offset: usize,
    /// Number of leading tokens already represented in `text`.
    read_offset: usize,
    /// All text decoded so far (streamed to the client incrementally).
    pub(crate) text: String,
    /// Control-token IDs filtered out before decoding so content-bearing
    /// special tokens (tool-call markup) survive streaming. See
    /// [`content_preserving_skip_ids`].
    skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
}

impl IncrementalDetok {
    /// Start from text already decoded for the first `token_count` tokens.
    pub(crate) const fn new(
        text: String,
        token_count: usize,
        skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
    ) -> Self {
        Self {
            prefix_offset: 0,
            read_offset: token_count,
            text,
            skip_ids,
        }
    }

    /// Decode `tokens`, dropping only the control tokens in `skip_ids` while
    /// preserving content-bearing special tokens. Mirrors
    /// [`SimpleEngine::decode_tokens`] so streamed and non-streamed text match.
    fn decode(&self, tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String, EngineError> {
        let run = |ids: &[u32]| {
            tokenizer
                .decode(ids, false)
                .map_err(|e| EngineError::Tokenization(e.to_string()))
        };
        if !tokens.iter().any(|id| self.skip_ids.contains(id)) {
            return run(tokens);
        }
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !self.skip_ids.contains(id))
            .collect();
        run(&filtered)
    }

    /// Decode the trailing window of `tokens`, appending newly stable text to
    /// `self.text` and returning it.
    pub(crate) fn append(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
    ) -> Result<String, EngineError> {
        let prefix_tokens = tokens
            .get(self.prefix_offset..self.read_offset)
            .unwrap_or_default();
        let window_tokens = tokens.get(self.prefix_offset..).unwrap_or_default();
        let prefix_text = self.decode(tokenizer, prefix_tokens)?;
        let window_text = self.decode(tokenizer, window_tokens)?;

        let over_window = window_tokens.len() > MAX_DETOK_WINDOW;
        if window_text.len() > prefix_text.len()
            && (!window_text.ends_with('\u{FFFD}') || over_window)
        {
            let new_text = window_text
                .get(prefix_text.len()..)
                .unwrap_or_default()
                .to_owned();
            self.prefix_offset = self.read_offset;
            self.read_offset = tokens.len();
            self.text.push_str(&new_text);
            return Ok(new_text);
        }
        if over_window && window_text.len() == prefix_text.len() {
            // The pending tokens decode to nothing (e.g. skipped special
            // tokens); drop them so the window stays bounded.
            self.prefix_offset = tokens.len();
            self.read_offset = tokens.len();
        }
        Ok(String::new())
    }

    /// Emit any text still held back by `append` (a trailing incomplete UTF-8
    /// sequence). Called when generation finishes so the total streamed text
    /// matches a full decode of the token buffer.
    pub(crate) fn flush(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
    ) -> Result<String, EngineError> {
        if self.read_offset >= tokens.len() {
            return Ok(String::new());
        }
        let prefix_tokens = tokens
            .get(self.prefix_offset..self.read_offset)
            .unwrap_or_default();
        let window_tokens = tokens.get(self.prefix_offset..).unwrap_or_default();
        let prefix_text = self.decode(tokenizer, prefix_tokens)?;
        let window_text = self.decode(tokenizer, window_tokens)?;
        let new_text = window_text
            .get(prefix_text.len()..)
            .unwrap_or_default()
            .to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = tokens.len();
        self.text.push_str(&new_text);
        Ok(new_text)
    }
}

/// Find the earliest stop-sequence occurrence that could involve the newly appended text.
///
/// Scans only the tail of `text` rather than the whole buffer.
/// Returns the absolute byte position where the match starts.
pub(crate) fn find_stop_in_tail(
    text: &str,
    new_len: usize,
    stop_sequences: &[String],
) -> Option<usize> {
    let max_stop = stop_sequences.iter().map(String::len).max().unwrap_or(0);
    if max_stop == 0 {
        return None;
    }
    let mut start = text.len().saturating_sub(new_len + max_stop - 1);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let tail = text.get(start..)?;
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = tail.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| start + pos)
}

/// Check if any stop sequence appears in the generated text.
/// Returns `Some(truncated_text)` if a stop sequence was found, None otherwise.
fn check_stop_sequences(text: &str, stop_sequences: &[String]) -> Option<String> {
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| text.get(..pos).unwrap_or_default().to_owned())
}

/// Derive a human-readable model name from a directory path.
///
/// Detects `HuggingFace` cache paths (`models--<org>--<name>/snapshots/<hash>`)
/// and extracts `<org>/<name>` instead of using the hash as the name.
/// Falls back to the directory's file name.
pub(crate) fn derive_model_name(model_dir: &Path) -> String {
    // HuggingFace cache: .../models--<org>--<name>/snapshots/<hash>
    if let (Some(leaf), Some(parent)) = (model_dir.file_name(), model_dir.parent()) {
        let leaf_str = leaf.to_string_lossy();
        if let (Some(snapshots), Some(grandparent)) = (parent.file_name(), parent.parent()) {
            if snapshots.to_string_lossy() == "snapshots" {
                let gp_name = grandparent
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default();
                if let Some(rest) = gp_name.strip_prefix("models--") {
                    // "org--model-name" -> "org/model-name"
                    if let Some(sep) = rest.find("--") {
                        let org = &rest[..sep];
                        let model = &rest[sep + 2..];
                        return format!("{org}/{model}");
                    }
                    return rest.to_owned();
                }
            }
        }
        // Not an HF cache path -- use the leaf directory name
        if !leaf_str.is_empty() {
            return leaf_str.to_string();
        }
    }
    "unknown".to_owned()
}

/// Extract EOS token IDs from config.json.
/// Resolve a special token (e.g. `<|im_end|>`) to its single vocab id via the
/// tokenizer, or `None` if it does not encode to exactly one token.
fn single_special_token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer
        .encode(token, false)
        .ok()
        .and_then(|enc| match enc.get_ids() {
            [single] => Some(*single),
            _ => None,
        })
}

/// Merge the chat turn terminator (`<|im_end|>`) into the config-derived EOS set,
/// deduped. Qwen checkpoints that already list it are unaffected; non-Qwen models
/// (`terminator == None`) pass through unchanged.
fn with_chat_terminator(mut eos: Vec<u32>, terminator: Option<u32>) -> Vec<u32> {
    if let Some(id) = terminator {
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    eos
}

pub(crate) fn extract_eos_tokens(model_dir: &Path) -> Vec<u32> {
    let config_path = model_dir.join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(path = %config_path.display(), error = %e, "Could not read config.json for EOS tokens");
            return vec![];
        }
    };

    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "Could not parse config.json for EOS tokens");
            return vec![];
        }
    };

    // Check top-level first, then text_config (VLM/Qwen3.5 nested config).
    // Filter null so explicit `"eos_token_id": null` falls through to text_config.
    let eos_value = config
        .get("eos_token_id")
        .filter(|v| !v.is_null())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("eos_token_id"))
        });

    let mut ids = eos_ids_from_value(eos_value);

    // Union eos_token_id from generation_config.json — HF applies it at inference
    // and it often lists turn-end tokens the base config.json omits.
    if let Ok(gen_str) = std::fs::read_to_string(model_dir.join("generation_config.json"))
        && let Ok(gen_cfg) = serde_json::from_str::<serde_json::Value>(&gen_str)
    {
        for id in eos_ids_from_value(gen_cfg.get("eos_token_id")) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }

    // Gemma chat models end assistant turns with <end_of_turn>, which their
    // config.json typically leaves out of eos_token_id. Without it generation runs
    // past the answer into filler. Add it for the Gemma family.
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|v| v.as_str())
        });
    if model_type.is_some_and(|t| t.starts_with("gemma"))
        && let Some(id) = special_token_id(model_dir, "<end_of_turn>")
        && !ids.contains(&id)
    {
        ids.push(id);
    }

    if ids.is_empty() {
        tracing::warn!("No eos_token_id found in config.json, generation will rely on max_tokens");
    }
    ids
}

/// Parse an `eos_token_id` JSON value (a single number or an array of numbers).
fn eos_ids_from_value(value: Option<&serde_json::Value>) -> Vec<u32> {
    match value {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map_or_else(Vec::new, |id| vec![id]),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().and_then(|val| u32::try_from(val).ok()))
            .collect(),
        _ => vec![],
    }
}

/// Resolve the id of an added special token by its `content` string, reading
/// `tokenizer_config.json`'s `added_tokens_decoder` map.
fn special_token_id(model_dir: &Path, content: &str) -> Option<u32> {
    let text = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    let decoder = config.get("added_tokens_decoder")?.as_object()?;
    decoder.iter().find_map(|(id, info)| {
        (info.get("content").and_then(serde_json::Value::as_str) == Some(content))
            .then(|| id.parse::<u32>().ok())
            .flatten()
    })
}

/// Detect whether a model supports thinking mode based on `model_type`.
/// Whether the model *supports* a thinking toggle (capability, not default).
///
/// The per-request default — e.g. Qwen3.6 reasons off unless asked — is decided
/// separately by `model_defaults_to_non_thinking` in the router; this only
/// answers "can it think at all".
///
/// Signals, in order:
/// 1. the chat template exposes an `enable_thinking` switch — the model
///    author's own marker, which covers Qwen3.5/3.6, `MiniCPM5`, and future
///    reasoning models without hardcoding model types; or
/// 2. a known reasoning `model_type`.
///
/// The caller additionally requires a single-token `</think>`, so a stray
/// mention can't enable thinking for a model that wasn't trained for it.
fn detect_thinking_support(model_dir: &Path) -> bool {
    if chat_template_mentions_enable_thinking(model_dir) {
        return true;
    }
    let Ok(config_str) = std::fs::read_to_string(model_dir.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return false;
    };
    // Check both top-level and nested text_config (VLM wrappers).
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("model_type"))
                .and_then(|v| v.as_str())
        });
    matches!(model_type, Some("qwen3_5" | "qwen3_5_moe"))
}

/// Whether the model's chat template references the `enable_thinking` toggle,
/// read from `chat_template.jinja` or `tokenizer_config.json`'s `chat_template`
/// (a string, or a `{name, template}` array).
fn chat_template_mentions_enable_thinking(model_dir: &Path) -> bool {
    const MARKER: &str = "enable_thinking";
    if let Ok(jinja) = std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
        return jinja.contains(MARKER);
    }
    let Ok(cfg_str) = std::fs::read_to_string(model_dir.join("tokenizer_config.json")) else {
        return false;
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_str) else {
        return false;
    };
    let template = cfg.get("chat_template");
    if let Some(s) = template.and_then(|v| v.as_str()) {
        return s.contains(MARKER);
    }
    if let Some(arr) = template.and_then(|v| v.as_array()) {
        return arr.iter().any(|entry| {
            entry
                .get("template")
                .and_then(|t| t.as_str())
                .is_some_and(|t| t.contains(MARKER))
        });
    }
    false
}

/// Delivery target for tokens produced by [`SimpleEngine::dflash_decode`].
///
/// The draft-verify loop is written once; this trait lets the same loop feed a
/// buffered response ([`DflashBufferedSink`]) or a streaming channel
/// ([`DflashStreamSink`]) without duplicating the gate/EMA logic. Token
/// production is identical for both, so a streamed response is byte-for-byte
/// equal to the buffered one.
trait DflashSink {
    type Output;

    /// Deliver the round's tokens. `tokens` is the full sequence so far;
    /// `forced` is `Some(reason)` when the loop already knows generation ends
    /// this round (EOS / length). Returns `true` if generation should also stop
    /// because a stop sequence fired (or a streaming client disconnected).
    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        engine: &SimpleEngine,
        tokens: &[u32],
        stop_sequences: &[String],
        prompt_len: u32,
        completion_len: u32,
        forced: Option<&'static str>,
    ) -> Result<bool, EngineError>;

    /// Produce the final result once the loop terminates.
    fn finish(
        self,
        engine: &SimpleEngine,
        tokens: &[u32],
        finish_reason: &str,
        prompt_len: u32,
        completion_len: u32,
    ) -> Result<Self::Output, EngineError>;
}

/// Buffered sink: accumulates the full output and returns a [`GenerationOutput`].
/// Mirrors the original non-streaming `DFlash` behavior exactly.
#[derive(Default)]
struct DflashBufferedSink {
    /// Set when a stop sequence truncated the output, so `finish` returns the
    /// truncated text rather than the full decode.
    truncated: Option<String>,
}

impl DflashSink for DflashBufferedSink {
    type Output = GenerationOutput;

    fn emit(
        &mut self,
        engine: &SimpleEngine,
        tokens: &[u32],
        stop_sequences: &[String],
        _prompt_len: u32,
        _completion_len: u32,
        _forced: Option<&'static str>,
    ) -> Result<bool, EngineError> {
        if stop_sequences.is_empty() {
            return Ok(false);
        }
        let text = engine.decode_tokens(tokens)?;
        if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
            self.truncated = Some(truncated);
            return Ok(true);
        }
        Ok(false)
    }

    fn finish(
        self,
        engine: &SimpleEngine,
        tokens: &[u32],
        finish_reason: &str,
        prompt_len: u32,
        completion_len: u32,
    ) -> Result<GenerationOutput, EngineError> {
        let text = match self.truncated {
            Some(t) => t,
            None => engine.decode_tokens(tokens)?,
        };
        Ok(GenerationOutput {
            text,
            finish_reason: finish_reason.to_owned(),
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            token_logprobs: None,
        })
    }
}

/// Streaming sink: emits one [`StreamingOutput`] chunk per round over the
/// channel, mirroring the MTP streaming contract (incremental text, stop-tail
/// truncation, a final `finished: true` chunk).
struct DflashStreamSink<'a> {
    sender: &'a tokio::sync::mpsc::Sender<StreamingOutput>,
    detok: IncrementalDetok,
}

impl DflashSink for DflashStreamSink<'_> {
    type Output = ();

    fn emit(
        &mut self,
        engine: &SimpleEngine,
        tokens: &[u32],
        stop_sequences: &[String],
        prompt_len: u32,
        completion_len: u32,
        forced: Option<&'static str>,
    ) -> Result<bool, EngineError> {
        let new_text = self.detok.append(&engine.tokenizer, tokens)?;
        let emitted_before = self.detok.text.len() - new_text.len();
        let (mut final_new_text, hit_stop_seq) = if stop_sequences.is_empty() {
            (new_text, false)
        } else {
            find_stop_in_tail(&self.detok.text, new_text.len(), stop_sequences).map_or(
                (new_text, false),
                |pos| {
                    let emit = self
                        .detok
                        .text
                        .get(emitted_before..pos)
                        .unwrap_or_default()
                        .to_owned();
                    (emit, true)
                },
            )
        };
        let finished = forced.is_some() || hit_stop_seq;
        // EOS / length terminate without a stop sequence: flush any buffered
        // partial bytes. A stop sequence already emitted only up to the cut.
        if finished && !hit_stop_seq {
            final_new_text.push_str(&self.detok.flush(&engine.tokenizer, tokens)?);
        }
        let finish_reason = finished.then(|| forced.unwrap_or("stop").to_owned());
        if self
            .sender
            .blocking_send(StreamingOutput {
                new_text: final_new_text,
                finished,
                finish_reason,
                prompt_tokens: prompt_len,
                completion_tokens: completion_len,
                token_logprob: None,
                prefill_progress: None,
            })
            .is_err()
        {
            // Receiver dropped (client disconnected) — stop generating.
            return Ok(true);
        }
        Ok(hit_stop_seq)
    }

    fn finish(
        self,
        _engine: &SimpleEngine,
        _tokens: &[u32],
        _finish_reason: &str,
        _prompt_len: u32,
        _completion_len: u32,
    ) -> Result<(), EngineError> {
        // The terminal `finished: true` chunk was already sent by `emit`.
        Ok(())
    }
}

impl SimpleEngine {
    /// Buffered `DFlash` generation (non-streaming response path).
    fn generate_dflash_inner(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        enable_thinking: bool,
        thinking_budget: u32,
    ) -> Result<GenerationOutput, EngineError> {
        self.dflash_decode(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            enable_thinking,
            thinking_budget,
            DflashBufferedSink::default(),
        )
    }

    /// Streaming `DFlash` generation: same draft-verify loop, tokens delivered
    /// chunk-by-chunk over `sender` as they are accepted.
    fn generate_dflash_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        thinking_budget: u32,
    ) -> Result<(), EngineError> {
        let detok = IncrementalDetok::new(
            String::new(),
            0,
            std::sync::Arc::clone(&self.decode_skip_ids),
        );
        self.dflash_decode(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            enable_thinking,
            thinking_budget,
            DflashStreamSink { sender, detok },
        )
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::{
        CanonicalDflashRound, DFlashVerifyMode, DFlashVerifyRound, DflashPromptPartition,
        DflashTapFrontier, EngineError, GenerationPromptSuffixes, IncrementalDetok, LivePair,
        SessionDsparkDecodeState, SessionGeneration, SimpleEngine, SpeculationRoute, Tokenizer,
        adaptive_draft_depth_for_cap, check_stop_sequences, continuation_prior_len,
        derive_model_name, detect_thinking_support, dflash_canonical_target_is_terminal,
        dflash_new_stop_prefix_len, dflash_resolve_target_then_draft, dflash_tail_draft_cap,
        drive_canonical_dspark_round, estimate_paged_kv_blocks, extract_eos_tokens,
        find_stop_in_tail, lock_or_recover, paired_dflash_exact_domain, parse_enabled_flag,
        resolve_speculation_route, stateless_mtp_family_eligible, with_chat_terminator,
    };
    use crate::chat_template::ChatMessage;
    use crate::decode::token_ledger::TokenLedger;
    use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
    use crate::scheduler::RoundRobinScheduler;
    use higgs_models::{
        AnyCache, AnyModel, SamplingParams, Speculation,
        cache::{KeyValueCache, SteppingKeyValueCache},
        dflash::{DFlashConfig, DFlashDrafter},
        transformer::{Model as TransformerModel, ModelArgs as TransformerModelArgs},
        turboquant::KvCacheConfig,
    };
    use mlx_rs::{
        Array, Dtype,
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
    };
    use std::path::Path;

    fn validated_session_target(boundary: i32) -> AnyCache {
        let layer = if boundary == 0 {
            SteppingKeyValueCache::new()
        } else {
            let keys = Array::zeros::<f32>(&[1, 1, boundary, 1]).unwrap();
            let values = Array::zeros::<f32>(&[1, 1, boundary, 1]).unwrap();
            SteppingKeyValueCache::from_arrays(keys, values).unwrap()
        };
        let _exec = higgs_models::mlx_exec::acquire();
        let cache = AnyCache::KV(vec![Some(layer)]);
        cache.eval().unwrap();
        cache.validate_absolute_boundary(boundary).unwrap();
        cache
    }

    fn session_test_drafter() -> DFlashDrafter {
        let config: DFlashConfig = serde_json::from_str(
            r#"{
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 8,
                "vocab_size": 8,
                "dflash_config": {
                    "target_layer_ids": [0]
                }
            }"#,
        )
        .unwrap();
        DFlashDrafter::new(config).unwrap()
    }

    fn paired_retained_state(boundary: i32) -> crate::cache::paired::RetainedState {
        let target = validated_session_target(boundary);
        let mut drafter = session_test_drafter();
        let cache = drafter.make_cache();
        let taps = (boundary > 0)
            .then(|| Array::zeros::<f32>(&[1, boundary, 4]).unwrap())
            .into_iter()
            .collect::<Vec<_>>();
        let _exec = higgs_models::mlx_exec::acquire();
        let snapshot = drafter.seal_after_taps(cache, &taps, boundary).unwrap();
        let tokens = vec![7; usize::try_from(boundary).unwrap()];
        crate::cache::paired::RetainedState::paired(target, snapshot, &tokens).unwrap()
    }

    fn advance_session_pair(pair: LivePair, suffix: &[u32]) -> LivePair {
        let _exec = higgs_models::mlx_exec::acquire();
        pair.prefill_known(suffix, |exact, target, _draft| {
            let AnyCache::KV(layers) = target else {
                panic!("synthetic session target must use a KV cache");
            };
            let layer = layers
                .first_mut()
                .and_then(Option::as_mut)
                .expect("synthetic session target layer");
            for &token in exact {
                let value = token as f32;
                let keys = Array::from_slice(&[value], &[1, 1, 1, 1]);
                let values = Array::from_slice(&[-value], &[1, 1, 1, 1]);
                layer.update_and_fetch(keys, values).unwrap();
            }
            let rows = i32::try_from(exact.len()).unwrap();
            let taps = vec![Array::zeros::<f32>(&[1, rows, 4]).unwrap()];
            Ok::<_, EngineError>(((), taps))
        })
        .unwrap()
        .0
    }

    fn session_live_pair(tokens: &[u32]) -> (LivePair, DFlashDrafter) {
        assert!(!tokens.is_empty());
        let drafter = session_test_drafter();
        let pair = LivePair::cold(validated_session_target(0), drafter.make_cache(), 1).unwrap();
        (advance_session_pair(pair, tokens), drafter)
    }

    fn session_cache_test_engine() -> SimpleEngine {
        let model_args: TransformerModelArgs = serde_json::from_str(
            r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "intermediate_size": 8,
                "num_attention_heads": 1,
                "rms_norm_eps": 0.00001,
                "vocab_size": 8,
                "num_key_value_heads": 1,
                "max_position_embeddings": 32,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();
        let kv_cache_config = KvCacheConfig {
            max_retained_sessions: 2,
            retained_idle_secs: 0,
            ..KvCacheConfig::default()
        };
        SimpleEngine {
            model: std::sync::Mutex::new(AnyModel::Transformer(
                TransformerModel::new(model_args).unwrap(),
            )),
            prefix_cache: std::sync::Mutex::new(crate::cache::DiskPrefixCache::memory_only(2, 1)),
            paged_cache: None,
            scheduler: std::sync::Mutex::new(RoundRobinScheduler::new()),
            sessions: std::sync::Mutex::new(std::collections::HashMap::new()),
            retained: std::sync::Mutex::new(std::collections::HashMap::new()),
            session_locks: std::sync::Mutex::new(std::collections::HashMap::new()),
            cache_metrics: super::CacheMetrics::default(),
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template: None,
            model_name: "synthetic-session-cache".to_owned(),
            eos_token_ids: Vec::new(),
            decode_skip_ids: std::sync::Arc::new(std::collections::HashSet::new()),
            enable_thinking: false,
            think_close_token: None,
            gen_prompt_suffixes: GenerationPromptSuffixes::default(),
            kv_cache_config,
            tuning: MlxRuntimeTuning::from_model_dir(
                Path::new("/__higgs_missing_session_cache_fixture__"),
                RequestedMlxProfile::Baseline,
            ),
            dflash: None,
            last_dflash_accepts: std::sync::Mutex::new(Vec::new()),
            last_dflash_draft_matches: std::sync::Mutex::new(Vec::new()),
            last_dflash_draft_counts: std::sync::Mutex::new(Vec::new()),
            last_ar_timing: std::sync::Mutex::new(None),
            last_dflash_timing: std::sync::Mutex::new(None),
        }
    }

    #[test]
    fn continued_generation_exposes_request_scoped_thinking_control() {
        let _: fn(
            &SimpleEngine,
            u64,
            &[u32],
            u32,
            &SamplingParams,
            bool,
        ) -> Result<SessionGeneration, EngineError> =
            SimpleEngine::generate_continued_with_thinking;
    }

    #[test]
    fn simple_engine_session_effectful_seal_failure_republishes_neither_half() {
        const SESSION_ID: u64 = 0xD5A4_FA11;
        let engine = session_cache_test_engine();
        let (pair, mut drafter) = session_live_pair(&[11]);

        let mlx_gate = higgs_models::mlx_exec::acquire();
        let publication =
            engine.seal_session_dspark_publication(SESSION_ID, pair, &mut drafter, &mlx_gate);
        assert!(
            publication.is_some(),
            "fixture must first seal a proven pair"
        );
        drop(mlx_gate);
        engine.publish_session_dspark_publication(SESSION_ID, publication, false);

        assert_eq!(engine.retained_session_tokens(SESSION_ID), Some(vec![11]));
        assert_eq!(engine.cache_stats().retained_paired_sessions, 1);

        let (retained, prior) = engine
            .take_continuable(SESSION_ID, &[11, 12])
            .expect("published pair must move out for an exact continuation");
        assert_eq!(prior, 1);
        let crate::cache::paired::RetainedState::Paired(pair) = retained else {
            panic!("fixture must resume an inseparable target/dSpark pair");
        };
        let pair = pair.resume(&[11], 1).unwrap();
        let pair = advance_session_pair(pair, &[12]);

        // Preserve tap capability so this enters effectful sealing, then inject
        // a hidden-width mismatch that deterministically fails DFlash sealing.
        drafter.config.hidden_size = 5;
        let mlx_gate = higgs_models::mlx_exec::acquire();
        let publication =
            engine.seal_session_dspark_publication(SESSION_ID, pair, &mut drafter, &mlx_gate);
        assert!(
            publication.is_none(),
            "effectful seal failure must yield no publishable session outcome"
        );
        drop(mlx_gate);
        engine.publish_session_dspark_publication(SESSION_ID, publication, false);

        let stats = engine.cache_stats();
        assert_eq!(engine.retained_session_tokens(SESSION_ID), None);
        assert_eq!(stats.retained_sessions, 0);
        assert_eq!(stats.retained_paired_sessions, 0);
        assert_eq!(stats.retained_paired_target_bytes, 0);
        assert_eq!(stats.retained_paired_dflash_bytes, 0);
    }

    #[test]
    fn session_dspark_decode_state_keeps_visible_eos_out_of_the_pair_boundary() {
        let drafter = session_test_drafter();
        let pair = LivePair::cold(validated_session_target(0), drafter.make_cache(), 1).unwrap();
        let decode = SessionDsparkDecodeState::begin(pair, 99).unwrap();

        assert_eq!(decode.tokens(), [99]);
        assert_eq!(decode.pending_token(), Some(99));
        let (pair, visible) = decode
            .finish(
                &[99],
                |_target, _draft, _frontier, _token| -> Result<_, EngineError> {
                    Err(EngineError::Generation(
                        "EOS must not enter the cache-only forward".to_owned(),
                    ))
                },
            )
            .unwrap();

        assert_eq!(visible, [99]);
        assert_eq!(
            pair.token_len(),
            0,
            "the visible terminal token must not become publication identity"
        );
    }

    #[test]
    fn dspark_radix_partition_uses_the_request_specific_generation_suffix() {
        let suffixes = GenerationPromptSuffixes {
            no_thinking: Box::from([14, 15, 16]),
            thinking: Box::from([12, 13, 14, 15, 16]),
        };
        assert_eq!(suffixes.for_request(false), &[14, 15, 16]);
        assert_eq!(suffixes.for_request(true), &[12, 13, 14, 15, 16]);

        let prompt = [10, 11, 12, 13, 14, 15, 16];
        let no_thinking = DflashPromptPartition::new(&prompt, suffixes.for_request(false)).unwrap();
        assert_eq!(no_thinking.body(), &[10, 11, 12, 13]);
        assert_eq!(no_thinking.from_boundary(4).unwrap(), &[14, 15, 16]);

        let thinking = DflashPromptPartition::new(&prompt, suffixes.for_request(true)).unwrap();
        assert_eq!(thinking.body(), &[10, 11]);
        assert_eq!(thinking.from_boundary(2).unwrap(), &[12, 13, 14, 15, 16]);
    }

    #[test]
    fn dspark_radix_partition_rejects_a_full_hit_without_generation_tail() {
        let prompt = [1, 2, 3];
        assert!(DflashPromptPartition::new(&prompt, &[]).is_none());
        assert!(DflashPromptPartition::new(&prompt, &prompt).is_none());
        assert!(DflashPromptPartition::new(&prompt, &[0, 2, 3]).is_none());
        assert!(DflashPromptPartition::new(&prompt, &[2, 3, 4]).is_none());
    }

    #[test]
    fn canonical_round_commit_is_shared_by_stateless_and_session_outputs() {
        let mut ledger = TokenLedger::new(10);
        ledger.record_forwarded(40).unwrap();
        ledger.emit_pending(41).unwrap();
        let mut output = vec![40, 41];
        let mut frontier = None;

        let committed = drive_canonical_dspark_round(&mut ledger, |anchor_ticket, history| {
            assert_eq!(history, [40]);
            let round = CanonicalDflashRound {
                anchor_ticket,
                successors: vec![42, 43],
                tap_frontier: DflashTapFrontier::new(11, 13, vec![], 0)?,
                draft_tokens: vec![42],
                draft_matches: 1,
            };
            let (next_frontier, commit) = round.into_lease_parts()?;
            frontier = Some(next_frontier);
            Ok(commit)
        })
        .unwrap();
        output.extend_from_slice(&committed.successors);

        assert_eq!(output, [40, 41, 42, 43]);
        assert_eq!(ledger.emitted_tokens(), output);
        assert_eq!(ledger.pending_token(), Some(43));
        assert_eq!(committed.successors, [42, 43]);
        assert_eq!(committed.draft_tokens, [42]);
        assert_eq!(frontier.unwrap().target_boundary(), 13);
    }

    #[test]
    fn canonical_round_can_only_replace_its_final_pending_successor() {
        let mut ledger = TokenLedger::new(10);
        ledger.emit_pending(41).unwrap();
        let mut output = vec![41];
        let committed = drive_canonical_dspark_round(&mut ledger, |anchor_ticket, _history| {
            let mut round = CanonicalDflashRound {
                anchor_ticket,
                successors: vec![42, 43],
                tap_frontier: DflashTapFrontier::new(10, 12, vec![], 0)?,
                draft_tokens: vec![42],
                draft_matches: 1,
            };
            round.replace_pending_successor(99)?;
            let (_frontier, commit) = round.into_lease_parts()?;
            Ok(commit)
        })
        .unwrap();
        output.extend_from_slice(&committed.successors);

        assert_eq!(output, [41, 42, 99]);
        assert_eq!(ledger.pending_token(), Some(99));
    }

    #[test]
    fn canonical_round_splits_frontier_before_output_publication() {
        let mut ledger = TokenLedger::new(10);
        ledger.emit_pending(41).unwrap();
        let anchor_ticket = ledger.begin_speculative_round().unwrap();
        let round = CanonicalDflashRound {
            anchor_ticket,
            successors: vec![42, 43],
            tap_frontier: DflashTapFrontier::new(10, 11, vec![], 0).unwrap(),
            draft_tokens: vec![42],
            draft_matches: 1,
        };
        let output = vec![41];

        assert!(round.into_lease_parts().is_err());

        assert_eq!(output, [41]);
        assert_eq!(ledger.emitted_tokens(), [41]);
        assert_eq!(
            ledger.retention_action(&[]),
            super::RetentionAction::ForwardInFlight { token: 41 }
        );
    }

    #[test]
    fn canonical_driver_never_commits_after_backend_validation_failure() {
        let mut ledger = TokenLedger::new(10);
        ledger.emit_pending(41).unwrap();

        let error = match drive_canonical_dspark_round(&mut ledger, |_ticket, _history| {
            Err(EngineError::Generation(
                "injected lease postcondition failure".to_owned(),
            ))
        }) {
            Ok(_) => panic!("a failed backend must not commit a canonical round"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("lease postcondition failure"));
        assert_eq!(ledger.emitted_tokens(), [41]);
        assert_eq!(
            ledger.retention_action(&[]),
            super::RetentionAction::ForwardInFlight { token: 41 },
            "a failed backend must leave the anchor in-flight rather than publishing successors"
        );
    }

    #[test]
    fn speculation_route_truth_table_is_exhaustive() {
        use SpeculationRoute::{Autoregressive, DFlash, MtpFamily};

        let cases = [
            (Speculation::None, false, false, Autoregressive),
            (Speculation::None, false, true, Autoregressive),
            (Speculation::None, true, false, Autoregressive),
            (Speculation::None, true, true, Autoregressive),
            (Speculation::Mtp, false, false, Autoregressive),
            (Speculation::Mtp, false, true, MtpFamily),
            (Speculation::Mtp, true, false, Autoregressive),
            (Speculation::Mtp, true, true, MtpFamily),
            (Speculation::DFlash, false, false, Autoregressive),
            (Speculation::DFlash, false, true, Autoregressive),
            (Speculation::DFlash, true, false, DFlash),
            (Speculation::DFlash, true, true, DFlash),
            (Speculation::Auto, false, false, Autoregressive),
            (Speculation::Auto, false, true, MtpFamily),
            (Speculation::Auto, true, false, DFlash),
            (Speculation::Auto, true, true, DFlash),
        ];

        for (selector, dflash_eligible, mtp_family_eligible, expected) in cases {
            assert_eq!(
                resolve_speculation_route(selector, dflash_eligible, mtp_family_eligible),
                expected,
                "selector={selector:?} dflash_eligible={dflash_eligible} mtp_family_eligible={mtp_family_eligible}"
            );
        }
    }

    #[test]
    fn penalties_disable_stateless_mtp_family_but_not_session_mtp() {
        let penalized = SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.1),
            speculation: Speculation::Mtp,
            ..SamplingParams::default()
        };

        assert_eq!(
            resolve_speculation_route(
                penalized.speculation,
                false,
                stateless_mtp_family_eligible(&penalized, false, false),
            ),
            SpeculationRoute::Autoregressive,
            "stateless prompt-lookup/native-MTP loops do not apply penalty history"
        );
        assert_eq!(
            resolve_speculation_route(penalized.speculation, false, true),
            SpeculationRoute::MtpFamily,
            "session MTP remains eligible because its loop receives params/history"
        );
    }

    #[test]
    fn session_speculation_eligibility_keeps_dspark_independent_of_mtp() {
        let greedy = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };

        assert_eq!(
            resolve_speculation_route(
                greedy.speculation,
                paired_dflash_exact_domain(&greedy, false),
                false,
            ),
            SpeculationRoute::DFlash,
            "Bonsai dSpark must not be nested behind an MTP head"
        );

        let none = SamplingParams {
            speculation: Speculation::None,
            ..greedy.clone()
        };
        assert_eq!(
            resolve_speculation_route(
                none.speculation,
                paired_dflash_exact_domain(&none, false),
                true,
            ),
            SpeculationRoute::Autoregressive
        );

        let mtp = SamplingParams {
            speculation: Speculation::Mtp,
            ..greedy.clone()
        };
        assert_eq!(
            resolve_speculation_route(
                mtp.speculation,
                paired_dflash_exact_domain(&mtp, false),
                true,
            ),
            SpeculationRoute::MtpFamily
        );
        assert_eq!(
            resolve_speculation_route(
                mtp.speculation,
                paired_dflash_exact_domain(&mtp, false),
                false,
            ),
            SpeculationRoute::Autoregressive
        );

        let forced_dflash = SamplingParams {
            speculation: Speculation::DFlash,
            ..greedy.clone()
        };
        assert_eq!(
            resolve_speculation_route(
                forced_dflash.speculation,
                paired_dflash_exact_domain(&forced_dflash, false),
                true,
            ),
            SpeculationRoute::DFlash
        );
        assert_eq!(
            resolve_speculation_route(forced_dflash.speculation, false, true),
            SpeculationRoute::Autoregressive,
            "forced dFlash must not silently become MTP"
        );

        let sampled = SamplingParams {
            temperature: 0.7,
            ..greedy.clone()
        };
        assert_eq!(
            resolve_speculation_route(
                sampled.speculation,
                paired_dflash_exact_domain(&sampled, false),
                true,
            ),
            SpeculationRoute::MtpFamily,
            "auto may use MTP when paired dSpark is outside its exact domain"
        );
        let forced_sampled = SamplingParams {
            speculation: Speculation::DFlash,
            ..sampled
        };
        assert_eq!(
            resolve_speculation_route(
                forced_sampled.speculation,
                paired_dflash_exact_domain(&forced_sampled, false),
                true,
            ),
            SpeculationRoute::Autoregressive
        );
        assert_eq!(
            resolve_speculation_route(
                greedy.speculation,
                paired_dflash_exact_domain(&greedy, true),
                true,
            ),
            SpeculationRoute::MtpFamily,
            "paired dSpark initially requires no-thinking mode"
        );
    }

    #[test]
    fn dspark_verify_mode_fails_closed() {
        assert_eq!(
            DFlashVerifyMode::for_drafter(true, None),
            DFlashVerifyMode::CanonicalS1
        );
        assert_eq!(
            DFlashVerifyMode::for_drafter(true, Some("sequential")),
            DFlashVerifyMode::CanonicalS1
        );
        assert_eq!(
            DFlashVerifyMode::for_drafter(true, Some("block")),
            DFlashVerifyMode::BatchedTape
        );
        assert_eq!(
            DFlashVerifyMode::for_drafter(true, Some("typo")),
            DFlashVerifyMode::CanonicalS1
        );
        assert_eq!(
            DFlashVerifyMode::for_drafter(false, Some("sequential")),
            DFlashVerifyMode::BatchedTape
        );

        let greedy = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        assert_eq!(
            DFlashVerifyMode::BatchedTape.for_request(true, &greedy, false),
            DFlashVerifyMode::BatchedTape
        );
        assert_eq!(
            DFlashVerifyMode::BatchedTape.for_request(true, &SamplingParams::default(), false),
            DFlashVerifyMode::CanonicalS1
        );
        let penalized = SamplingParams {
            temperature: 0.0,
            repetition_penalty: Some(1.1),
            ..SamplingParams::default()
        };
        assert_eq!(
            DFlashVerifyMode::BatchedTape.for_request(true, &penalized, false),
            DFlashVerifyMode::CanonicalS1
        );
        assert_eq!(
            DFlashVerifyMode::BatchedTape.for_request(true, &greedy, true),
            DFlashVerifyMode::CanonicalS1
        );
        assert_eq!(
            DFlashVerifyMode::BatchedTape.for_request(false, &penalized, true),
            DFlashVerifyMode::BatchedTape
        );

        let mut committed = DFlashVerifyRound::committed(vec![1, 2], vec![], 0);
        assert_eq!(committed.validate_output_boundary().unwrap(), 2);
        assert!(committed.replace_pending_output(0, 9).is_err());
        if let DFlashVerifyRound::Committed { output_tokens, .. } = &mut committed {
            output_tokens.truncate(1);
        }
        assert!(
            committed.validate_output_boundary().is_err(),
            "an output policy cannot truncate already-committed target state"
        );
        let mut tentative = DFlashVerifyRound::TentativeTape {
            target_rows: 2,
            target_tokens: vec![1, 2],
            output_tokens: vec![1, 2],
            taps: vec![],
            layer_tapes: vec![],
            pending_replaced: false,
            expected_taps: 0,
        };
        tentative.replace_pending_output(0, 9).unwrap();
        assert_eq!(tentative.validate_output_boundary().unwrap(), 1);
    }

    #[test]
    fn dspark_tail_cap_never_verifies_unemittable_rows() {
        assert_eq!(dflash_tail_draft_cap(4, 128, true), 4);
        assert_eq!(dflash_tail_draft_cap(4, 5, true), 4);
        assert_eq!(dflash_tail_draft_cap(4, 4, true), 3);
        assert_eq!(dflash_tail_draft_cap(4, 3, true), 2);
        assert_eq!(dflash_tail_draft_cap(4, 2, true), 1);
        assert_eq!(dflash_tail_draft_cap(4, 1, true), 1);
        assert_eq!(
            dflash_tail_draft_cap(16, 3, false),
            16,
            "generic DFlash confidence scheduling owns its width"
        );
    }

    #[test]
    fn dspark_canonical_stops_before_advancing_past_natural_think_close() {
        let targets = [10_u32, 99, 12];
        let drafts = [10_u32, 99, 12];
        let mut committed = 0_usize;
        for (position, &target) in targets.iter().enumerate() {
            committed += 1;
            if dflash_canonical_target_is_terminal(position, target, &drafts, &[], false, Some(99))
            {
                break;
            }
        }
        assert_eq!(committed, 2, "the poison row after </think> was advanced");
    }

    #[test]
    fn dspark_stop_boundary_detects_only_newly_completed_stops() {
        let stops = vec!["STOP".to_owned()];
        let decode = |tokens: &[u32]| -> Result<String, ()> {
            Ok(tokens.iter().fold(String::new(), |mut text, token| {
                text.push_str(match token {
                    1 => "abcS",
                    2 => "T",
                    3 => "O",
                    4 => "P",
                    5 => "STOP-old",
                    6 => "x",
                    _ => "?",
                });
                text
            }))
        };
        assert_eq!(
            dflash_new_stop_prefix_len(&[1, 2], &[3, 4, 6], &stops, decode).unwrap(),
            Some(2),
            "stop spanning committed and candidate tokens must cap the transaction"
        );
        assert_eq!(
            dflash_new_stop_prefix_len(&[5], &[6], &stops, decode).unwrap(),
            None,
            "a stop wholly inside old output must not terminate a new round"
        );
    }

    #[test]
    fn dspark_resolves_target_barrier_before_proposal_host_read() {
        let events = std::cell::RefCell::new(Vec::new());
        let (target, draft) = dflash_resolve_target_then_draft(
            || {
                events.borrow_mut().push("target");
                Ok(vec![1_u32])
            },
            || {
                events.borrow_mut().push("draft");
                Ok(vec![1_u32])
            },
        )
        .unwrap();
        assert_eq!((target, draft), (vec![1], vec![1]));
        assert_eq!(*events.borrow(), ["target", "draft"]);
    }

    #[test]
    fn continuation_prior_len_guard() {
        // Retained tokens are a strict prefix AND the new prompt extends them → continue.
        assert_eq!(
            continuation_prior_len(&[1, 2, 3], &[1, 2, 3, 4, 5]),
            Some(3)
        );
        // Diverged mid-prefix (client edited/retried) → clean fallback.
        assert_eq!(continuation_prior_len(&[1, 2, 3], &[1, 2, 9, 4]), None);
        // No new tokens (same length) → nothing to continue.
        assert_eq!(continuation_prior_len(&[1, 2, 3], &[1, 2, 3]), None);
        // Empty retained → nothing to reuse.
        assert_eq!(continuation_prior_len(&[], &[1, 2]), None);
        // Retained longer than the new prompt → not a prefix.
        assert_eq!(continuation_prior_len(&[1, 2, 3, 4], &[1, 2]), None);
        // session_id collision: a DIFFERENT conversation reuses the same id and
        // shares no prefix → clean fallback, never wrong reuse.
        assert_eq!(continuation_prior_len(&[1, 2, 3], &[7, 8, 9, 10]), None);
        // History edited at the very first token → fallback.
        assert_eq!(continuation_prior_len(&[1, 2, 3], &[9, 2, 3, 4]), None);
        // A genuine prefix is accepted even across a collision — and that is
        // SOUND: a retained cache is always paired with its exact tokens, so
        // reuse is only ever offered when those tokens really do lead the new
        // prompt; the reconstructed KV therefore always matches the sequence.
        assert_eq!(continuation_prior_len(&[5, 6], &[5, 6, 7]), Some(2));
    }

    #[test]
    fn retention_caps_bound_the_retained_map() {
        use super::{RetainedKv, evict_idle_from, retention_token_cap, stash_into};
        use crate::cache::paired::RetainedState;
        use higgs_models::AnyCache;
        use higgs_models::turboquant::KvCacheConfig;
        use std::collections::HashMap;
        use std::time::{Duration, Instant};

        // An empty KV cache is a valid `AnyCache` and needs no GPU.
        let dummy = |tokens: Vec<u32>| {
            RetainedState::target_only_unchecked_for_test(AnyCache::KV(Vec::new()), tokens)
        };

        // -- count cap: never exceed max_sessions (LRU-evicted) --
        let mut map: HashMap<u64, RetainedKv> = HashMap::new();
        for sid in 0..5u64 {
            stash_into(&mut map, sid, dummy(vec![1, 2, 3]), 2, 0);
        }
        assert_eq!(map.len(), 2, "count cap must bound retained sessions");

        // stash_into reports how many sessions it LRU-evicted (for the metrics counter).
        let mut cap_map: HashMap<u64, RetainedKv> = HashMap::new();
        assert_eq!(stash_into(&mut cap_map, 1, dummy(vec![1]), 2, 0), 0);
        assert_eq!(stash_into(&mut cap_map, 2, dummy(vec![1]), 2, 0), 0);
        assert_eq!(
            stash_into(&mut cap_map, 3, dummy(vec![1]), 2, 0),
            1,
            "inserting past the cap evicts exactly one"
        );

        // max_sessions=0 is clamped to 1 (never retain nothing-but-evict-all).
        let mut one: HashMap<u64, RetainedKv> = HashMap::new();
        stash_into(&mut one, 1, dummy(vec![1]), 0, 0);
        assert_eq!(one.len(), 1, "max_sessions=0 clamps to 1");

        // -- per-session token cap: oversized conversation is not retained --
        let mut tc: HashMap<u64, RetainedKv> = HashMap::new();
        stash_into(&mut tc, 7, dummy(vec![0; 5]), 8, 10); // 5 <= 10 → kept
        assert!(tc.contains_key(&7));
        stash_into(&mut tc, 7, dummy(vec![0; 20]), 8, 10); // 20 > 10 → dropped
        assert!(
            !tc.contains_key(&7),
            "oversized session must not be retained, and its prior cache is forgotten"
        );

        // token cap 0 = unlimited
        let mut un: HashMap<u64, RetainedKv> = HashMap::new();
        stash_into(&mut un, 1, dummy(vec![0; 100_000]), 8, 0);
        assert!(un.contains_key(&1), "max_session_tokens=0 means unlimited");

        // cap_exempt is used after over-cap caches are TurboQuant-compressed for
        // retention: the compressed cache should stay retained instead of
        // forcing the next turn to re-prefill the full long context.
        let capped = KvCacheConfig {
            max_session_tokens: 10,
            ..KvCacheConfig::default()
        };
        let ordinary = dummy(vec![0; 20]);
        assert_eq!(retention_token_cap(capped, false, &ordinary), 10);
        assert_eq!(retention_token_cap(capped, true, &ordinary), 0);
        let mut exempt: HashMap<u64, RetainedKv> = HashMap::new();
        let ordinary = dummy(vec![0; 20]);
        let ordinary_cap = retention_token_cap(capped, true, &ordinary);
        stash_into(
            &mut exempt,
            99,
            ordinary,
            capped.max_retained_sessions,
            ordinary_cap,
        );
        assert!(
            exempt.contains_key(&99),
            "cap-exempt compressed sessions must still be retained"
        );

        // -- TTL eviction boundary --
        let mut ttl: HashMap<u64, RetainedKv> = HashMap::new();
        for sid in 0..2u64 {
            ttl.insert(
                sid,
                RetainedKv {
                    state: dummy(vec![1]),
                    last_used: Instant::now(),
                },
            );
        }
        assert_eq!(
            evict_idle_from(&mut ttl, Duration::from_secs(999)),
            0,
            "fresh entries survive a long TTL"
        );
        assert_eq!(ttl.len(), 2);
        assert_eq!(
            evict_idle_from(&mut ttl, Duration::ZERO),
            2,
            "a zero TTL evicts everything"
        );
        assert!(ttl.is_empty());
    }

    #[test]
    fn paired_retention_does_not_inherit_target_only_cap_exemption() {
        use super::{RetainedKv, retention_token_cap, stash_into};
        use higgs_models::turboquant::KvCacheConfig;
        use std::collections::HashMap;

        let config = KvCacheConfig {
            max_session_tokens: 10,
            ..KvCacheConfig::default()
        };
        let paired = paired_retained_state(20);

        assert_eq!(
            retention_token_cap(config, true, &paired),
            config.max_session_tokens,
            "target TurboQuant must not unbound the uncompressed dSpark half"
        );

        let mut retained: HashMap<u64, RetainedKv> = HashMap::new();
        let effective_cap = retention_token_cap(config, true, &paired);
        stash_into(
            &mut retained,
            100,
            paired,
            config.max_retained_sessions,
            effective_cap,
        );
        assert!(
            retained.is_empty(),
            "an over-cap target+dSpark pair must be dropped after target compression"
        );
    }

    #[test]
    fn paired_session_ttl_evicts_the_whole_pair() {
        use super::{RetainedKv, evict_idle_from, retained_paired_stats};
        use std::collections::HashMap;
        use std::time::{Duration, Instant};

        let mut retained: HashMap<u64, RetainedKv> = HashMap::new();
        retained.insert(
            100,
            RetainedKv {
                state: paired_retained_state(1),
                last_used: Instant::now(),
            },
        );

        let before = retained_paired_stats(&retained);
        assert_eq!(before.entries, 1);
        assert!(before.target_bytes > 0);
        assert!(before.dflash_bytes > 0);
        assert_eq!(
            evict_idle_from(&mut retained, Duration::from_secs(999)),
            0,
            "a fresh retained pair must survive its TTL"
        );

        assert_eq!(
            evict_idle_from(&mut retained, Duration::ZERO),
            1,
            "TTL expiry must remove the one ownership entry containing both halves"
        );
        assert!(retained.is_empty());
        assert_eq!(
            retained_paired_stats(&retained),
            super::RetainedPairedStats::default(),
            "no target or dSpark byte accounting may survive whole-pair eviction"
        );
    }

    #[test]
    fn paired_session_bytes_plateau_at_count_cap_plus_one() {
        use super::{RetainedKv, retained_paired_stats, stash_into};
        use std::collections::HashMap;

        let mut retained: HashMap<u64, RetainedKv> = HashMap::new();
        let sample = paired_retained_state(1);
        let (target_bytes, dflash_bytes) = sample.paired_estimated_bytes().unwrap();
        assert!(target_bytes > 0);
        assert!(dflash_bytes > 0);
        stash_into(&mut retained, 1, sample, 2, 1);
        stash_into(&mut retained, 2, paired_retained_state(1), 2, 1);
        let evicted = stash_into(&mut retained, 3, paired_retained_state(1), 2, 1);

        let plateau = retained_paired_stats(&retained);
        assert_eq!(evicted, 1);
        assert_eq!(retained.len(), 2);
        assert_eq!(plateau.entries, 2);
        assert_eq!(plateau.target_bytes, target_bytes * 2);
        assert_eq!(plateau.dflash_bytes, dflash_bytes * 2);
    }

    /// Write a config.json file into the given directory with the provided JSON content.
    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).unwrap();
    }

    /// The streaming DFlash path must produce byte-for-byte the same text as the
    /// non-streaming DFlash path — only delivery differs — and the `speculation`
    /// selector must route correctly. Greedy + a low-entropy counting prompt so
    /// the exact decoders agree. Manual:
    ///
    /// ```text
    /// HIGGS_DFLASH_TARGET_DIR=<target dir> HIGGS_DFLASH_DRAFTER_DIR=<drafter dir> \
    /// cargo test -p higgs-engine dflash_streaming_matches_nonstreaming -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "loads real DFlash target + drafter; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
    fn dflash_streaming_matches_nonstreaming() {
        use super::{SimpleEngine, StreamingOutput};
        use crate::chat_template::ChatMessage;
        use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
        use higgs_models::{SamplingParams, Speculation, turboquant::KvCacheConfig};

        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .try_init();

        let (Ok(target), Ok(drafter)) = (
            std::env::var("HIGGS_DFLASH_TARGET_DIR"),
            std::env::var("HIGGS_DFLASH_DRAFTER_DIR"),
        ) else {
            println!(
                "skipping dflash_streaming_matches_nonstreaming: set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"
            );
            return;
        };

        let tuning =
            MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
        let engine = SimpleEngine::load_with_dflash(
            &target,
            KvCacheConfig::default(),
            tuning,
            false,
            Some(Path::new(&drafter)),
            None,
        )
        .expect("load DFlash engine");

        let messages = [ChatMessage {
            role: "user".to_owned(),
            content: "Count from 1 to 40. Print only the numbers separated by commas.".to_owned(),
            tool_calls: None,
        }];
        // Thinking OFF: compare the raw counting stream.
        let toks = engine
            .prepare_chat_prompt_with_thinking(&messages, None, false)
            .expect("chat prompt");
        let max_tokens = 96;
        let params = |spec| SamplingParams {
            temperature: 0.0,
            speculation: spec,
            ..SamplingParams::default()
        };
        // Make DFlash deterministic for an exact streaming-vs-buffered diff. The
        // realized-speedup gate is wall-clock dependent (cold vs warm kernels
        // floor to AR differently → different block overshoot at the length
        // limit), so two *separate* gated runs can differ in length. Gate OFF +
        // a fixed block removes that timing dependence; both runs then follow
        // the identical trajectory and must match byte-for-byte.
        dflash_set_fixed_block_env(16);

        // speculation=mtp must reach the MTP head even though a drafter is
        // loaded. `dflash_decode` clears `last_dflash_accepts` on entry and is
        // the only writer; running MTP first leaves it empty, proving the
        // DFlash loop was not entered.
        let mtp = engine
            .generate_with_thinking(
                &toks,
                max_tokens,
                &params(Speculation::Mtp),
                &[],
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .expect("mtp");
        assert!(!mtp.text.is_empty(), "speculation=mtp produced output");
        assert!(
            engine.last_dflash_accepts().is_empty(),
            "speculation=mtp must not enter the DFlash loop"
        );

        // Non-streaming DFlash.
        let non_stream = engine
            .generate_with_thinking(
                &toks,
                max_tokens,
                &params(Speculation::DFlash),
                &[],
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .expect("non-stream dflash");
        assert!(
            !engine.last_dflash_accepts().is_empty(),
            "speculation=dflash must drive the DFlash loop"
        );

        // Streaming DFlash. Capacity exceeds the chunk count, so the engine's
        // blocking_send never blocks without a concurrent receiver.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamingOutput>(8192);
        engine
            .generate_streaming_with_thinking(
                &toks,
                max_tokens,
                &params(Speculation::DFlash),
                &[],
                false,
                None,
                &tx,
                false,
                false,
                None,
                None,
                None,
            )
            .expect("stream dflash");
        drop(tx);
        let mut streamed = String::new();
        let mut saw_finished = false;
        while let Ok(chunk) = rx.try_recv() {
            streamed.push_str(&chunk.new_text);
            saw_finished |= chunk.finished;
        }
        assert!(
            saw_finished,
            "streaming must emit a terminal finished chunk"
        );
        assert_eq!(
            non_stream.text, streamed,
            "streaming DFlash must equal non-streaming DFlash byte-for-byte"
        );

        // Exact greedy decoders agree on token values for this deterministic
        // prompt; lengths may differ by block overshoot, so check prefix.
        assert!(
            dflash_prefix_consistent(&non_stream.text, &mtp.text),
            "MTP and DFlash must agree on the counting sequence:\n dflash={:?}\n mtp={:?}",
            non_stream.text,
            mtp.text
        );
    }

    /// P5: `DFlash` speculative decode must be byte-identical to plain AR greedy
    /// decode (T=0) and reports acceptance length + tok/s. Loads one paired
    /// target/drafter engine, warms both request paths, then alternates plain AR
    /// and `DFlash` in a palindromic order to reduce load and thermal bias.
    /// Manual:
    ///
    /// ```text
    /// HIGGS_DFLASH_TARGET_DIR=<target dir> \
    /// HIGGS_DFLASH_DRAFTER_DIR=<full-Q4 dSpark MLX dir> \
    /// cargo test -p higgs-engine --release \
    ///   dflash_matches_ar_greedy_and_reports_accept_len -- \
    ///   --ignored --exact --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "loads real Bonsai target + full-Q4 dSpark drafter; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
    fn dflash_matches_ar_greedy_and_reports_accept_len() {
        use super::{GenerationPhaseTiming, SimpleEngine};
        use crate::chat_template::ChatMessage;
        use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
        use higgs_models::{SamplingParams, Speculation, turboquant::KvCacheConfig};

        // Surface the engine's accept_len / per-iter trace logs.
        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .try_init();

        let target = std::env::var("HIGGS_DFLASH_TARGET_DIR")
            .expect("set HIGGS_DFLASH_TARGET_DIR to the Bonsai-27B target model dir");
        let drafter = std::env::var("HIGGS_DFLASH_DRAFTER_DIR")
            .expect("set HIGGS_DFLASH_DRAFTER_DIR to the full-Q4 dSpark MLX dir");
        dflash_assert_benchmark_power_state("preflight");

        // Own every performance-sensitive policy knob inside the ignored
        // single-test process. In particular, disabling prefix caching makes
        // AR use the same one-shot prompt schedule as DFlash and excludes
        // cache snapshot/store work from the ppN+tg128 wall measurement.
        // Pinning row4 here prevents a nominal "reference" run from silently
        // measuring the pre-TG-LUT verifier when the shell command omits flags.
        let benchmark_draft_cap = std::env::var("HIGGS_TEST_DRAFT_CAP")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .filter(|value| (1..=4).contains(value))
            .unwrap_or(4);
        dflash_set_test_env("HIGGS_DFLASH_VERIFY_MODE", "block");
        dflash_set_test_env("HIGGS_DFLASH_GATE", "0");
        dflash_set_test_env("HIGGS_DSPARK_DRAFT_CAP", &benchmark_draft_cap.to_string());
        dflash_set_test_env("HIGGS_DSPARK_TARGET_HEAD", "0");
        dflash_set_test_env("HIGGS_PREFIX_CACHE", "0");
        dflash_set_test_env("HIGGS_BONSAI_TG_LUT4", "1");
        dflash_set_test_env("HIGGS_BONSAI_TG_LUT4_FUSED_MLP", "0");
        dflash_set_test_env("HIGGS_BONSAI_TG_LUT4_M5_WG", "256");
        dflash_set_test_env("HIGGS_DFLASH_FUSED_CONV", "0");
        dflash_set_test_env("HIGGS_DFLASH_GDN_CONFIG_CACHE", "0");
        dflash_set_test_env("HIGGS_BONSAI_ALIGNED_FAST_QMV", "0");

        // Use a chat-templated code/reasoning prompt (dSpark is trained and
        // measured on tasks such as GSM8K/HumanEval/MT-Bench, not raw factual
        // completions). The block verifier's proven request boundary is
        // greedy, no-thinking, no penalties, with the realized-speed gate off.
        let user_prompt = std::env::var("HIGGS_TEST_PROMPT").unwrap_or_else(|_| {
            "Write a Python function that returns the n-th Fibonacci number iteratively.".to_owned()
        });
        let user_prompt = user_prompt.as_str();
        // This throughput harness deliberately rejects thinking mode: forced
        // thinking tokens make the request fall back to canonical S=1.
        let enable_thinking = std::env::var("HIGGS_TEST_THINKING")
            .map(|v| v != "0")
            .unwrap_or(false);
        assert!(
            !enable_thinking,
            "full-Q4 block benchmark requires HIGGS_TEST_THINKING=0"
        );
        let max_tokens = std::env::var("HIGGS_TEST_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(128);
        assert_eq!(max_tokens, 128, "paired benchmark is the tg128 workload");
        assert!(
            std::env::var("HIGGS_DFLASH_GATE").is_ok_and(|value| value == "0"),
            "raw block benchmark requires HIGGS_DFLASH_GATE=0"
        );
        let params = |speculation| SamplingParams {
            temperature: 0.0,
            speculation,
            ..SamplingParams::default()
        };

        let tuning =
            MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
        let engine = SimpleEngine::load_with_dflash(
            &target,
            KvCacheConfig::default(),
            tuning,
            false,
            Some(Path::new(&drafter)),
            None,
        )
        .expect("load paired target + drafter");
        let dflash = engine.dflash.as_ref().expect("DFlash state");
        assert!(
            dflash.is_dspark,
            "paired benchmark requires a dSpark sidecar"
        );
        assert!(
            !dflash.dspark_target_head,
            "final reference benchmark requires the drafter's frozen Q4 head"
        );
        assert_eq!(dflash.block_size, 4, "Bonsai dSpark is trained for block 4");
        assert_eq!(
            dflash.draft_cap, benchmark_draft_cap,
            "loaded draft cap must match the requested benchmark experiment"
        );
        assert_eq!(
            dflash.verify_mode,
            DFlashVerifyMode::BatchedTape,
            "final throughput benchmark requires the block verifier"
        );
        let drafter_quant = {
            let drafter = lock_or_recover(&dflash.drafter);
            drafter
                .config
                .quantization
                .clone()
                .expect("full-Q4 drafter must declare quantization")
        };
        assert_eq!(drafter_quant.bits, 4, "frozen drafter head must be Q4");
        assert_eq!(
            drafter_quant.group_size, 32,
            "frozen drafter head must use group size 32"
        );
        assert_eq!(
            drafter_quant.mode,
            higgs_models::quant_mode::QuantMode::Affine,
            "frozen drafter head must use affine Q4"
        );

        let messages = [ChatMessage {
            role: "user".to_owned(),
            content: user_prompt.to_owned(),
            tool_calls: None,
        }];
        let toks = engine
            .prepare_chat_prompt_with_thinking(&messages, None, enable_thinking)
            .expect("chat prompt");
        println!(
            "paired benchmark: prompt_tokens={} output_tokens={} full_q4=true block={} draft_cap={}",
            toks.len(),
            max_tokens,
            dflash.block_size,
            dflash.draft_cap
        );

        let run = |speculation: Speculation,
                   output_tokens: u32|
         -> (String, f64, GenerationPhaseTiming) {
            // Benchmark a fresh pp+tg request on both paths. Plain AR normally
            // benefits from the engine's in-memory radix cache on repeats,
            // whereas DFlash owns a separate shared target/drafter prefill and
            // does not consult that cache; retaining it here would compare a
            // five-token AR suffix against a full DFlash prompt.
            engine.clear_prefix_cache();
            let t = std::time::Instant::now();
            let out = engine
                .generate_with_thinking(
                    &toks,
                    output_tokens,
                    &params(speculation),
                    &[],
                    false,
                    None,
                    enable_thinking,
                    None,
                    None,
                    None,
                )
                .expect("generate");
            assert_eq!(
                out.completion_tokens, output_tokens,
                "tg workload ended before its requested output length"
            );
            assert_eq!(
                out.finish_reason, "length",
                "tg workload must end by length"
            );
            let elapsed = t.elapsed().as_secs_f64();
            let phase_timing = match speculation {
                Speculation::None => engine
                    .last_ar_timing()
                    .expect("plain AR request must publish phase timing"),
                Speculation::DFlash => engine
                    .last_dflash_timing()
                    .expect("DFlash request must publish phase timing"),
                Speculation::Auto | Speculation::Mtp => unreachable!(),
            };
            assert_eq!(
                phase_timing.decode_tokens,
                output_tokens.saturating_sub(1),
                "prefill owns token one and decode must own the remainder"
            );
            (out.text, elapsed, phase_timing)
        };

        let warmup_tokens = std::env::var("HIGGS_TEST_WARMUP_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8);
        if warmup_tokens > 0 {
            let _ = run(Speculation::None, warmup_tokens);
            let _ = run(Speculation::DFlash, warmup_tokens);
        }

        let mut ar_text: Option<String> = None;
        let mut ar_wall_secs = Vec::with_capacity(2);
        let mut df_wall_secs = Vec::with_capacity(2);
        let mut ar_phase_timings = Vec::with_capacity(2);
        let mut df_phase_timings = Vec::with_capacity(2);
        let mut df_matched_drafts = 0_u64;
        let mut df_drafted_tokens = 0_u64;
        // AR, DFlash, DFlash, AR cancels a first-order thermal trend while
        // keeping one model load and fresh request-local caches for every run.
        for (index, speculation) in [
            Speculation::None,
            Speculation::DFlash,
            Speculation::DFlash,
            Speculation::None,
        ]
        .into_iter()
        .enumerate()
        {
            let (text, elapsed, phase_timing) = run(speculation, max_tokens);
            let tps = f64::from(max_tokens) / elapsed;
            println!("run={} mode={speculation:?} wall_tok_s={tps:.2}", index + 1);
            println!(
                "run={} mode={speculation:?} prefill_ms={:.1} decode_ms={:.1} request_ms={:.1} decode_tok_s={:.2}",
                index + 1,
                phase_timing.prefill.as_secs_f64() * 1e3,
                phase_timing.decode.as_secs_f64() * 1e3,
                phase_timing.request.as_secs_f64() * 1e3,
                f64::from(phase_timing.decode_tokens) / phase_timing.decode.as_secs_f64()
            );
            match speculation {
                Speculation::None => {
                    if let Some(reference) = &ar_text {
                        assert_eq!(&text, reference, "AR repeat changed greedy output");
                    } else {
                        ar_text = Some(text.clone());
                    }
                    ar_wall_secs.push(elapsed);
                    ar_phase_timings.push(phase_timing);
                }
                Speculation::DFlash => {
                    assert_eq!(
                        Some(&text),
                        ar_text.as_ref(),
                        "DFlash diverged from AR greedy or crossed max_tokens"
                    );
                    let accepts = engine.last_dflash_accepts();
                    let matches = engine.last_dflash_draft_matches();
                    let draft_counts = engine.last_dflash_draft_counts();
                    assert_eq!(
                        accepts.len(),
                        matches.len(),
                        "each spec round needs emitted and matched-draft telemetry"
                    );
                    assert_eq!(
                        accepts.len(),
                        draft_counts.len(),
                        "each spec round needs its actual proposal width"
                    );
                    let emitted: u32 = accepts.iter().copied().sum();
                    let matched: u32 = matches.iter().copied().sum();
                    assert_eq!(
                        emitted,
                        max_tokens - 1,
                        "prefill emits token one; every remaining tg token must come from spec rounds"
                    );
                    let rounds = u32::try_from(accepts.len()).expect("round count fits u32");
                    let drafted: u32 = draft_counts.iter().copied().sum();
                    df_matched_drafts += u64::from(matched);
                    df_drafted_tokens += u64::from(drafted);
                    println!("DFlash accepts: {accepts:?}");
                    println!("DFlash draft matches: {matches:?}");
                    println!("DFlash draft counts: {draft_counts:?}");
                    println!(
                        "DFlash tau={:.3} exact_match_rate={:.2}%",
                        f64::from(emitted) / f64::from(rounds),
                        100.0 * f64::from(matched) / f64::from(drafted)
                    );
                    df_wall_secs.push(elapsed);
                    df_phase_timings.push(phase_timing);
                }
                Speculation::Auto | Speculation::Mtp => unreachable!(),
            }
        }
        dflash_assert_benchmark_power_state("postflight");

        let aggregate_wall_tps = |samples: &[f64]| {
            f64::from(max_tokens) * samples.len() as f64 / samples.iter().sum::<f64>()
        };
        let aggregate_decode_tps = |samples: &[GenerationPhaseTiming]| {
            let tokens: u32 = samples.iter().map(|sample| sample.decode_tokens).sum();
            let secs: f64 = samples
                .iter()
                .map(|sample| sample.decode.as_secs_f64())
                .sum();
            f64::from(tokens) / secs
        };
        let aggregate_request_tps = |samples: &[GenerationPhaseTiming]| {
            f64::from(max_tokens) * samples.len() as f64
                / samples
                    .iter()
                    .map(|sample| sample.request.as_secs_f64())
                    .sum::<f64>()
        };
        let mean_prefill_ms = |samples: &[GenerationPhaseTiming]| {
            samples
                .iter()
                .map(|sample| sample.prefill.as_secs_f64())
                .sum::<f64>()
                * 1e3
                / samples.len() as f64
        };
        let relative_drift = |first: f64, last: f64| {
            let midpoint = (first + last) * 0.5;
            if midpoint > 0.0 {
                (last - first).abs() / midpoint
            } else {
                0.0
            }
        };
        let ar_first = ar_phase_timings.first().expect("first AR timing");
        let ar_last = ar_phase_timings.last().expect("last AR timing");
        let ar_decode_drift = relative_drift(
            ar_first.decode.as_secs_f64() / f64::from(ar_first.decode_tokens),
            ar_last.decode.as_secs_f64() / f64::from(ar_last.decode_tokens),
        );
        let ar_wall_drift = relative_drift(
            *ar_wall_secs.first().expect("first AR wall sample"),
            *ar_wall_secs.last().expect("last AR wall sample"),
        );
        let ar_endpoint_drift = ar_decode_drift.max(ar_wall_drift);
        println!(
            "AR endpoint drift: decode={:.2}% wall={:.2}% gate=3.00%",
            ar_decode_drift * 100.0,
            ar_wall_drift * 100.0
        );
        assert!(
            ar_endpoint_drift <= 0.03,
            "AR endpoint drift {:.2}% exceeds the 3% thermal-stability gate",
            ar_endpoint_drift * 100.0
        );

        let ar_tps = aggregate_wall_tps(&ar_wall_secs);
        let df_tps = aggregate_wall_tps(&df_wall_secs);
        let ar_decode_tps = aggregate_decode_tps(&ar_phase_timings);
        let df_decode_tps = aggregate_decode_tps(&df_phase_timings);
        println!(
            "AR aggregate: wall={ar_tps:.2} decode={ar_decode_tps:.2} request={:.2} tok/s prefill_mean={:.1} ms",
            aggregate_request_tps(&ar_phase_timings),
            mean_prefill_ms(&ar_phase_timings)
        );
        println!(
            "DFlash aggregate: wall={df_tps:.2} decode={df_decode_tps:.2} request={:.2} tok/s prefill_mean={:.1} ms",
            aggregate_request_tps(&df_phase_timings),
            mean_prefill_ms(&df_phase_timings)
        );
        println!("wall speedup vs AR: {:.3}x", df_tps / ar_tps);
        println!(
            "decode speedup vs AR: {:.3}x",
            df_decode_tps / ar_decode_tps
        );

        // The established Bonsai battery result is 1.435x decode throughput
        // versus AR (20.42 -> 29.31 tok/s). Preserve that demonstrated speedup
        // with a 3% relative tolerance, rather than pinning machine-specific
        // absolute rates. Acceptance uses the Prism-style matched/drafted ratio.
        const REFERENCE_DECODE_SPEEDUP: f64 = 1.435;
        const MAX_REFERENCE_TOLERANCE: f64 = 0.03;
        const MIN_ACCEPTANCE_RATE: f64 = 0.87;
        const MIN_DECODE_SPEEDUP: f64 = REFERENCE_DECODE_SPEEDUP * (1.0 - MAX_REFERENCE_TOLERANCE);
        assert!(
            df_drafted_tokens > 0,
            "the dSpark acceptance gate requires drafted tokens"
        );
        let aggregate_acceptance = df_matched_drafts as f64 / df_drafted_tokens as f64;
        let decode_speedup = df_decode_tps / ar_decode_tps;
        println!(
            "dSpark release gates: acceptance={:.2}% floor={:.2}% decode_speedup={decode_speedup:.5}x floor={MIN_DECODE_SPEEDUP:.5}x",
            aggregate_acceptance * 100.0,
            MIN_ACCEPTANCE_RATE * 100.0,
        );
        assert!(
            aggregate_acceptance >= MIN_ACCEPTANCE_RATE,
            "aggregate dSpark acceptance {:.2}% is below the {:.2}% release floor",
            aggregate_acceptance * 100.0,
            MIN_ACCEPTANCE_RATE * 100.0
        );
        assert!(
            decode_speedup >= MIN_DECODE_SPEEDUP,
            "dSpark decode speedup {decode_speedup:.5}x is below the \
             {MIN_DECODE_SPEEDUP:.5}x release floor derived from the demonstrated \
             {REFERENCE_DECODE_SPEEDUP:.3}x battery result"
        );

        // Correctness gate: speculative decode must obey the same output bound
        // and produce exactly the same greedy text as autoregressive decode.
        assert!(
            ar_text.is_some_and(|text| !text.is_empty()),
            "empty generation"
        );
    }

    struct DFlashArMetrics {
        h_bits: f64,
        top1_prob: f64,
        tokens: Vec<u32>,
        text: String,
    }

    struct DFlashRunMetrics {
        text: String,
        accept_mean: f64,
        p10: u32,
        p50: u32,
        p90: u32,
        accept_frac: f64,
    }

    struct DFlashSweepRow {
        task: String,
        h_bits: f64,
        top1_prob: f64,
        accept_mean: f64,
        p10: u32,
        p50: u32,
        p90: u32,
        accept_frac: f64,
        byte_exact: bool,
    }

    #[allow(unsafe_code)]
    fn dflash_set_test_env(key: &str, value: &str) {
        // SAFETY: This ignored manual harness mutates DFlash env knobs before
        // model loading/generation and is intended to be run alone.
        unsafe { std::env::set_var(key, value) };
    }

    fn dflash_assert_benchmark_power_state(stage: &str) {
        #[cfg(target_os = "macos")]
        {
            if std::env::var("HIGGS_TEST_REQUIRE_AC").is_ok_and(|value| value == "0") {
                return;
            }

            let pmset = std::process::Command::new("pmset")
                .args(["-g", "batt"])
                .output()
                .expect("run pmset for dSpark benchmark power preflight");
            assert!(
                pmset.status.success(),
                "dSpark benchmark {stage}: pmset failed with {}",
                pmset.status
            );
            let pmset_stdout = String::from_utf8_lossy(&pmset.stdout);
            assert!(
                pmset_stdout.contains("AC Power"),
                "dSpark benchmark {stage} requires AC power; pmset reported: {pmset_stdout}"
            );

            let minimum_percent = std::env::var("HIGGS_TEST_MIN_BATTERY_PERCENT")
                .ok()
                .and_then(|value| value.parse::<u8>().ok())
                .unwrap_or(20);
            let battery_percent = pmset_stdout.split_whitespace().find_map(|field| {
                let percent_index = field.find('%')?;
                field.get(..percent_index)?.parse::<u8>().ok()
            });
            if let Some(percent) = battery_percent {
                assert!(
                    percent >= minimum_percent,
                    "dSpark benchmark {stage} requires battery >= {minimum_percent}%, got {percent}%"
                );
            }

            let ioreg = std::process::Command::new("ioreg")
                .args(["-rn", "AppleSmartBattery"])
                .output()
                .expect("run ioreg for dSpark benchmark charger preflight");
            assert!(
                ioreg.status.success(),
                "dSpark benchmark {stage}: ioreg failed with {}",
                ioreg.status
            );
            let ioreg_stdout = String::from_utf8_lossy(&ioreg.stdout);
            if ioreg_stdout.contains("AppleSmartBattery") {
                assert!(
                    ioreg_stdout.contains("\"ExternalConnected\" = Yes"),
                    "dSpark benchmark {stage}: charger is detected but not delivering external power"
                );
            }
        }
    }

    fn dflash_set_fixed_block_env(block_size: u32) {
        dflash_set_test_env("HIGGS_DFLASH_GATE", "0");
        dflash_set_test_env("HIGGS_DFLASH_ADAPTIVE", "0");
        dflash_set_test_env("HIGGS_DFLASH_BLOCK_SIZE", &block_size.to_string());
    }

    fn dflash_chat_tokens(engine: &SimpleEngine, prompt: &str) -> Vec<u32> {
        let messages = [ChatMessage {
            role: "user".to_owned(),
            content: prompt.to_owned(),
            tool_calls: None,
        }];
        engine
            .prepare_chat_prompt_with_thinking(&messages, None, false)
            .expect("chat prompt")
    }

    fn dflash_entropy_and_top1(row: &[f32]) -> (u32, f64, f64) {
        assert!(!row.is_empty(), "empty logits row");
        let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let max_logit = f64::from(ranked[0].1);
        let denom = row
            .iter()
            .map(|&v| (f64::from(v) - max_logit).exp())
            .sum::<f64>();
        let denom = denom.max(f64::MIN_POSITIVE);
        let top_n = ranked.len().min(50);
        let top_probs: Vec<f64> = ranked
            .iter()
            .take(top_n)
            .map(|&(_, v)| (f64::from(v) - max_logit).exp() / denom)
            .collect();
        let top_mass = top_probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let entropy = top_probs
            .iter()
            .map(|&p| {
                let q = p / top_mass;
                if q > 0.0 { -q * q.log2() } else { 0.0 }
            })
            .sum::<f64>();
        let top1_prob = top_probs.first().copied().unwrap_or(0.0);
        let token = u32::try_from(ranked[0].0).expect("vocab id overflow");
        (token, entropy, top1_prob)
    }

    fn dflash_ar_entropy_pass(
        engine: &SimpleEngine,
        prompt: &str,
        max_tokens: u32,
    ) -> DFlashArMetrics {
        let prompt_tokens = dflash_chat_tokens(engine, prompt);
        let mut model = lock_or_recover(&engine.model);
        let mut cache = model
            .make_cache_with_config(engine.kv_cache_config)
            .expect("cache");
        let prompt_array = Array::from(prompt_tokens.as_slice()).index(NewAxis);
        let seq_len = prompt_array.shape().get(1).copied().unwrap_or(0);
        let mut logits = if seq_len > engine.tuning.chunked_prefill_threshold() {
            model
                .forward_chunked(
                    &prompt_array,
                    &mut cache,
                    engine.tuning.chunked_prefill_chunk_size(),
                )
                .expect("chunked prefill")
        } else {
            model
                .forward_last_token(&prompt_array, None, &mut cache)
                .expect("prefill")
        };

        let max_tokens_usize = usize::try_from(max_tokens).expect("max_tokens overflow");
        let mut tokens = Vec::with_capacity(max_tokens_usize);
        let mut entropy_sum = 0.0_f64;
        let mut top1_sum = 0.0_f64;

        for step in 0..max_tokens_usize {
            let row = logits
                .index((.., -1, ..))
                .reshape(&[-1])
                .expect("reshape logits")
                .as_dtype(Dtype::Float32)
                .expect("f32 logits");
            eval([&row]).expect("eval logits");
            let (next_token, entropy, top1_prob) = dflash_entropy_and_top1(row.as_slice::<f32>());
            entropy_sum += entropy;
            top1_sum += top1_prob;
            tokens.push(next_token);

            if step + 1 < max_tokens_usize {
                let next_token_i32 = i32::try_from(next_token).expect("token id overflow");
                let single = Array::from_slice(&[next_token_i32], &[1, 1]);
                logits = model
                    .forward(&single, None, &mut cache)
                    .expect("decode step");
            }
        }
        drop(model);

        let text = engine.decode_tokens(&tokens).expect("decode AR tokens");
        let denom = f64::from(max_tokens);
        DFlashArMetrics {
            h_bits: entropy_sum / denom,
            top1_prob: top1_sum / denom,
            tokens,
            text,
        }
    }

    fn dflash_accept_metrics(accepts: &[u32], block_size: u32) -> (f64, u32, u32, u32, f64) {
        if accepts.is_empty() {
            return (0.0, 0, 0, 0, 0.0);
        }
        let mean = accepts.iter().map(|&n| f64::from(n)).sum::<f64>() / accepts.len() as f64;
        let mut sorted = accepts.to_vec();
        sorted.sort_unstable();
        let last = sorted.len() - 1;
        let p10 = sorted[last * 10 / 100];
        let p50 = sorted[last * 50 / 100];
        let p90 = sorted[last * 90 / 100];
        // Every emitted speculative round contains one target correction (or
        // the full-match bonus). Only the remaining rows are accepted draft
        // tokens. Subtract that mandatory target token before reporting the
        // Prism-style draft acceptance fraction.
        let matched_drafts = accepts
            .iter()
            .map(|&emitted| f64::from(emitted.saturating_sub(1).min(block_size)))
            .sum::<f64>()
            / accepts.len() as f64;
        let accept_frac = matched_drafts / f64::from(block_size.max(1));
        (mean, p10, p50, p90, accept_frac)
    }

    #[test]
    fn dspark_accept_fraction_excludes_correction_or_bonus_token() {
        let (mean, _, _, _, fraction) = dflash_accept_metrics(&[5, 4], 4);
        assert!((mean - 4.5).abs() < f64::EPSILON);
        assert!((fraction - 0.875).abs() < f64::EPSILON);
    }

    fn dflash_generation_pass(
        engine: &SimpleEngine,
        prompt: &str,
        max_tokens: u32,
        block_size: u32,
        greedy: &SamplingParams,
    ) -> DFlashRunMetrics {
        let effective_block = engine
            .dflash
            .as_ref()
            .and_then(|state| u32::try_from(state.block_size).ok())
            .expect("DFlash engine missing a positive block size");
        assert_eq!(
            block_size, effective_block,
            "metrics must use the block size selected when the engine loaded"
        );
        let prompt_tokens = dflash_chat_tokens(engine, prompt);
        let out = engine
            .generate_with_thinking(
                &prompt_tokens,
                max_tokens,
                greedy,
                &[],
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .expect("DFlash generation");
        let accepts = engine.last_dflash_accepts();
        let (accept_mean, p10, p50, p90, accept_frac) = dflash_accept_metrics(&accepts, block_size);
        DFlashRunMetrics {
            text: out.text,
            accept_mean,
            p10,
            p50,
            p90,
            accept_frac,
        }
    }

    fn dflash_prefix_consistent(left: &str, right: &str) -> bool {
        if left.len() <= right.len() {
            right.starts_with(left)
        } else {
            left.starts_with(right)
        }
    }

    fn dflash_row_from_metrics(
        task: impl Into<String>,
        ar: &DFlashArMetrics,
        df: DFlashRunMetrics,
    ) -> DFlashSweepRow {
        DFlashSweepRow {
            task: task.into(),
            h_bits: ar.h_bits,
            top1_prob: ar.top1_prob,
            accept_mean: df.accept_mean,
            p10: df.p10,
            p50: df.p50,
            p90: df.p90,
            accept_frac: df.accept_frac,
            byte_exact: !ar.tokens.is_empty() && dflash_prefix_consistent(&ar.text, &df.text),
        }
    }

    fn dflash_sweep_row(
        engine: &SimpleEngine,
        task: impl Into<String>,
        prompt: &str,
        max_tokens: u32,
        block_size: u32,
        greedy: &SamplingParams,
    ) -> DFlashSweepRow {
        let ar = dflash_ar_entropy_pass(engine, prompt, max_tokens);
        let df = dflash_generation_pass(engine, prompt, max_tokens, block_size, greedy);
        dflash_row_from_metrics(task, &ar, df)
    }

    fn dflash_print_table(title: &str, rows: &[DFlashSweepRow]) {
        println!("\n{title}");
        println!(
            "| task | H_bits | top1_prob | accept_mean | p10 | p50 | p90 | accept_frac | byte_exact |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
        for row in rows {
            println!(
                "| {} | {:.3} | {:.3} | {:.2} | {} | {} | {} | {:.3} | {} |",
                row.task,
                row.h_bits,
                row.top1_prob,
                row.accept_mean,
                row.p10,
                row.p50,
                row.p90,
                row.accept_frac,
                row.byte_exact
            );
        }
    }

    fn dflash_repeat_prompt_to_chat_tokens(
        engine: &SimpleEngine,
        prompt: &str,
        target_tokens: usize,
    ) -> String {
        let base_len = dflash_chat_tokens(engine, prompt).len().max(1);
        let reps = (target_tokens / base_len).max(1);
        let mut padded = std::iter::repeat_n(prompt, reps)
            .collect::<Vec<_>>()
            .join("\n\n");
        while dflash_chat_tokens(engine, &padded).len() < target_tokens {
            padded.push_str("\n\n");
            padded.push_str(prompt);
        }
        padded
    }

    #[test]
    #[ignore = "loads real DFlash target + drafter; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::as_conversions
    )]
    fn dflash_entropy_sweep() {
        use crate::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
        use higgs_models::turboquant::KvCacheConfig;

        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .try_init();

        let (target, drafter) = match (
            std::env::var("HIGGS_DFLASH_TARGET_DIR"),
            std::env::var("HIGGS_DFLASH_DRAFTER_DIR"),
        ) {
            (Ok(target), Ok(drafter)) => (target, drafter),
            _ => {
                println!(
                    "skipping dflash_entropy_sweep: set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"
                );
                return;
            }
        };

        const MAX_TOKENS: u32 = 160;
        const DEFAULT_BLOCK_SIZE: u32 = 16;
        const MULT_TABLES: &str = "Print the multiplication tables from 1 x 1 through 12 x 12 in ascending order. Use one equation per line.";
        const COUNT_200: &str = "Count from 1 to 200. Print only the numbers separated by commas.";
        const JSON_RECORDS: &str = "Emit 24 JSON objects, one per line, with schema {\"id\": number, \"name\": \"item-N\", \"active\": true, \"score\": number}. Use deterministic ascending ids.";
        const CSV_TABLE: &str = "Create a CSV table with columns day,city,temperature_c,condition for 40 rows. Use predictable city names City01 through City40.";
        const STRUCT_GETTERS: &str = "Write repetitive Rust code for a UserProfile struct with fields id, name, email, age, city, country, plan, created_at, and add simple getter methods for every field.";
        const SORT_ALGORITHM: &str = "Write an iterative insertion sort implementation in Python, then walk through sorting [9, 4, 7, 1, 3, 8] step by step.";
        const CAPITALS: &str = "List the capitals of these countries in order: France, Germany, Italy, Spain, Portugal, Netherlands, Belgium, Austria, Poland, Czechia, Denmark, Sweden, Norway, Finland, Ireland, Greece, Turkey, Egypt, Japan, Canada.";
        const UNIT_CONVERSIONS: &str = "Create a unit conversion table for meters to feet from 1 through 40 meters. Use four decimal places.";
        const TRANSLATION: &str = "Translate this paragraph from English to French: The research team tested a compact solar pump in three villages. During the dry season, farmers used it to irrigate tomato fields, compare fuel savings, and record maintenance issues.";
        const GSM8K: &str = "Solve this word problem with clear arithmetic steps: A bakery made 186 muffins. It sold 48 before lunch, baked 3 more trays with 24 muffins each, then packed the rest equally into 7 boxes. How many muffins were in each box, and how many were left over?";
        const PHOTOSYNTHESIS: &str = "Explain photosynthesis for a technically curious reader. Cover chlorophyll, light reactions, carbon fixation, water splitting, and why the process matters to ecosystems.";
        const STORY: &str = "Write a vivid short story about an archivist who discovers that a city map changes every midnight. Include sensory detail and an unresolved final image.";

        const PROMPTS: [(&str, &str); 12] = [
            ("multiplication tables", MULT_TABLES),
            ("count 1..200", COUNT_200),
            ("fixed-schema JSON", JSON_RECORDS),
            ("CSV table", CSV_TABLE),
            ("struct getters", STRUCT_GETTERS),
            ("iterative sort", SORT_ALGORITHM),
            ("capitals list", CAPITALS),
            ("unit conversion", UNIT_CONVERSIONS),
            ("EN-FR translation", TRANSLATION),
            ("GSM8K word problem", GSM8K),
            ("photosynthesis", PHOTOSYNTHESIS),
            ("short story", STORY),
        ];

        let greedy = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };

        let load_engine = |block_size: u32| -> SimpleEngine {
            dflash_set_fixed_block_env(block_size);
            let tuning =
                MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
            SimpleEngine::load_with_dflash(
                &target,
                KvCacheConfig::default(),
                tuning,
                false,
                Some(Path::new(&drafter)),
                None,
            )
            .expect("load DFlash engine")
        };

        let engine = load_engine(DEFAULT_BLOCK_SIZE);
        let effective_block = engine
            .dflash
            .as_ref()
            .and_then(|state| u32::try_from(state.block_size).ok())
            .expect("loaded DFlash state");
        let is_dspark = engine.dflash.as_ref().is_some_and(|state| state.is_dspark);

        let mut entropy_rows = Vec::new();
        for &(label, prompt) in &PROMPTS {
            entropy_rows.push(dflash_sweep_row(
                &engine,
                label,
                prompt,
                MAX_TOKENS,
                effective_block,
                &greedy,
            ));
        }
        entropy_rows.sort_by(|a, b| a.h_bits.total_cmp(&b.h_bits));
        dflash_print_table("## ENTROPY", &entropy_rows);

        let mut context_rows = Vec::new();
        for &(label, prompt) in &[
            ("context multiplication", MULT_TABLES),
            ("context story", STORY),
        ] {
            for target_tokens in [512_usize, 4096, 16_384] {
                let padded = dflash_repeat_prompt_to_chat_tokens(&engine, prompt, target_tokens);
                context_rows.push(dflash_sweep_row(
                    &engine,
                    format!("{label}-{target_tokens}"),
                    &padded,
                    MAX_TOKENS,
                    effective_block,
                    &greedy,
                ));
            }
        }
        dflash_print_table("## CONTEXT", &context_rows);

        let sort_ar = dflash_ar_entropy_pass(&engine, SORT_ALGORITHM, MAX_TOKENS);
        drop(engine);

        let mut block_rows = Vec::new();
        let block_sizes = if is_dspark {
            // dSpark's log-SNR/Markov schedule is trained for one fixed block.
            // Pretending HIGGS_DFLASH_BLOCK_SIZE changed it produced duplicate
            // runs with false 8/16 labels and wrong utilization denominators.
            vec![effective_block]
        } else {
            vec![4_u32, 8, 16]
        };
        for block_size in block_sizes {
            let block_engine = load_engine(block_size);
            let actual_block = block_engine
                .dflash
                .as_ref()
                .and_then(|state| u32::try_from(state.block_size).ok())
                .expect("loaded DFlash block state");
            let df = dflash_generation_pass(
                &block_engine,
                SORT_ALGORITHM,
                MAX_TOKENS,
                actual_block,
                &greedy,
            );
            block_rows.push(dflash_row_from_metrics(
                format!("iterative sort block {actual_block}"),
                &sort_ar,
                df,
            ));
        }
        dflash_print_table("## BLOCK", &block_rows);
    }

    /// Gemma chat models stop on `<end_of_turn>`, which their `config.json` often
    /// omits from `eos_token_id`. It must be resolved from the tokenizer and added.
    #[test]
    fn extract_eos_tokens_adds_gemma_end_of_turn() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_type":"gemma3_text","eos_token_id":1}"#,
        );
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"106":{"content":"<end_of_turn>"}}}"#,
        )
        .unwrap();
        let ids = extract_eos_tokens(dir.path());
        assert!(ids.contains(&1), "base eos retained: {ids:?}");
        assert!(ids.contains(&106), "gemma <end_of_turn> added: {ids:?}");
    }

    /// `generation_config.json`'s `eos_token_id` (used by HF at inference) is unioned in.
    #[test]
    fn extract_eos_tokens_unions_generation_config() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type":"llama","eos_token_id":2}"#);
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id":[2,7]}"#,
        )
        .unwrap();
        let ids = extract_eos_tokens(dir.path());
        assert!(
            ids.contains(&2) && ids.contains(&7),
            "union with generation_config: {ids:?}"
        );
    }

    /// Non-Gemma models must not pull in `<end_of_turn>`.
    #[test]
    fn extract_eos_tokens_non_gemma_ignores_end_of_turn() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type":"llama","eos_token_id":2}"#);
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"106":{"content":"<end_of_turn>"}}}"#,
        )
        .unwrap();
        let ids = extract_eos_tokens(dir.path());
        assert_eq!(ids, vec![2], "non-gemma must not add end_of_turn: {ids:?}");
    }

    // --- detect_thinking_support tests ---

    /// MiniCPM5-style: a non-reasoning `model_type` (llama) but a chat template
    /// that exposes the `enable_thinking` switch ⇒ thinking-capable.
    #[test]
    fn detect_thinking_from_template_marker() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "llama"}"#);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{%- if enable_thinking %}<think>\n{%- endif %}",
        )
        .unwrap();
        assert!(detect_thinking_support(dir.path()));
    }

    /// Qwen3.5 reasoning `model_type` is detected even without a template file.
    #[test]
    fn detect_thinking_from_model_type() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "qwen3_5_moe"}"#);
        assert!(detect_thinking_support(dir.path()));
    }

    /// A plain Llama (no reasoning `model_type`, no `enable_thinking` in the
    /// template) must NOT be treated as a thinking model.
    #[test]
    fn no_thinking_for_plain_llama() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "llama"}"#);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{%- for m in messages %}{{ m.content }}{%- endfor %}",
        )
        .unwrap();
        assert!(!detect_thinking_support(dir.path()));
    }

    // --- derive_model_name tests ---

    #[test]
    fn test_derive_model_name_plain_directory() {
        let name = derive_model_name(Path::new("/home/user/models/Llama-3.2-1B"));
        assert_eq!(name, "Llama-3.2-1B");
    }

    #[test]
    fn test_derive_model_name_hf_cache_path() {
        let path = "/Users/me/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5";
        let name = derive_model_name(Path::new(path));
        assert_eq!(name, "mlx-community/Qwen3-Coder-Next-4bit");
    }

    #[test]
    fn test_derive_model_name_hf_cache_no_org() {
        let path = "/cache/models--MyModel/snapshots/abc123";
        let name = derive_model_name(Path::new(path));
        assert_eq!(name, "MyModel");
    }

    #[test]
    fn test_derive_model_name_relative_path() {
        let name = derive_model_name(Path::new("./my-model"));
        assert_eq!(name, "my-model");
    }

    #[test]
    fn adaptive_draft_depth_respects_configured_cap() {
        let mut depth = adaptive_draft_depth_for_cap(1);

        depth.observe(1, 1);

        assert_eq!(depth.current(), 1);
    }

    /// Create a temp dir, write config.json with the given content, and return
    /// the result of `extract_eos_tokens`.
    fn eos_from_config(json: &str) -> Vec<u32> {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), json);
        super::extract_eos_tokens(dir.path())
    }

    #[test]
    fn test_with_chat_terminator() {
        // VibeThinker case: config lists only <|endoftext|>; tokenizer adds <|im_end|>.
        assert_eq!(
            with_chat_terminator(vec![151643], Some(151645)),
            vec![151643, 151645]
        );
        // Already present (Qwen3.5/3.6): no duplicate.
        assert_eq!(
            with_chat_terminator(vec![151643, 151645], Some(151645)),
            vec![151643, 151645]
        );
        // Non-Qwen model with no <|im_end|>: unchanged.
        assert_eq!(with_chat_terminator(vec![2], None), vec![2]);
    }

    #[test]
    fn test_single_stop_sequence_found() {
        let result = check_stop_sequences("Hello world, goodbye!", &["goodbye".to_owned()]);
        assert_eq!(result, Some("Hello world, ".to_owned()));
    }

    #[test]
    fn test_no_stop_sequence_match() {
        let stops = vec!["goodbye".to_owned(), "farewell".to_owned()];
        assert!(check_stop_sequences("Hello world", &stops).is_none());
    }

    #[test]
    fn test_empty_stop_sequences_list() {
        assert!(check_stop_sequences("Hello world", &[]).is_none());
    }

    #[test]
    fn test_empty_text() {
        assert!(check_stop_sequences("", &["hello".to_owned()]).is_none());
    }

    #[test]
    fn test_stop_sequence_at_beginning() {
        let result = check_stop_sequences("STOP rest of text", &["STOP".to_owned()]);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_at_end() {
        let result = check_stop_sequences("Hello world END", &["END".to_owned()]);
        assert_eq!(result, Some("Hello world ".to_owned()));
    }

    fn assert_stop_sequence(text: &str, stops: &[&str], expected: &str) {
        let owned_stops: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let result = check_stop_sequences(text, &owned_stops);
        assert_eq!(result, Some(expected.to_owned()));
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins() {
        assert_stop_sequence("aaa bbb ccc ddd", &["ccc", "bbb"], "aaa ");
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins_reverse_order() {
        assert_stop_sequence("aaa bbb ccc ddd", &["bbb", "ccc"], "aaa ");
    }

    #[test]
    fn test_overlapping_stop_sequences_prefix() {
        // "ab" is a prefix of "abc". "ab" appears first at position 0.
        let stops = vec!["abc".to_owned(), "ab".to_owned()];
        assert_eq!(check_stop_sequences("abc def", &stops), Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_appears_multiple_times() {
        let result = check_stop_sequences("before stop middle stop after", &["stop".to_owned()]);
        assert_eq!(result, Some("before ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_is_entire_text() {
        assert_eq!(
            check_stop_sequences("STOP", &["STOP".to_owned()]),
            Some(String::new())
        );
    }

    #[test]
    fn test_stop_sequence_with_newlines() {
        let result = check_stop_sequences("line one\nline two\nline three", &["\n".to_owned()]);
        assert_eq!(result, Some("line one".to_owned()));
    }

    #[test]
    fn test_extract_eos_tokens_single_number() {
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": 151643}"#),
            vec![151_643]
        );
    }

    #[test]
    fn test_extract_eos_tokens_array() {
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": [151643, 151645]}"#),
            vec![151_643, 151_645]
        );
    }

    #[test]
    fn test_extract_eos_tokens_missing_field() {
        assert!(eos_from_config(r#"{"model_type": "qwen2"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_unexpected_type() {
        assert!(eos_from_config(r#"{"eos_token_id": "string"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_missing_config_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(super::extract_eos_tokens(dir.path()).is_empty());
    }

    // -- Additional check_stop_sequences edge cases --

    #[test]
    fn test_stop_sequence_substring_of_another() {
        assert_stop_sequence("Hello stop_now world", &["stop_now", "stop"], "Hello ");
    }

    #[test]
    fn test_stop_sequence_unicode() {
        let stops = vec!["\u{1F600}".to_owned()];
        assert!(check_stop_sequences("Hello world, a]b stop here", &stops).is_none());

        let result = check_stop_sequences("Hello \u{1F600} world", &stops);
        assert_eq!(result, Some("Hello ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_unicode_multibyte() {
        let stops = vec!["arr\u{00EA}t".to_owned()];
        let result = check_stop_sequences("Bonjour le monde, arr\u{00EA}t ici", &stops);
        assert_eq!(result, Some("Bonjour le monde, ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_very_long_text_short_stop() {
        let long_text = format!("{}STOP{}", "a".repeat(10_000), "b".repeat(5_000));
        let result = check_stop_sequences(&long_text, &["STOP".to_owned()]);
        assert_eq!(result, Some("a".repeat(10_000)));
    }

    // -- Additional extract_eos_tokens edge cases --

    #[test]
    fn test_extract_eos_tokens_float_value() {
        // serde_json parses 151643.0 as a float, and as_u64() returns None for floats
        assert!(eos_from_config(r#"{"eos_token_id": 151643.0}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_string_value() {
        assert!(eos_from_config(r#"{"eos_token_id": "not_a_number"}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_nested_array() {
        // Inner arrays are not numbers, so as_u64() returns None for them
        assert!(eos_from_config(r#"{"eos_token_id": [[1, 2], [3, 4]]}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_negative_number() {
        // as_u64() returns None for negative numbers
        assert!(eos_from_config(r#"{"eos_token_id": -1}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_very_large_number() {
        // u32::MAX is 4294967295; as_u64() succeeds but u32::try_from fails
        assert!(eos_from_config(r#"{"eos_token_id": 4294967296}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_empty_array() {
        assert!(eos_from_config(r#"{"eos_token_id": []}"#).is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_mixed_types_in_array() {
        // Only numeric entries are extracted; "two" is skipped
        assert_eq!(
            eos_from_config(r#"{"eos_token_id": [1, "two", 3]}"#),
            vec![1, 3]
        );
    }

    #[test]
    fn test_estimate_paged_kv_blocks_clamps_to_minimum() {
        assert_eq!(estimate_paged_kv_blocks(512, 512, 512, 64), 256);
    }

    #[test]
    fn test_estimate_paged_kv_blocks_clamps_to_maximum() {
        assert_eq!(estimate_paged_kv_blocks(usize::MAX, 1, 1, 1), 4096);
    }

    #[test]
    fn test_estimate_paged_kv_blocks_scales_with_geometry() {
        assert_eq!(
            estimate_paged_kv_blocks(512 * 1024 * 1024, 8, 128, 64),
            2048
        );
    }

    #[test]
    fn test_parse_enabled_flag_accepts_common_truthy_values() {
        assert_eq!(parse_enabled_flag(Some("1")), Some(true));
        assert_eq!(parse_enabled_flag(Some("true")), Some(true));
        assert_eq!(parse_enabled_flag(Some("On")), Some(true));
        assert_eq!(parse_enabled_flag(Some("yes")), Some(true));
        assert_eq!(parse_enabled_flag(None), None);
        assert_eq!(parse_enabled_flag(Some("0")), Some(false));
        assert_eq!(parse_enabled_flag(Some("false")), Some(false));
        assert_eq!(parse_enabled_flag(Some("off")), Some(false));
        assert_eq!(parse_enabled_flag(Some("no")), Some(false));
        assert_eq!(parse_enabled_flag(Some("unexpected")), None);
    }

    // -----------------------------------------------------------------------
    // find_stop_in_tail
    // -----------------------------------------------------------------------

    #[test]
    fn find_stop_in_tail_empty_stops_returns_none() {
        assert_eq!(find_stop_in_tail("hello world", 5, &[]), None);
    }

    #[test]
    fn find_stop_in_tail_finds_stop_in_new_text() {
        let stops = vec!["</s>".to_owned()];
        // "hello</s>" with the last 4 bytes new
        assert_eq!(find_stop_in_tail("hello</s>", 4, &stops), Some(5));
    }

    #[test]
    fn find_stop_in_tail_finds_stop_spanning_boundary() {
        let stops = vec!["STOP".to_owned()];
        // "abcSTOP" where only "OP" is new; "ST" was emitted previously
        assert_eq!(find_stop_in_tail("abcSTOP", 2, &stops), Some(3));
    }

    #[test]
    fn find_stop_in_tail_ignores_stop_fully_in_old_text() {
        let stops = vec!["XY".to_owned()];
        // "XYabcdefgh" with 2 new bytes: the scan window covers the last
        // 2 + (2 - 1) = 3 bytes only, so the old "XY" is not rescanned.
        assert_eq!(find_stop_in_tail("XYabcdefgh", 2, &stops), None);
    }

    #[test]
    fn find_stop_in_tail_earliest_of_multiple() {
        let stops = vec!["BBB".to_owned(), "AAA".to_owned()];
        assert_eq!(find_stop_in_tail("xAAAyBBBz", 9, &stops), Some(1));
    }

    #[test]
    fn find_stop_in_tail_handles_multibyte_boundary() {
        let stops = vec!["端".to_owned()];
        // The tail start can land mid-codepoint; it must back up to a
        // char boundary instead of panicking. "日本語端" = 4 chars, 12 bytes.
        assert_eq!(find_stop_in_tail("日本語端", 3, &stops), Some(9));
    }

    // -----------------------------------------------------------------------
    // IncrementalDetok
    // -----------------------------------------------------------------------

    /// Minimal word-level tokenizer with a byte-level decoder for detok tests.
    fn word_tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "WordLevel",
                "vocab": {"Hello": 0, "Ġworld": 1, "!": 2, "ðŁĺ": 3, "Ģ": 4},
                "unk_token": "Hello"
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    /// Empty skip-id set for detok tests that exercise plain (non-special)
    /// tokens, where `skip_special_tokens` makes no difference.
    fn no_skip() -> std::sync::Arc<std::collections::HashSet<u32>> {
        std::sync::Arc::new(std::collections::HashSet::new())
    }

    /// `content_preserving_skip_ids` strips control tokens (chat delimiters,
    /// sentinels, the explicit EOS list) but keeps content-bearing special
    /// tokens so tool-call markup reaches the parser.
    #[test]
    fn skip_ids_strip_control_but_keep_content_special_tokens() {
        let mut tok = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let _ = tok.add_special_tokens([
            tokenizers::AddedToken::from("<|im_end|>", true),
            tokenizers::AddedToken::from("</s>", true),
            tokenizers::AddedToken::from("<function>", true),
            tokenizers::AddedToken::from("<tool_call>", true),
        ]);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let skip = super::content_preserving_skip_ids(&tok, &[im_end, 7]);

        // Stripped: <|…|> delimiter, the </s> sentinel, and the explicit eos id.
        assert!(skip.contains(&im_end), "<|…|> delimiter is a control token");
        assert!(
            skip.contains(&tok.token_to_id("</s>").unwrap()),
            "</s> sentinel stripped"
        );
        assert!(skip.contains(&7), "explicit eos id retained");
        // Preserved: content-bearing tool-call markup the parser depends on.
        assert!(
            !skip.contains(&tok.token_to_id("<function>").unwrap()),
            "<function> preserved for the tool-call parser"
        );
        assert!(
            !skip.contains(&tok.token_to_id("<tool_call>").unwrap()),
            "<tool_call> preserved"
        );
    }

    #[test]
    fn incremental_detok_emits_per_token_diffs() {
        let tokenizer = word_tokenizer();
        let mut tokens: Vec<u32> = vec![0];
        let first = tokenizer.decode(&tokens, true).unwrap();
        let mut detok = IncrementalDetok::new(first.clone(), tokens.len(), no_skip());

        tokens.push(1);
        let second = detok.append(&tokenizer, &tokens).unwrap();
        tokens.push(2);
        let third = detok.append(&tokenizer, &tokens).unwrap();

        let full = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(format!("{first}{second}{third}"), full);
        assert_eq!(detok.text, full);
    }

    #[test]
    fn incremental_detok_flush_without_pending_is_empty() {
        let tokenizer = word_tokenizer();
        let mut tokens: Vec<u32> = vec![0];
        let first = tokenizer.decode(&tokens, true).unwrap();
        let mut detok = IncrementalDetok::new(first, tokens.len(), no_skip());

        tokens.push(1);
        detok.append(&tokenizer, &tokens).unwrap();
        assert_eq!(detok.flush(&tokenizer, &tokens).unwrap(), "");
    }

    // Tokens 3 and 4 are the byte-level pieces of 😀 (U+1F600).
    // Appending only token 3 must hold back the incomplete UTF-8 sequence
    // (return ""), and appending both tokens must emit the full emoji.
    #[test]
    fn incremental_detok_first_token_partial_utf8_held_back() {
        let tokenizer = word_tokenizer();
        let mut detok = IncrementalDetok::new(String::new(), 0, no_skip());

        // First partial piece: held back, no replacement char emitted
        let held = detok.append(&tokenizer, &[3]).unwrap();
        assert_eq!(held, "", "partial UTF-8 token must be held back");

        // Completing the sequence emits the full emoji
        let emitted = detok.append(&tokenizer, &[3, 4]).unwrap();
        assert_eq!(emitted, "😀", "completing UTF-8 should emit the emoji");

        let full = tokenizer.decode(&[3u32, 4], true).unwrap();
        assert_eq!(detok.text, full, "detok.text must equal full decode");
    }

    // flush() must emit any bytes held back by append(), and a second flush
    // on the same (now-fully-drained) detok must return "".
    #[test]
    fn incremental_detok_flush_emits_pending() {
        let tokenizer = word_tokenizer();
        let mut detok = IncrementalDetok::new(String::new(), 0, no_skip());

        // Partial piece held back
        let held = detok.append(&tokenizer, &[3]).unwrap();
        assert_eq!(held, "", "partial UTF-8 must be held back before flush");

        // flush must emit something (a replacement char is acceptable here)
        let flushed = detok.flush(&tokenizer, &[3]).unwrap();
        assert!(!flushed.is_empty(), "flush must emit held-back text");

        // A second flush on an already-drained detok returns ""
        let flushed2 = detok.flush(&tokenizer, &[3]).unwrap();
        assert_eq!(flushed2, "", "second flush must return empty string");
    }

    // find_stop_in_tail must identify a stop whose prefix was already emitted,
    // returning the byte position that lets the caller emit the prefix ("hi")
    // before the stop ("STOP") in the same new-text window.
    #[test]
    fn find_stop_in_tail_first_token_prefix_and_stop() {
        let stops = vec!["STOP".to_owned()];
        // "hiSTOP": all 6 bytes are new, stop starts at byte 2
        assert_eq!(
            find_stop_in_tail("hiSTOP", 6, &stops),
            Some(2),
            "stop at pos 2 so 'hi' can be emitted as prefix"
        );
    }
}
