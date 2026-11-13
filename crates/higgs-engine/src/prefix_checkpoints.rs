//! Checkpointed recurrent state for prefix-cache reuse on hybrid models.
//!
//! A dense attention cache can be cut back to any token position, but a
//! recurrent (gated delta net) state is only valid at the position where it
//! was captured. So that a prompt sharing only the start of a cached entry can
//! still resume, prefill pauses at regular token boundaries and keeps a copy of
//! every recurrent layer's state there. A later prompt resumes from the
//! nearest checkpoint at or before the point where it diverges.
//!
//! Measured cost of one checkpoint (all recurrent layers, `ssm_state` in
//! float32 plus the convolution history): 154 MB for the 27B dense hybrid
//! (48 recurrent layers) and 63 MB for the 35B-A3B mixture-of-experts hybrid
//! (30 recurrent layers). That is the cost of roughly 2,300 and 1,500 cached
//! tokens of attention KV, which is why checkpoints are sparse and budgeted.

use std::collections::VecDeque;
use std::sync::Arc;

use higgs_models::cache::KeyValueCache;
use higgs_models::qwen3_next::ArraysCache;
use higgs_models::{AnyCache, LayerCache};
use mlx_rs::error::Exception;
use mlx_rs::transforms::eval;

use crate::paged_prefix_cache::DEFAULT_BLOCK_SIZE;

/// Tokens between checkpoints.
///
/// A prompt that diverges inside an interval re-runs at most this many tokens
/// of the shared prefix. It equals the default prefill chunk of the large
/// hybrid models so checkpoint boundaries are chunk boundaries;
/// [`CheckpointPolicy::resolve`] rounds it up to a multiple of the actual
/// chunk size.
pub const DEFAULT_CHECKPOINT_INTERVAL_TOKENS: usize = 512;

/// Most checkpoints one entry keeps.
///
/// When a prompt has more interval boundaries than this, the newest half of the allowance goes to the most
/// recent boundaries (a conversation usually diverges near its end) and the
/// rest is spread evenly over the earlier ones (a shared system prompt and tool
/// definitions end early).
pub const MAX_CHECKPOINTS_PER_ENTRY: usize = 8;

/// Total bytes of checkpoint state kept across all entries. Beyond it the
/// least recently used entries lose their checkpoints first.
pub const DEFAULT_CHECKPOINT_BUDGET_BYTES: usize = 2 << 30;

/// Overrides [`DEFAULT_CHECKPOINT_BUDGET_BYTES`], in mebibytes. `0` turns
/// checkpointing off, leaving hybrid entries resumable only at their end.
pub const CHECKPOINT_BUDGET_ENV: &str = "HIGGS_PREFIX_CHECKPOINT_BUDGET_MB";

/// Where checkpoints are taken and how many are kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointPolicy {
    interval_tokens: usize,
    max_per_entry: usize,
    budget_bytes: usize,
}

impl CheckpointPolicy {
    pub const fn new(interval_tokens: usize, max_per_entry: usize, budget_bytes: usize) -> Self {
        Self {
            interval_tokens,
            max_per_entry,
            budget_bytes,
        }
    }

    pub const fn disabled() -> Self {
        Self::new(DEFAULT_CHECKPOINT_INTERVAL_TOKENS, 0, 0)
    }

    /// The default policy for a model prefilling in `chunk_tokens` chunks,
    /// with the budget overridden by [`CHECKPOINT_BUDGET_ENV`].
    pub fn resolve(chunk_tokens: usize) -> Self {
        let budget_bytes = std::env::var(CHECKPOINT_BUDGET_ENV)
            .ok()
            .and_then(|text| text.trim().parse::<usize>().ok())
            .map_or(DEFAULT_CHECKPOINT_BUDGET_BYTES, |megabytes| {
                megabytes.saturating_mul(1 << 20)
            });
        Self::new(
            aligned_interval(DEFAULT_CHECKPOINT_INTERVAL_TOKENS, chunk_tokens),
            MAX_CHECKPOINTS_PER_ENTRY,
            budget_bytes,
        )
    }

    pub const fn is_enabled(&self) -> bool {
        self.max_per_entry > 0 && self.budget_bytes > 0 && self.interval_tokens > 0
    }

    pub const fn interval_tokens(&self) -> usize {
        self.interval_tokens
    }

    pub const fn max_per_entry(&self) -> usize {
        self.max_per_entry
    }

    pub const fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Positions at which a prefill that starts at `resume_at` and whose cache
    /// key is `key_len` tokens long should capture state: interval multiples
    /// strictly inside `(resume_at, key_len)`, thinned to the per-entry cap.
    pub fn positions(&self, resume_at: usize, key_len: usize) -> Vec<usize> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let grid: Vec<usize> = (1..)
            .map(|multiple| multiple * self.interval_tokens)
            .skip_while(|&position| position <= resume_at)
            .take_while(|&position| position < key_len)
            .collect();
        thin_positions(grid, self.max_per_entry)
    }
}

/// The smallest multiple of `chunk_tokens` that is at least `wanted` and a
/// multiple of the cache block size, so checkpoints land on chunk and block
/// boundaries alike.
fn aligned_interval(wanted: usize, chunk_tokens: usize) -> usize {
    let chunk = chunk_tokens.max(1);
    let mut interval = wanted.div_ceil(chunk) * chunk;
    while interval % DEFAULT_BLOCK_SIZE != 0 {
        interval += chunk;
    }
    interval
}

/// Keep at most `max` of the ascending `positions`: the newest `max / 2`, and
/// the rest spread evenly over the earlier ones starting with the earliest.
pub fn thin_positions(positions: Vec<usize>, max: usize) -> Vec<usize> {
    if positions.len() <= max {
        return positions;
    }
    let newest = max / 2;
    let spread = max - newest;
    let older_count = positions.len() - newest;
    let mut kept: Vec<usize> = (0..spread)
        .filter_map(|slot| positions.get(slot * older_count / spread).copied())
        .collect();
    kept.extend(positions.iter().skip(older_count).copied());
    kept
}

/// The recurrent state of every linear-attention layer at one token position.
///
/// Owns its storage: arrays are evaluated and copied at capture, so no view
/// keeps a prefill chunk's activations alive and nothing aliases the live
/// cache, which goes on to update its own state in place. Cheap to share
/// through `Arc` between entries.
#[derive(Debug)]
pub struct RecurrentCheckpoint {
    position: usize,
    /// Indexed like the model's layers; `None` for attention layers.
    layers: Vec<Option<ArraysCache>>,
    bytes: usize,
}

impl RecurrentCheckpoint {
    /// Copy the recurrent state out of a hybrid `cache` holding `position` tokens.
    //
    // MLX arrays are neither `Send` nor `Sync`; like the cache entries that
    // hold them, checkpoints are only touched on the engine thread.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn capture(cache: &AnyCache, position: usize) -> Result<Arc<Self>, Exception> {
        let AnyCache::Hybrid(layers) = cache else {
            return Err(Exception::custom("only hybrid caches hold recurrent state"));
        };
        let mut bytes = 0;
        let copied = layers
            .iter()
            .map(|layer| match layer {
                Some(LayerCache::Arrays(state)) => {
                    state.eval_arrays()?;
                    let conv_state = state.conv_state.as_ref().map(mlx_rs::Array::deep_clone);
                    let ssm_state = state.ssm_state.as_ref().map(mlx_rs::Array::deep_clone);
                    bytes += conv_state
                        .iter()
                        .chain(&ssm_state)
                        .map(mlx_rs::Array::nbytes)
                        .sum::<usize>();
                    Ok(Some(ArraysCache {
                        conv_state,
                        ssm_state,
                        conv_pos: state.conv_pos,
                        offset: state.offset,
                    }))
                }
                Some(LayerCache::KV(_)) | None => Ok(None),
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        Ok(Arc::new(Self {
            position,
            layers: copied,
            bytes,
        }))
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// The saved state of layer `index`, as a cache ready to resume from.
    pub fn layer(&self, index: usize) -> Option<ArraysCache> {
        self.layers.get(index).and_then(Clone::clone)
    }
}

/// Whether `cache` can be cut back to a shorter length given only a recurrent
/// checkpoint: every attention layer must be plain dense KV.
pub fn supports_checkpoints(cache: &AnyCache) -> bool {
    let AnyCache::Hybrid(layers) = cache else {
        return false;
    };
    layers.iter().all(|layer| match layer {
        Some(LayerCache::KV(kv)) => {
            !kv.is_quantized() && !kv.is_mla() && kv.keys().is_some() == kv.values().is_some()
        }
        Some(LayerCache::Arrays(_)) | None => true,
    })
}

/// What a prefill reached at a stop position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrival {
    /// Not at a stop, or at one that needs nothing from the caller.
    Passed,
    /// The cache now holds exactly the prefix-cache key; the caller stores it.
    EntryEnd,
}

/// The pauses a prefill makes so state can be captured, and the checkpoints
/// collected so far (inherited from a resumed entry, then newly captured).
#[derive(Debug, Default)]
pub struct PrefillCheckpoints {
    /// Absolute token positions still to stop at, ascending.
    upcoming: VecDeque<usize>,
    /// The position at which the cache holds exactly the cache key, when that
    /// is before the end of the prompt.
    entry_end: Option<usize>,
    collected: Vec<Arc<RecurrentCheckpoint>>,
}

impl PrefillCheckpoints {
    /// Nothing to capture, only `inherited` checkpoints to carry forward.
    pub const fn carrying(inherited: Vec<Arc<RecurrentCheckpoint>>) -> Self {
        Self {
            upcoming: VecDeque::new(),
            entry_end: None,
            collected: inherited,
        }
    }

    /// Plan the stops of a prefill that resumes `cache` at `resume_at` and
    /// runs `prompt_len` tokens in total, keyed on the first `key_len`.
    ///
    /// A recurrent state cannot be cut back to the key once the generation
    /// suffix has run, so a hybrid prefill with a suffix stops at the key.
    pub fn plan(
        policy: &CheckpointPolicy,
        cache: &AnyCache,
        resume_at: usize,
        key_len: usize,
        prompt_len: usize,
        inherited: Vec<Arc<RecurrentCheckpoint>>,
    ) -> Self {
        if !matches!(cache, AnyCache::Hybrid(_)) {
            return Self::carrying(inherited);
        }
        let entry_end = (key_len > resume_at && key_len < prompt_len).then_some(key_len);
        let upcoming = if supports_checkpoints(cache) {
            policy.positions(resume_at, key_len).into()
        } else {
            VecDeque::new()
        };
        Self {
            upcoming,
            entry_end,
            collected: inherited,
        }
    }

    /// The first stop after position `after`.
    pub fn next_stop(&self, after: usize) -> Option<usize> {
        self.upcoming
            .iter()
            .copied()
            .chain(self.entry_end)
            .filter(|&stop| stop > after)
            .min()
    }

    /// Give up any stops not yet reached (the prefill cannot be paused).
    pub fn cancel_stops(&mut self) {
        self.upcoming.clear();
        self.entry_end = None;
    }

    /// Record that the prefill has run to `position` with `cache` in hand.
    pub fn arrive(&mut self, position: usize, cache: &AnyCache) -> Result<Arrival, Exception> {
        self.settle(cache)?;
        if self.upcoming.front() == Some(&position) {
            self.upcoming.pop_front();
            self.collected
                .push(RecurrentCheckpoint::capture(cache, position)?);
        }
        if self.entry_end == Some(position) {
            self.entry_end = None;
            return Ok(Arrival::EntryEnd);
        }
        Ok(Arrival::Passed)
    }

    /// Evaluate the cache so the lazy graph of earlier chunks is released.
    fn settle(&self, cache: &AnyCache) -> Result<(), Exception> {
        if self.upcoming.is_empty() && self.entry_end.is_none() {
            return Ok(());
        }
        let AnyCache::Hybrid(layers) = cache else {
            return Ok(());
        };
        let mut targets = Vec::new();
        for layer in layers.iter().flatten() {
            match layer {
                LayerCache::KV(kv) => targets.extend(kv.eval_targets()),
                LayerCache::Arrays(state) => {
                    targets.extend(state.conv_state.iter().chain(&state.ssm_state));
                }
            }
        }
        eval(targets)
    }

    /// Checkpoints to store with the entry: inherited and newly captured.
    pub fn collected(&self) -> Vec<Arc<RecurrentCheckpoint>> {
        self.collected.clone()
    }
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stdout
)]
#[cfg(test)]
mod tests {
    use super::*;

    fn policy(interval: usize, max: usize) -> CheckpointPolicy {
        CheckpointPolicy::new(interval, max, usize::MAX)
    }

    #[test]
    fn positions_are_interval_multiples_strictly_inside_the_key() {
        let policy = policy(512, 8);
        assert_eq!(policy.positions(0, 2000), vec![512, 1024, 1536]);
        assert_eq!(policy.positions(0, 1024), vec![512]);
        assert_eq!(policy.positions(0, 512), Vec::<usize>::new());
        assert_eq!(policy.positions(600, 2000), vec![1024, 1536]);
        assert_eq!(policy.positions(1024, 2000), vec![1536]);
    }

    #[test]
    fn positions_are_thinned_to_the_cap_keeping_the_newest_and_the_earliest() {
        let policy = policy(100, 4);
        let positions = policy.positions(0, 1050);
        assert_eq!(positions.len(), 4);
        assert_eq!(positions.first(), Some(&100));
        assert_eq!(&positions[2..], &[900, 1000]);
    }

    #[test]
    fn thinning_keeps_everything_under_the_cap() {
        assert_eq!(thin_positions(vec![1, 2, 3], 3), vec![1, 2, 3]);
        assert_eq!(thin_positions(vec![1, 2, 3], 8), vec![1, 2, 3]);
    }

    #[test]
    fn thinning_spreads_the_older_half_evenly() {
        let grid: Vec<usize> = (1..=12).collect();
        assert_eq!(thin_positions(grid, 8), vec![1, 3, 5, 7, 9, 10, 11, 12]);
    }

    #[test]
    fn a_disabled_policy_plans_nothing() {
        assert!(
            CheckpointPolicy::disabled()
                .positions(0, 100_000)
                .is_empty()
        );
        assert!(!CheckpointPolicy::disabled().is_enabled());
        assert!(!CheckpointPolicy::new(512, 8, 0).is_enabled());
    }

    #[test]
    fn interval_is_a_multiple_of_chunk_and_block_size() {
        assert_eq!(aligned_interval(512, 512), 512);
        assert_eq!(aligned_interval(512, 384), 768);
        assert_eq!(aligned_interval(512, 768), 768);
        assert_eq!(aligned_interval(512, 1024), 1024);
        assert_eq!(aligned_interval(512, 100), 800);
        assert_eq!(aligned_interval(512, 32), 512);
    }

    /// Largest absolute difference relative to the largest magnitude in `expected`.
    fn relative_difference(expected: &mlx_rs::Array, actual: &mlx_rs::Array) -> f32 {
        let to_vec = |array: &mlx_rs::Array| {
            array
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        let expected_values = to_vec(expected);
        let actual_values = to_vec(actual);
        assert_eq!(expected_values.len(), actual_values.len());
        let largest = expected_values
            .iter()
            .fold(0.0_f32, |max, x| max.max(x.abs()));
        let difference = expected_values
            .iter()
            .zip(&actual_values)
            .fold(0.0_f32, |max, (a, b)| max.max((a - b).abs()));
        difference / largest
    }

    /// Deterministic token ids that avoid the special-token range.
    fn pseudo_random_tokens(seed: u64, count: usize) -> Vec<u32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                2000 + u32::try_from((state >> 33) % 50_000).unwrap()
            })
            .collect()
    }

    struct Prefilled {
        cache: AnyCache,
        last_logits: mlx_rs::Array,
        resumed_at: usize,
    }

    /// The prefill the simple engine runs for a chat prompt with no generation
    /// suffix: look up, resume, pause at the planned checkpoints, store.
    fn prefill_through_cache(
        model: &mut higgs_models::AnyModel,
        store: &mut crate::paged_prefix_cache::PagedPrefixCache,
        prompt: &[u32],
        chunk: i32,
    ) -> Prefilled {
        use crate::generation_support::{PrefixResume, resume_from_prefix};
        use mlx_rs::{Array, ops::indexing::IndexOp};

        let PrefixResume {
            resume_at,
            mut cache,
            checkpoints: inherited,
        } = resume_from_prefix(store.find_longest_prefix(prompt), prompt, || {
            model.make_cache().map_err(crate::error::EngineError::Mlx)
        })
        .unwrap();
        let policy = *store.checkpoint_policy();
        let mut stops = PrefillCheckpoints::plan(
            &policy,
            &cache,
            resume_at,
            prompt.len(),
            prompt.len(),
            inherited,
        );
        let remaining =
            Array::from(prompt.get(resume_at..).unwrap()).index(mlx_rs::ops::indexing::NewAxis);
        let mut position = resume_at;
        while let Some(stop) = stops.next_stop(position) {
            let segment = remaining.index((
                ..,
                i32::try_from(position - resume_at).unwrap()
                    ..i32::try_from(stop - resume_at).unwrap(),
            ));
            model.forward_chunked(&segment, &mut cache, chunk).unwrap();
            position = stop;
            stops.arrive(stop, &cache).unwrap();
        }
        let tail = remaining.index((.., i32::try_from(position - resume_at).unwrap()..));
        let logits = model.forward_chunked(&tail, &mut cache, chunk).unwrap();
        let last_logits = logits.index((.., -1, ..));
        last_logits.eval().unwrap();
        store.store_with_checkpoints(prompt, &cache, stops.collected());
        Prefilled {
            cache,
            last_logits,
            resumed_at: resume_at,
        }
    }

    fn decode_steps(
        model: &mut higgs_models::AnyModel,
        cache: &mut AnyCache,
        tokens: &[u32],
    ) -> Vec<mlx_rs::Array> {
        use mlx_rs::{Array, ops::indexing::IndexOp};
        tokens
            .iter()
            .map(|&token| {
                let logits = model
                    .forward(&Array::from_slice(&[token], &[1, 1]), None, cache)
                    .unwrap();
                let last = logits.index((.., -1, ..));
                last.eval().unwrap();
                last
            })
            .collect()
    }

    const DECODE_TOKENS: [u32; 4] = [2111, 3222, 4333, 5444];

    /// Logits after the prompt and after each of [`DECODE_TOKENS`].
    fn logits_after(
        model: &mut higgs_models::AnyModel,
        prefilled: &mut Prefilled,
    ) -> Vec<mlx_rs::Array> {
        let mut all = vec![prefilled.last_logits.clone()];
        all.extend(decode_steps(model, &mut prefilled.cache, &DECODE_TOKENS));
        all
    }

    fn worst_difference(expected: &[mlx_rs::Array], actual: &[mlx_rs::Array]) -> f32 {
        expected.iter().zip(actual).fold(0.0_f32, |worst, (a, b)| {
            worst.max(relative_difference(a, b))
        })
    }

    /// Prefill `prompt` in two pieces, split at `split`, on one live cache.
    fn split_control_logits(
        model: &mut higgs_models::AnyModel,
        prompt: &[u32],
        split: usize,
        chunk: i32,
    ) -> Vec<mlx_rs::Array> {
        use mlx_rs::{
            Array,
            ops::indexing::{IndexOp, NewAxis},
        };
        let mut cache = model.make_cache().unwrap();
        let head = Array::from(&prompt[..split]).index(NewAxis);
        model.forward_chunked(&head, &mut cache, chunk).unwrap();
        let tail = Array::from(&prompt[split..]).index(NewAxis);
        let logits = model.forward_chunked(&tail, &mut cache, chunk).unwrap();
        let last_logits = logits.index((.., -1, ..));
        last_logits.eval().unwrap();
        let mut all = vec![last_logits];
        all.extend(decode_steps(model, &mut cache, &DECODE_TOKENS));
        all
    }

    fn cold_logits(
        model: &mut higgs_models::AnyModel,
        prompt: &[u32],
        chunk: i32,
    ) -> Vec<mlx_rs::Array> {
        let mut store = crate::paged_prefix_cache::PagedPrefixCache::new(0, 32);
        let mut cold = prefill_through_cache(model, &mut store, prompt, chunk);
        logits_after(model, &mut cold)
    }

    /// Size, shapes and dtypes of one full set of recurrent state.
    /// `HIGGS_MODEL_PATH=<hybrid model dir> cargo test -p higgs-engine --release -- --ignored checkpoint_size --nocapture`
    #[test]
    #[ignore = "requires real model weights (set HIGGS_MODEL_PATH)"]
    fn checkpoint_size_of_a_real_model() {
        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = crate::model_loader::load_model(&model_path).unwrap();
        let prompt = pseudo_random_tokens(1, 600);
        let mut store = crate::paged_prefix_cache::PagedPrefixCache::new(0, 32);
        let prefilled = prefill_through_cache(&mut model, &mut store, &prompt, 512);
        let checkpoint = RecurrentCheckpoint::capture(&prefilled.cache, 600).unwrap();
        let layers: Vec<_> = checkpoint.layers.iter().flatten().collect();
        println!("recurrent layers: {}", layers.len());
        if let Some(first) = layers.first() {
            let describe = |array: &Option<mlx_rs::Array>| {
                array
                    .as_ref()
                    .map(|a| format!("{:?} {:?} {} bytes", a.shape(), a.dtype(), a.nbytes()))
            };
            println!("conv_state: {:?}", describe(&first.conv_state));
            println!("ssm_state:  {:?}", describe(&first.ssm_state));
        }
        println!("bytes per checkpoint: {}", checkpoint.bytes());
        if let higgs_models::AnyCache::Hybrid(all) = &prefilled.cache {
            let attention_bytes: usize = all
                .iter()
                .flatten()
                .filter_map(|layer| match layer {
                    higgs_models::LayerCache::KV(kv) => {
                        Some(kv.keys().unwrap().nbytes() + kv.values().unwrap().nbytes())
                    }
                    higgs_models::LayerCache::Arrays(_) => None,
                })
                .sum();
            println!(
                "attention layers: {} KV bytes for {} tokens of capacity (live cache, including padding)",
                attention_bytes, 600
            );
        }
    }

    /// Resuming from a checkpoint must reproduce a cold prefill.
    /// `HIGGS_MODEL_PATH=<hybrid model dir> cargo test -p higgs-engine --release -- --ignored resume_from_checkpoint --nocapture --test-threads=1`
    #[test]
    #[ignore = "requires real model weights (set HIGGS_MODEL_PATH)"]
    #[allow(clippy::too_many_lines)]
    fn resume_from_checkpoint_matches_cold_prefill() {
        use crate::paged_prefix_cache::PagedPrefixCache;
        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = crate::model_loader::load_model(&model_path).unwrap();
        let chunk = 512;
        let policy = CheckpointPolicy::new(512, MAX_CHECKPOINTS_PER_ENTRY, usize::MAX);
        let mut store = PagedPrefixCache::new(16, 32).with_checkpoint_policy(policy);

        let stored = pseudo_random_tokens(7, 2700);
        let first = prefill_through_cache(&mut model, &mut store, &stored, chunk);
        assert_eq!(first.resumed_at, 0);
        println!(
            "stored a {}-token prompt: {} MB of checkpoints",
            stored.len(),
            store.checkpoint_bytes() >> 20
        );

        let floor = {
            let reference = cold_logits(&mut model, &stored, chunk);
            let rechunked = cold_logits(&mut model, &stored, 333);
            worst_difference(&reference, &rechunked)
        };
        let allowed = 0.03_f32.max(floor * 2.0);
        println!("noise floor (cold, chunk 512 vs 333): {floor}; allowed {allowed}");

        let diverge = |shared: usize, seed: u64| -> Vec<u32> {
            let mut prompt = stored[..shared].to_vec();
            prompt.extend(pseudo_random_tokens(seed, 150));
            prompt
        };
        let scenarios = [
            ("a: just after a checkpoint", diverge(1034, 11), 1024),
            ("b: middle of an interval", diverge(1024 + 300, 12), 1024),
            ("c: before the first checkpoint", diverge(400, 13), 0),
            (
                "d: whole key is a prefix",
                diverge(stored.len(), 14),
                stored.len(),
            ),
        ];
        let mut worst = 0.0_f32;
        for (name, prompt, expected_resume) in scenarios {
            let cold = cold_logits(&mut model, &prompt, chunk);
            let mut resumed = prefill_through_cache(&mut model, &mut store, &prompt, chunk);
            assert_eq!(
                resumed.resumed_at, expected_resume,
                "{name}: resume position"
            );
            let warm = logits_after(&mut model, &mut resumed);
            let difference = worst_difference(&cold, &warm);
            println!(
                "{name}: resumed at {} of {}, largest logit difference {difference}",
                resumed.resumed_at,
                prompt.len()
            );
            assert!(!difference.is_nan(), "{name}: NaN logits");
            if expected_resume == prompt.len() - 150 {
                // Resuming at an unaligned position changes where the prompt is
                // split into chunks, which alone moves half-precision results
                // (mixture-of-experts routing most of all). The same split
                // without a cache round trip isolates that from the cache.
                let control = split_control_logits(&mut model, &prompt, expected_resume, chunk);
                let round_trip = worst_difference(&control, &warm);
                println!("{name}: same split on a live cache vs resumed: {round_trip}");
                assert!(
                    round_trip < 1e-4,
                    "{name}: cache round trip changed logits by {round_trip}"
                );
            } else {
                assert!(difference < allowed, "{name}: {difference} >= {allowed}");
            }
            worst = worst.max(difference);
        }

        // Generate from a resumed prefill, then extend the conversation and
        // edit an earlier turn: the entry stored by the resume must serve both.
        let turn_one = diverge(1034, 21);
        let mut live = prefill_through_cache(&mut model, &mut store, &turn_one, chunk);
        assert_eq!(live.resumed_at, 1024);
        let reply = pseudo_random_tokens(22, 40);
        decode_steps(&mut model, &mut live.cache, &reply[..8]);
        let mut turn_two = turn_one.clone();
        turn_two.extend(&reply);
        turn_two.extend(pseudo_random_tokens(23, 120));
        let cold_extension = cold_logits(&mut model, &turn_two, chunk);
        let mut extended = prefill_through_cache(&mut model, &mut store, &turn_two, chunk);
        assert_eq!(
            extended.resumed_at,
            turn_one.len(),
            "extension resumes at the stored end"
        );
        let extension_difference =
            worst_difference(&cold_extension, &logits_after(&mut model, &mut extended));
        println!("extend after generating: largest logit difference {extension_difference}");
        assert!(extension_difference < allowed);

        let mut edited = turn_one[..1100].to_vec();
        edited.extend(pseudo_random_tokens(24, 100));
        let cold_edit = cold_logits(&mut model, &edited, chunk);
        let mut regenerated = prefill_through_cache(&mut model, &mut store, &edited, chunk);
        assert_eq!(
            regenerated.resumed_at, 1024,
            "an edit resumes from the checkpoint the resumed entry inherited"
        );
        let edit_difference =
            worst_difference(&cold_edit, &logits_after(&mut model, &mut regenerated));
        println!("edit an earlier turn: largest logit difference {edit_difference}");
        assert!(edit_difference < allowed);
        println!(
            "worst scenario difference {worst}; checkpoint memory held {} MB",
            store.checkpoint_bytes() >> 20
        );
    }
}
