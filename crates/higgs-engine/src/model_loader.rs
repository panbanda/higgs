use std::path::{Path, PathBuf};

use higgs_models::{
    AnyModel,
    adapter::{self, Capabilities, ModelFamily, ModelVersion},
    load_tokenizer as shared_load_tokenizer,
};

use crate::error::EngineError;

/// Configuration for loading a model from a directory.
#[derive(Debug)]
pub struct ModelConfig {
    /// Local checkpoint directory used for model and tokenizer loading.
    pub model_dir: PathBuf,
    /// Effective model type, using nested `text_config.model_type` for wrappers.
    pub model_type: String,
    /// Stable identifier of the adapter selected for this checkpoint.
    pub adapter_id: &'static str,
    /// Broad family detected from the effective model type.
    pub family: ModelFamily,
    /// Numeric version parsed from `model_type`, or `None` when no numeric version parses.
    pub version: Option<ModelVersion>,
    /// Capabilities of the resolved adapter implementation, not claims copied from the checkpoint.
    pub capabilities: Capabilities,
}

impl ModelConfig {
    /// Detect model type and create a config from a model directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref().to_path_buf();
        let detected = adapter::detect(&model_dir)?;
        let resolved = adapter::resolve(&detected)?;
        let info = resolved.describe();

        Ok(Self {
            model_dir,
            model_type: detected.model_type,
            adapter_id: info.id,
            family: detected.family,
            version: detected.version,
            capabilities: info.capabilities,
        })
    }
}

/// Whether loading `model_dir` would attach a multi-token-prediction head, which the
/// simple engine uses for speculative decode.
///
/// Reads the config and weight index only; the model is not loaded.
pub fn checkpoint_has_mtp_head<P: AsRef<Path>>(model_dir: P) -> bool {
    let Ok(config) = ModelConfig::from_dir(model_dir.as_ref()) else {
        return false;
    };
    config.capabilities.mtp && higgs_models::qwen3_next::checkpoint_has_mtp_head(model_dir)
}

/// Load a model from a directory, auto-detecting the architecture.
pub fn load_model<P: AsRef<Path>>(model_dir: P) -> Result<AnyModel, EngineError> {
    let detected = adapter::detect(model_dir.as_ref()).map_err(EngineError::Model)?;
    let resolved = adapter::resolve(&detected).map_err(EngineError::Model)?;
    resolved.load(&detected).map_err(EngineError::Model)
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer<P: AsRef<Path>>(model_dir: P) -> Result<tokenizers::Tokenizer, EngineError> {
    shared_load_tokenizer(model_dir).map_err(|e| EngineError::Tokenization(e.to_string()))
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::indexing_slicing
)]
#[cfg(test)]
mod tests {
    use super::*;
    use higgs_models::error::ModelError;

    /// Largest absolute difference between two same-shaped arrays, relative
    /// to the largest magnitude in `expected`.
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

    /// Per-row logits `[1, vocab]` of the last position of `logits`.
    fn last_logits_of_row(logits: &mlx_rs::Array, row: usize) -> mlx_rs::Array {
        use mlx_rs::ops::indexing::IndexOp;
        let row_index = i32::try_from(row).unwrap();
        logits.index((row_index..row_index + 1, -1, ..))
    }

    fn unbatched_prefill(
        model: &mut higgs_models::AnyModel,
        prompt: &[u32],
    ) -> (higgs_models::AnyCache, mlx_rs::Array) {
        use mlx_rs::{Array, ops::indexing::IndexOp};
        let mut cache = model.make_cache().unwrap();
        let length = i32::try_from(prompt.len()).unwrap();
        let logits = model
            .forward(&Array::from_slice(prompt, &[1, length]), None, &mut cache)
            .unwrap();
        (cache, logits.index((.., -1, ..)))
    }

    fn unbatched_step(
        model: &mut higgs_models::AnyModel,
        cache: &mut higgs_models::AnyCache,
        token: u32,
    ) -> mlx_rs::Array {
        use mlx_rs::{Array, ops::indexing::IndexOp};
        let logits = model
            .forward(&Array::from_slice(&[token], &[1, 1]), None, cache)
            .unwrap();
        let last = logits.index((.., -1, ..));
        last.eval().unwrap();
        last
    }

    /// One batched step; `tokens` holds one token per row. Returns `[B, 1, vocab]`.
    fn batched_step(
        model: &mut higgs_models::AnyModel,
        batch: &mut higgs_models::batched_cache::AnyBatchedCache,
        tokens: &[u32],
    ) -> mlx_rs::Array {
        let rows = i32::try_from(tokens.len()).unwrap();
        let logits = model
            .forward_batched(&mlx_rs::Array::from_slice(tokens, &[rows, 1]), batch)
            .unwrap();
        logits.eval().unwrap();
        logits
    }

    const PROMPTS: [&[u32]; 3] = [
        &[101, 2002, 3003, 404, 505, 6006, 707, 88, 99],
        &[909, 808, 7007],
        &[1100, 1200, 130, 1400, 150],
    ];
    const DECODE_TOKENS: [[u32; 8]; 3] = [
        [210, 2200, 230, 2400, 250, 2600, 270, 2800],
        [3100, 320, 3300, 340, 3500, 360, 3700, 380],
        [41, 4200, 430, 4400, 450, 4600, 470, 4800],
    ];

    /// Logits of rows that decoded alone and were never batched, and of the
    /// same rows decoded in a batch of three copies of themselves.
    struct Reference {
        prefill: Vec<mlx_rs::Array>,
        steps: Vec<Vec<mlx_rs::Array>>,
        /// Same-size batch of identical rows: shares the batch-size-dependent
        /// kernels of the scenarios below but has no padding, merging or
        /// splitting to get wrong.
        same_size: Vec<Vec<mlx_rs::Array>>,
    }

    impl Reference {
        fn new(model: &mut higgs_models::AnyModel) -> Self {
            use higgs_models::batched_cache::AnyBatchedCache;
            let mut prefill = Vec::new();
            let mut steps = Vec::new();
            let mut same_size = Vec::new();
            for (prompt, tokens) in PROMPTS.iter().zip(&DECODE_TOKENS) {
                let (mut cache, prefill_logits) = unbatched_prefill(model, prompt);
                prefill_logits.eval().unwrap();
                prefill.push(prefill_logits);
                steps.push(
                    tokens
                        .iter()
                        .map(|&token| unbatched_step(model, &mut cache, token))
                        .collect(),
                );

                let copies = (0..3).map(|_| unbatched_prefill(model, prompt).0).collect();
                let mut batch = AnyBatchedCache::from_rows(copies).unwrap();
                same_size.push(
                    tokens
                        .iter()
                        .map(|&token| {
                            let logits = batched_step(model, &mut batch, &[token; 3]);
                            let row = last_logits_of_row(&logits, 0);
                            row.eval().unwrap();
                            row
                        })
                        .collect(),
                );
            }
            Self {
                prefill,
                steps,
                same_size,
            }
        }

        /// Batch-size noise: unbatched against a same-size batch of identical rows.
        fn noise_floor(&self) -> f32 {
            self.steps
                .iter()
                .zip(&self.same_size)
                .flat_map(|(alone, batched)| alone.iter().zip(batched))
                .fold(0.0_f32, |worst, (alone, batched)| {
                    worst.max(relative_difference(alone, batched))
                })
        }
    }

    impl Reference {
        /// How many of the compared steps pick the same next token alone and in a batch.
        fn top_token_agreement(&self) -> (usize, usize) {
            let pairs: Vec<_> = self
                .steps
                .iter()
                .zip(&self.same_size)
                .flat_map(|(alone, batched)| alone.iter().zip(batched))
                .collect();
            let agreeing = pairs
                .iter()
                .filter(|(alone, batched)| {
                    let top = |logits: &mlx_rs::Array| {
                        mlx_rs::ops::indexing::argmax_axis(logits, -1, None)
                            .unwrap()
                            .item_cast::<u32>()
                    };
                    top(alone) == top(batched)
                })
                .count();
            (agreeing, pairs.len())
        }
    }

    /// Worst logit difference of a scenario against each reference.
    #[derive(Clone, Copy, Default)]
    struct Differences {
        unbatched: f32,
        same_size: f32,
    }

    impl Differences {
        fn merge(self, other: Self) -> Self {
            Self {
                unbatched: self.unbatched.max(other.unbatched),
                same_size: self.same_size.max(other.same_size),
            }
        }

        fn unbatched_only(unbatched: f32) -> Self {
            Self {
                unbatched,
                same_size: 0.0,
            }
        }
    }

    /// Rows start at the given decode positions and step together.
    fn run_batched_steps(
        model: &mut higgs_models::AnyModel,
        batch: &mut higgs_models::batched_cache::AnyBatchedCache,
        reference: &Reference,
        rows: &[usize],
        first_positions: &[usize],
        step_count: usize,
    ) -> Differences {
        let mut positions = first_positions.to_vec();
        let mut worst = Differences::default();
        for _ in 0..step_count {
            let tokens: Vec<u32> = rows
                .iter()
                .zip(&positions)
                .map(|(&row, &position)| DECODE_TOKENS[row][position])
                .collect();
            let logits = batched_step(model, batch, &tokens);
            for ((batch_row, &row), position) in rows.iter().enumerate().zip(&mut positions) {
                let actual = last_logits_of_row(&logits, batch_row);
                worst = worst.merge(Differences {
                    unbatched: relative_difference(&reference.steps[row][*position], &actual),
                    same_size: relative_difference(&reference.same_size[row][*position], &actual),
                });
                *position += 1;
            }
        }
        worst
    }

    fn batched_prefill_then_decode(
        model: &mut higgs_models::AnyModel,
        reference: &Reference,
    ) -> Differences {
        use mlx_rs::Array;
        let longest = PROMPTS.iter().map(|p| p.len()).max().unwrap();
        let mut padded = Vec::new();
        for prompt in PROMPTS {
            padded.extend(std::iter::repeat_n(0_u32, longest - prompt.len()));
            padded.extend_from_slice(prompt);
        }
        let lengths: Vec<i32> = PROMPTS
            .iter()
            .map(|p| i32::try_from(p.len()).unwrap())
            .collect();
        let (logits, mut batch) = model
            .prefill_batched(
                &Array::from_slice(&padded, &[3, i32::try_from(longest).unwrap()]),
                &lengths,
            )
            .unwrap();
        let mut worst = Differences::default();
        for (row, expected) in reference.prefill.iter().enumerate() {
            worst = worst.merge(Differences::unbatched_only(relative_difference(
                expected,
                &last_logits_of_row(&logits, row),
            )));
        }
        worst.merge(run_batched_steps(
            model,
            &mut batch,
            reference,
            &[0, 1, 2],
            &[0, 0, 0],
            3,
        ))
    }

    /// Rows prefilled one at a time, merged, decoded together, then split
    /// with every row continuing alone. Returns (steady, leave).
    fn merge_decode_split(
        model: &mut higgs_models::AnyModel,
        reference: &Reference,
    ) -> (Differences, f32) {
        use higgs_models::batched_cache::AnyBatchedCache;
        let caches = PROMPTS
            .iter()
            .map(|prompt| unbatched_prefill(model, prompt).0)
            .collect();
        let mut batch = AnyBatchedCache::from_rows(caches).unwrap();
        let steady = run_batched_steps(model, &mut batch, reference, &[0, 1, 2], &[0, 0, 0], 5);
        let mut leave = 0.0_f32;
        let mut rows = batch.into_rows().unwrap();
        for ((cache, tokens), expected) in rows.iter_mut().zip(&DECODE_TOKENS).zip(&reference.steps)
        {
            for (&token, expected_logits) in tokens.iter().zip(expected).skip(5) {
                let logits = unbatched_step(model, cache, token);
                leave = leave.max(relative_difference(expected_logits, &logits));
            }
        }
        (steady, leave)
    }

    /// One row decodes unbatched (rotating its recurrent convolution ring),
    /// is merged into a batch of one, decoded batched, then split and
    /// continued unbatched. A batch of one uses the same kernels as unbatched
    /// decode, so the difference isolates the cache conversions from batch-size
    /// rounding. Returns (batched, continued).
    fn single_row_round_trip(
        model: &mut higgs_models::AnyModel,
        reference: &Reference,
    ) -> (f32, f32) {
        use higgs_models::batched_cache::AnyBatchedCache;
        let (mut cache, _) = unbatched_prefill(model, PROMPTS[0]);
        for &token in &DECODE_TOKENS[0][..2] {
            unbatched_step(model, &mut cache, token);
        }
        let mut batch = AnyBatchedCache::from_rows(vec![cache]).unwrap();
        let batched = run_batched_steps(model, &mut batch, reference, &[0], &[2], 3).unbatched;
        let mut rows = batch.into_rows().unwrap();
        let mut continued = 0.0_f32;
        if let Some(split) = rows.first_mut() {
            for (&token, expected) in DECODE_TOKENS[0].iter().zip(&reference.steps[0]).skip(5) {
                let logits = unbatched_step(model, split, token);
                continued = continued.max(relative_difference(expected, &logits));
            }
        }
        (batched, continued)
    }

    /// Rows 0 and 1 decode batched next to a spare copy of row 0, the batch
    /// is split, the spare dropped, and a freshly prefilled third row joins,
    /// its recurrent state at a different ring position. Every step runs at
    /// batch size three, so the same-size comparison applies throughout.
    fn join_mid_decode(model: &mut higgs_models::AnyModel, reference: &Reference) -> Differences {
        use higgs_models::batched_cache::AnyBatchedCache;
        let caches = [0, 1, 0]
            .iter()
            .map(|&prompt| unbatched_prefill(model, PROMPTS[prompt]).0)
            .collect();
        let mut before = AnyBatchedCache::from_rows(caches).unwrap();
        let mut worst = run_batched_steps(model, &mut before, reference, &[0, 1, 0], &[0, 0, 0], 2);
        let mut rows = before.into_rows().unwrap();
        rows.truncate(2);
        let (joiner, prefill) = unbatched_prefill(model, PROMPTS[2]);
        worst = worst.merge(Differences::unbatched_only(relative_difference(
            &reference.prefill[2],
            &prefill,
        )));
        rows.push(joiner);
        let mut after = AnyBatchedCache::from_rows(rows).unwrap();
        worst.merge(run_batched_steps(
            model,
            &mut after,
            reference,
            &[0, 1, 2],
            &[2, 2, 0],
            4,
        ))
    }

    /// A cache after prefill and `steps` unbatched decode steps; the recurrent
    /// convolution ring has rotated `steps` positions.
    fn cache_after(
        model: &mut higgs_models::AnyModel,
        row: usize,
        steps: usize,
    ) -> higgs_models::AnyCache {
        let (mut cache, _) = unbatched_prefill(model, PROMPTS[row]);
        for &token in &DECODE_TOKENS[row][..steps] {
            unbatched_step(model, &mut cache, token);
        }
        cache
    }

    /// Rows of equal total length (so no padding) but different recurrent ring
    /// positions and contents merge into one batch. Each row's logits are
    /// compared with a batch of three copies of itself at the same batch size
    /// and history, which shares every kernel; unlike the padded scenarios,
    /// nothing differs but the merge, so the result must be exact up to
    /// attention-free rounding, even for a mixture-of-experts model.
    fn merge_rows_at_different_ring_positions(model: &mut higgs_models::AnyModel) -> f32 {
        use higgs_models::batched_cache::AnyBatchedCache;
        // (prompt, decode steps taken alone): total lengths 9, 9 and 9.
        const STARTS: [(usize, usize); 3] = [(0, 0), (1, 6), (2, 4)];
        const BATCHED_STEPS: usize = 2;

        let mut expected: Vec<Vec<mlx_rs::Array>> = Vec::new();
        for &(row, taken) in &STARTS {
            let copies = (0..3).map(|_| cache_after(model, row, taken)).collect();
            let mut batch = AnyBatchedCache::from_rows(copies).unwrap();
            expected.push(
                (taken..taken + BATCHED_STEPS)
                    .map(|position| {
                        let logits =
                            batched_step(model, &mut batch, &[DECODE_TOKENS[row][position]; 3]);
                        let first = last_logits_of_row(&logits, 0);
                        first.eval().unwrap();
                        first
                    })
                    .collect(),
            );
        }

        let caches = STARTS
            .iter()
            .map(|&(row, taken)| cache_after(model, row, taken))
            .collect();
        let mut batch = AnyBatchedCache::from_rows(caches).unwrap();
        let mut worst = 0.0_f32;
        for step in 0..BATCHED_STEPS {
            let tokens: Vec<u32> = STARTS
                .iter()
                .map(|&(row, taken)| DECODE_TOKENS[row][taken + step])
                .collect();
            let logits = batched_step(model, &mut batch, &tokens);
            for (batch_row, row_expected) in expected.iter().enumerate() {
                worst = worst.max(relative_difference(
                    &row_expected[step],
                    &last_logits_of_row(&logits, batch_row),
                ));
            }
        }
        worst
    }

    /// Batched prefill and decode must reproduce one-at-a-time prefill and
    /// decode for every architecture `supports_batched_decode` admits. Needs a
    /// real checkpoint:
    /// `HIGGS_MODEL_PATH=<model dir> cargo test -p higgs-engine -- --ignored batched_decode_matches --nocapture`
    ///
    /// Scenarios, each compared per row against rows that are never batched:
    /// - prefill (architectures with batched prefill): left-padded batched
    ///   prefill, then batched decode.
    /// - steady: rows prefilled one at a time and merged, then batched decode.
    /// - leave: the batch is split and every row continues unbatched.
    /// - join: a freshly prefilled row joins a batch that is mid-decode.
    ///
    /// Kernels round differently by batch size, and a mixture-of-experts model
    /// amplifies that (routing and 40 residual layers), so each scenario is
    /// also compared with a batch of identical rows of the same size. That
    /// comparison has no batch-size noise, only merge, padding, mask and
    /// position handling.
    #[test]
    #[ignore = "requires real model weights (set HIGGS_MODEL_PATH)"]
    fn batched_decode_matches_unbatched_decode() {
        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = load_model(&model_path).unwrap();
        assert!(
            model.supports_batched_decode(),
            "this architecture does not claim batched decode support"
        );

        let reference = Reference::new(&mut model);
        let floor = reference.noise_floor();
        let prefill = model
            .supports_batched_prefill()
            .then(|| batched_prefill_then_decode(&mut model, &reference));
        let (steady, leave) = merge_decode_split(&mut model, &reference);
        let join = join_mid_decode(&mut model, &reference);
        let (single_batched, single_continued) = single_row_round_trip(&mut model, &reference);
        let phases = merge_rows_at_different_ring_positions(&mut model);

        println!("largest logit difference relative to the largest logit");
        let (agreeing, compared) = reference.top_token_agreement();
        println!("  noise floor (3 identical rows vs unbatched, 8 steps): {floor}");
        println!("  same top token as unbatched in {agreeing} of {compared} steps");
        println!("                                          vs unbatched   vs same-size batch");
        if let Some(worst) = prefill {
            println!(
                "  batched prefill + 3 decode steps:      {:>12} {:>12}",
                worst.unbatched, worst.same_size
            );
        }
        println!(
            "  steady (merge, 5 steps):               {:>12} {:>12}",
            steady.unbatched, steady.same_size
        );
        println!("  leave (split, 3 steps unbatched):      {leave:>12}");
        println!(
            "  join (row joins mid-decode, 4 steps):  {:>12} {:>12}",
            join.unbatched, join.same_size
        );
        println!("  equal lengths, different ring positions vs same-size batch: {phases}");
        println!("  one row batched of 1 after unbatched steps: {single_batched}");
        println!("  ... then split and continued unbatched:     {single_continued}");

        // Half-precision kernels round differently by batch size (about 1.5%
        // on Qwen3-1.7B-4bit, 0.4% on Llama-3.2-1B-4bit). A positional or
        // masking bug is an order of magnitude larger.
        let allowed = 0.03_f32.max(floor * 2.0);
        let mut checks = vec![
            ("steady vs unbatched", steady.unbatched, allowed),
            ("steady vs same-size batch", steady.same_size, allowed),
            ("leave", leave, allowed),
            ("join vs unbatched", join.unbatched, allowed),
            ("join vs same-size batch", join.same_size, allowed),
            ("ring positions", phases, 0.03),
            ("batch of one", single_batched, 0.03),
            ("batch of one, continued", single_continued, 0.03),
        ];
        if let Some(worst) = prefill {
            checks.push(("prefill vs unbatched", worst.unbatched, allowed));
            checks.push(("prefill vs same-size batch", worst.same_size, allowed));
        }
        for (name, worst, limit) in checks {
            assert!(
                !worst.is_nan(),
                "{name}: batched decode produced NaN logits"
            );
            assert!(
                worst < limit,
                "{name}: batched logits differ by {worst} of the largest logit (limit {limit}, noise floor {floor})"
            );
        }
    }

    /// Create a temp dir with a config.json containing the given `model_type` and
    /// return the `ModelConfig` result.
    fn config_for_model(model_type: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
        let result = ModelConfig::from_dir(dir.path());
        (dir, result)
    }

    /// Write arbitrary content to config.json in a temp dir and return
    /// the `ModelConfig` result.
    fn config_from_raw(content: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), content).unwrap();
        let result = ModelConfig::from_dir(dir.path());
        (dir, result)
    }

    #[test]
    fn model_config_from_dir_qwen2() {
        let (dir, result) = config_for_model("qwen2");
        let config = result.unwrap();
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.model_dir, dir.path());
    }

    #[test]
    fn model_config_from_dir_qwen3() {
        let (_dir, result) = config_for_model("qwen3");
        assert_eq!(result.unwrap().model_type, "qwen3");
    }

    #[test]
    fn model_config_from_dir_llama() {
        let (_dir, result) = config_for_model("llama");
        assert_eq!(result.unwrap().model_type, "llama");
    }

    #[test]
    fn model_config_from_dir_mistral() {
        let (_dir, result) = config_for_model("mistral");
        assert_eq!(result.unwrap().model_type, "mistral");
    }

    #[test]
    fn model_config_from_dir_qwen3_next() {
        let (_dir, result) = config_for_model("qwen3_next");
        assert_eq!(result.unwrap().model_type, "qwen3_next");
    }

    #[test]
    fn model_config_from_dir_qwen3_moe() {
        let (_dir, result) = config_for_model("qwen3_moe");
        assert_eq!(result.unwrap().model_type, "qwen3_moe");
    }

    #[test]
    fn model_config_from_dir_gemma2() {
        let (_dir, result) = config_for_model("gemma2");
        assert_eq!(result.unwrap().model_type, "gemma2");
    }

    #[test]
    fn model_config_from_dir_gemma3() {
        let (_dir, result) = config_for_model("gemma3");
        assert_eq!(result.unwrap().model_type, "gemma3");
    }

    #[test]
    fn model_config_from_dir_gemma3_text() {
        let (_dir, result) = config_for_model("gemma3_text");
        assert_eq!(result.unwrap().model_type, "gemma3_text");
    }

    #[test]
    fn model_config_from_dir_gemma4() {
        let (_dir, result) = config_for_model("gemma4");
        assert_eq!(result.unwrap().model_type, "gemma4");
    }

    #[test]
    fn model_config_from_dir_gemma4_text() {
        let (_dir, result) = config_for_model("gemma4_text");
        assert_eq!(result.unwrap().model_type, "gemma4_text");
    }

    #[test]
    fn model_config_from_dir_phi3() {
        let (_dir, result) = config_for_model("phi3");
        assert_eq!(result.unwrap().model_type, "phi3");
    }

    #[test]
    fn model_config_from_dir_starcoder2() {
        let (_dir, result) = config_for_model("starcoder2");
        assert_eq!(result.unwrap().model_type, "starcoder2");
    }

    #[test]
    fn model_config_from_dir_deepseek_v2() {
        let (_dir, result) = config_for_model("deepseek_v2");
        assert_eq!(result.unwrap().model_type, "deepseek_v2");
    }

    #[test]
    fn model_config_from_dir_qwen3_5() {
        let (_dir, result) = config_for_model("qwen3_5");
        assert_eq!(result.unwrap().model_type, "qwen3_5");
    }

    #[test]
    fn model_config_from_dir_qwen3_5_moe() {
        let (_dir, result) = config_for_model("qwen3_5_moe");
        assert_eq!(result.unwrap().model_type, "qwen3_5_moe");
    }

    #[test]
    fn model_config_from_dir_unsupported_model_type() {
        let (_dir, result) = config_for_model("gpt2");
        match result {
            Err(e) => assert!(e.to_string().contains("gpt2")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn model_config_from_dir_missing_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let err = ModelConfig::from_dir(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Io(_))));
    }

    #[test]
    fn model_config_from_dir_invalid_json() {
        let (_dir, result) = config_from_raw("not valid json {{{");
        let err = result.unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Json(_))));
    }

    #[test]
    fn model_config_from_dir_missing_model_type_field() {
        let (_dir, result) = config_from_raw(r#"{"vocab_size": 32000, "hidden_size": 4096}"#);
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            EngineError::Model(ModelError::UnsupportedModel(_))
        ));
    }

    #[test]
    fn load_model_routes_bonsai_q1_to_packed_engine() {
        // A bits=1 / group=128 qwen3 config now routes to the packed Bonsai-Q1
        // engine (its bits=1 kernels live in higgs-models::metal_kernel) instead
        // of being rejected up front. With no weights in the dir the load still
        // fails inside the engine — but it must no longer be gated out, and the
        // old "requires MLX bits=1" guard error must be gone.
        let (dir, _result) = config_from_raw(
            r#"{
                "model_type": "qwen3",
                "quantization": {"bits": 1, "group_size": 128}
            }"#,
        );
        match load_model(dir.path()) {
            Ok(_) => panic!("expected load failure: config-only dir has no weights"),
            Err(EngineError::Model(ModelError::UnsupportedModel(_))) => {
                panic!("Bonsai-Q1 must route to the packed engine, not be rejected as unsupported")
            }
            Err(err) => assert!(
                !err.to_string().contains("requires MLX bits=1"),
                "stale bits=1 guard error should be gone, got: {err}"
            ),
        }
    }

    #[test]
    fn load_tokenizer_missing_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        match load_tokenizer(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Tokenization error")),
            Ok(_) => panic!("Expected error for missing tokenizer.json"),
        }
    }
}
