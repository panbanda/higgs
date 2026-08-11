//! Microbenchmark for filtered sampling (top-k / top-p / min-p).
//!
//! Run with:
//!   `cargo bench -p higgs-models --bench sample_filtered`
//!
//! The filtered-sampling path runs once per decoded token whenever any of
//! `top_k`, `top_p < 1.0`, or `min_p` is set. The dominant cost in the current
//! implementation is `argsort` over the full vocab; this bench measures that.

#![allow(
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::unusual_byte_groupings,
    clippy::disallowed_methods
)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use higgs_models::{SamplingParams, sample};
use mlx_rs::Array;
use mlx_rs::transforms::eval;

fn make_logits(vocab: usize) -> Array {
    let mut data = vec![0.0_f32; vocab];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for slot in &mut data {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (state >> 33) as u32;
        let unit = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000) - 1.0;
        *slot = unit.mul_add(20.0, -10.0);
    }
    let vocab_i32: i32 = vocab.try_into().expect("vocab fits in i32");
    Array::from_slice(&data, &[1, vocab_i32])
}

fn bench_sample_filtered(c: &mut Criterion) {
    // (label, vocab_size)
    let vocab_configs: &[(&str, usize)] = &[("vocab=32k", 32_000), ("vocab=152k", 152_064)];
    // top_k values to exercise. None = full sort path.
    let top_k_values: &[Option<u32>] = &[
        Some(40), // common chat default
        Some(100),
        Some(1024),
        None, // full-sort path; baseline
    ];

    let mut group = c.benchmark_group("sample_filtered");

    for &(vocab_label, vocab) in vocab_configs {
        let logits = make_logits(vocab);
        eval([&logits]).expect("eval logits");

        for &top_k in top_k_values {
            let params = SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k,
                min_p: Some(0.05),
                repetition_penalty: None,
                frequency_penalty: None,
                presence_penalty: None,
                speculation: higgs_models::Speculation::Auto,
                thinking_budget: None,
            };
            let label = format!(
                "{vocab_label}/top_k={}",
                top_k.map_or_else(|| "none".to_owned(), |k| k.to_string())
            );

            group.bench_with_input(BenchmarkId::from_parameter(&label), &(), |b, ()| {
                b.iter(|| {
                    let result = sample(black_box(&logits), black_box(&params)).expect("sample");
                    eval([&result]).expect("eval");
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_sample_filtered);
criterion_main!(benches);
