//! Ignored benchmarks on whole models and prefill.

use super::prefill::load_test_model;
use crate::{
    qwen3_next::{
        cache::LayerCache,
        linear::quantized_forward,
        loading::{load_qwen3_5_model, load_qwen3_5_moe_model, load_qwen3_next_model},
        qgemv::qgemv_4bit,
    },
    utils::AttentionMask,
};
use mlx_rs::{
    Array, Stream,
    module::Module,
    ops::{self, indexing::IndexOp},
};

#[test]
#[ignore = "requires model files on disk"]
fn bench_actual_model_forward() {
    let Some(model_path) = std::env::var_os("HIGGS_QWEN3_NEXT_BENCH_MODEL") else {
        println!("Skipping: set HIGGS_QWEN3_NEXT_BENCH_MODEL to a local model directory");
        return;
    };
    let model_path = std::path::PathBuf::from(model_path);
    if !model_path.exists() {
        println!("Model not found at {}, skipping", model_path.display());
        return;
    }

    let mut model = load_qwen3_next_model(&model_path).unwrap();
    let mut cache: Vec<Option<LayerCache>> = Vec::new();

    // Prefill with a short prompt
    let prompt = Array::from_slice(&[9707u32, 1879], &[1, 2]);
    let prefill_out = model.forward(&prompt, None, &mut cache).unwrap();
    // Eval prefill outputs + cache states
    let mut to_eval: Vec<&Array> = vec![&prefill_out];
    for lc in &cache {
        if let Some(lc) = lc {
            match lc {
                LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        to_eval.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        to_eval.push(c);
                    }
                }
                LayerCache::KV(_) => {} // KV cache evals itself internally
            }
        }
    }
    mlx_rs::transforms::eval(to_eval).unwrap();

    // Get first token
    let logits = prefill_out.index((.., -1, ..));
    let token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
    mlx_rs::transforms::eval([&token]).unwrap();

    // Decode loop timing
    let mut current = token;
    for i in 0..22 {
        let input = current.index((.., ops::indexing::NewAxis));
        let t_fwd_start = std::time::Instant::now();
        let out = model.forward(&input, None, &mut cache).unwrap();
        let next = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
        let t_fwd = t_fwd_start.elapsed();

        let t_eval_start = std::time::Instant::now();
        // Eval next token AND all cache states (like Python does)
        let mut eval_list: Vec<&Array> = vec![&next];
        for lc in cache.iter() {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            eval_list.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            eval_list.push(c);
                        }
                    }
                    LayerCache::KV(_) => {}
                }
            }
        }
        mlx_rs::transforms::eval(eval_list).unwrap();
        let t_eval = t_eval_start.elapsed();

        let t_item_start = std::time::Instant::now();
        let _id: u32 = next.item_cast();
        let t_item = t_item_start.elapsed();

        let total = t_fwd + t_eval + t_item;
        if i < 5 || i >= 20 {
            println!(
                "Step {i}: fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms ({:.1} tok/s)",
                t_fwd.as_secs_f64() * 1000.0,
                t_eval.as_secs_f64() * 1000.0,
                t_item.as_secs_f64() * 1000.0,
                total.as_secs_f64() * 1000.0,
                1.0 / total.as_secs_f64(),
            );
        }
        current = next;
    }
}

#[test]
#[ignore = "requires model files on disk"]
fn bench_actual_qwen3_5_dense_decode() {
    use std::time::Instant;

    let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
    });
    if !std::path::Path::new(&model_path).exists() {
        println!("Model not found at {model_path}, skipping");
        return;
    }

    let prompt_len: i32 = std::env::var("BENCH_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let decode_steps: usize = std::env::var("BENCH_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let mut model = load_qwen3_5_model(&model_path).unwrap();
    let tokens: Vec<u32> = (0..prompt_len as u32)
        .map(|i| i % model.args.vocab_size as u32)
        .collect();
    let prompt = Array::from_slice(&tokens, &[1, prompt_len]);

    let mut cache: Vec<Option<LayerCache>> = Vec::new();
    let prefill_out = if prompt_len > 512 {
        model
            .forward_chunked(&prompt, None, &mut cache, 512)
            .unwrap()
    } else {
        model.forward(&prompt, None, &mut cache).unwrap()
    };

    let mut prefill_eval: Vec<&Array> = vec![&prefill_out];
    for lc in &cache {
        if let Some(lc) = lc {
            match lc {
                LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        prefill_eval.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        prefill_eval.push(c);
                    }
                }
                LayerCache::KV(_) => {}
            }
        }
    }
    mlx_rs::transforms::eval(prefill_eval).unwrap();

    let logits = prefill_out.index((.., -1, ..));
    let token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
    mlx_rs::transforms::eval([&token]).unwrap();

    let mut current = token;
    let mut total_forward_ns = 0u128;
    let mut total_eval_ns = 0u128;
    let mut total_item_ns = 0u128;

    for i in 0..decode_steps {
        let input = current.index((.., ops::indexing::NewAxis));

        let t0 = Instant::now();
        let out = model.forward(&input, None, &mut cache).unwrap();
        let next = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
        let fwd_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let mut eval_targets: Vec<&Array> = vec![&next];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            eval_targets.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            eval_targets.push(c);
                        }
                    }
                    LayerCache::KV(_) => {}
                }
            }
        }
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let eval_ns = t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let _: u32 = next.item_cast();
        let item_ns = t0.elapsed().as_nanos();

        total_forward_ns += fwd_ns;
        total_eval_ns += eval_ns;
        total_item_ns += item_ns;

        let total_ns = fwd_ns + eval_ns + item_ns;
        println!(
            "step={i:>2} fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms tok/s={:.2}",
            fwd_ns as f64 / 1e6,
            eval_ns as f64 / 1e6,
            item_ns as f64 / 1e6,
            total_ns as f64 / 1e6,
            1e9 / total_ns as f64,
        );
        current = next;
    }

    let steps = decode_steps as f64;
    let avg_total_ns = total_forward_ns + total_eval_ns + total_item_ns;
    println!(
        "AVG decode: fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms tok/s={:.2}",
        total_forward_ns as f64 / steps / 1e6,
        total_eval_ns as f64 / steps / 1e6,
        total_item_ns as f64 / steps / 1e6,
        avg_total_ns as f64 / steps / 1e6,
        steps * 1e9 / avg_total_ns as f64,
    );
}

#[test]
#[ignore = "requires model files on disk"]
fn bench_actual_qwen3_5_dense_decode_breakdown() {
    use std::time::Instant;

    let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
    });
    if !std::path::Path::new(&model_path).exists() {
        println!("Model not found at {model_path}, skipping");
        return;
    }

    let prompt_len: i32 = std::env::var("BENCH_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let decode_steps: usize = std::env::var("BENCH_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let mut model = load_qwen3_5_model(&model_path).unwrap();
    let tokens: Vec<u32> = (0..prompt_len as u32)
        .map(|i| i % model.args.vocab_size as u32)
        .collect();
    let prompt = Array::from_slice(&tokens, &[1, prompt_len]);

    let mut cache: Vec<Option<LayerCache>> = Vec::new();
    let prefill_out = if prompt_len > 512 {
        model
            .forward_chunked(&prompt, None, &mut cache, 512)
            .unwrap()
    } else {
        model.forward(&prompt, None, &mut cache).unwrap()
    };

    let mut prefill_eval: Vec<&Array> = vec![&prefill_out];
    for lc in &cache {
        if let Some(lc) = lc {
            match lc {
                LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        prefill_eval.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        prefill_eval.push(c);
                    }
                }
                LayerCache::KV(_) => {}
            }
        }
    }
    mlx_rs::transforms::eval(prefill_eval).unwrap();

    let logits = prefill_out.index((.., -1, ..));
    let mut current = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
    mlx_rs::transforms::eval([&current]).unwrap();

    let mut total_embed_ns = 0u128;
    let mut total_gdn_ns = 0u128;
    let mut total_attn_ns = 0u128;
    let mut total_mlp_ns = 0u128;
    let mut total_norm_resid_ns = 0u128;
    let mut total_lm_ns = 0u128;

    let mut gdn_layers = 0u32;
    let mut attn_layers = 0u32;

    for step in 0..decode_steps {
        let input = current.index((.., ops::indexing::NewAxis));

        let t0 = Instant::now();
        let mut h = model.model.embed_tokens.forward(&input).unwrap();
        mlx_rs::transforms::eval([&h]).unwrap();
        total_embed_ns += t0.elapsed().as_nanos();

        let mut step_gdn_ns = 0u128;
        let mut step_attn_ns = 0u128;
        let mut step_mlp_ns = 0u128;
        let mut step_norm_resid_ns = 0u128;
        let mut step_gdn_layers = 0u32;
        let mut step_attn_layers = 0u32;

        for (layer, layer_cache) in model.model.layers.iter_mut().zip(cache.iter_mut()) {
            let lc = layer_cache.as_mut().unwrap();

            let t0 = Instant::now();
            let normed = layer.input_layernorm.forward(&h).unwrap();
            mlx_rs::transforms::eval([&normed]).unwrap();
            step_norm_resid_ns += t0.elapsed().as_nanos();

            let t0 = Instant::now();
            let r = if layer.is_linear {
                let gdn = layer.linear_attn.as_mut().unwrap();
                let LayerCache::Arrays(sc) = lc else {
                    panic!("Expected ArraysCache");
                };
                let out = gdn.forward(&normed, None, sc).unwrap();
                let mut tgts: Vec<&Array> = vec![&out];
                if let Some(ref s) = sc.ssm_state {
                    tgts.push(s);
                }
                if let Some(ref c) = sc.conv_state {
                    tgts.push(c);
                }
                mlx_rs::transforms::eval(tgts).unwrap();
                step_gdn_layers += 1;
                out
            } else {
                let attn = layer.self_attn.as_mut().unwrap();
                let LayerCache::KV(kvc) = lc else {
                    panic!("Expected KVCache");
                };
                let out = attn.forward(&normed, None, kvc).unwrap();
                mlx_rs::transforms::eval([&out]).unwrap();
                step_attn_layers += 1;
                out
            };
            let op_ns = t0.elapsed().as_nanos();
            if layer.is_linear {
                step_gdn_ns += op_ns;
            } else {
                step_attn_ns += op_ns;
            }

            let t0 = Instant::now();
            let h2 = h.add(r).unwrap();
            let normed_post = layer.post_attention_layernorm.forward(&h2).unwrap();
            mlx_rs::transforms::eval([&normed_post]).unwrap();
            step_norm_resid_ns += t0.elapsed().as_nanos();

            let t0 = Instant::now();
            let mlp_out = layer.mlp.forward(&normed_post).unwrap();
            mlx_rs::transforms::eval([&mlp_out]).unwrap();
            step_mlp_ns += t0.elapsed().as_nanos();

            let t0 = Instant::now();
            h = h2.add(mlp_out).unwrap();
            mlx_rs::transforms::eval([&h]).unwrap();
            step_norm_resid_ns += t0.elapsed().as_nanos();
        }

        let t0 = Instant::now();
        h = model.model.norm.forward(&h).unwrap();
        mlx_rs::transforms::eval([&h]).unwrap();
        step_norm_resid_ns += t0.elapsed().as_nanos();

        let t0 = Instant::now();
        let logits = match model.lm_head.as_ref() {
            Some(head) => head.forward(&h).unwrap(),
            None => model.model.embed_tokens.as_linear(&h).unwrap(),
        };
        let next = ops::indexing::argmax_axis(&logits.index((.., -1, ..)), -1, false).unwrap();
        mlx_rs::transforms::eval([&logits, &next]).unwrap();
        total_lm_ns += t0.elapsed().as_nanos();

        let total_step_ns = step_gdn_ns + step_attn_ns + step_mlp_ns + step_norm_resid_ns;
        println!(
            "step={step:>2} total={:.2}ms gdn={:.2}ms attn={:.2}ms mlp={:.2}ms norm/resid={:.2}ms lm_head={:.2}ms tok/s={:.2}",
            (total_step_ns + total_embed_ns / (step as u128 + 1) + total_lm_ns / (step as u128 + 1))
                as f64
                / 1e6,
            step_gdn_ns as f64 / 1e6,
            step_attn_ns as f64 / 1e6,
            step_mlp_ns as f64 / 1e6,
            step_norm_resid_ns as f64 / 1e6,
            (total_lm_ns / (step as u128 + 1)) as f64 / 1e6,
            1e9 / ((step_gdn_ns + step_attn_ns + step_mlp_ns + step_norm_resid_ns)
                + total_embed_ns / (step as u128 + 1)
                + total_lm_ns / (step as u128 + 1)) as f64,
        );

        total_gdn_ns += step_gdn_ns;
        total_attn_ns += step_attn_ns;
        total_mlp_ns += step_mlp_ns;
        total_norm_resid_ns += step_norm_resid_ns;
        gdn_layers = step_gdn_layers;
        attn_layers = step_attn_layers;
        current = next;
    }

    let steps = decode_steps as f64;
    let total_ns = total_embed_ns
        + total_gdn_ns
        + total_attn_ns
        + total_mlp_ns
        + total_norm_resid_ns
        + total_lm_ns;
    let pct = |ns: u128| ns as f64 * 100.0 / total_ns as f64;

    println!(
        "AVG decode breakdown: total={:.2}ms tok/s={:.2}",
        total_ns as f64 / steps / 1e6,
        steps * 1e9 / total_ns as f64,
    );
    println!(
        "  embed:      {:>7.2}ms {:>5.1}%",
        total_embed_ns as f64 / steps / 1e6,
        pct(total_embed_ns),
    );
    println!(
        "  GDN x{}:    {:>7.2}ms {:>5.1}% [{:.2}ms/layer]",
        gdn_layers,
        total_gdn_ns as f64 / steps / 1e6,
        pct(total_gdn_ns),
        total_gdn_ns as f64 / steps / gdn_layers.max(1) as f64 / 1e6,
    );
    println!(
        "  Attn x{}:   {:>7.2}ms {:>5.1}% [{:.2}ms/layer]",
        attn_layers,
        total_attn_ns as f64 / steps / 1e6,
        pct(total_attn_ns),
        total_attn_ns as f64 / steps / attn_layers.max(1) as f64 / 1e6,
    );
    println!(
        "  MLP:        {:>7.2}ms {:>5.1}% [{:.2}ms/layer]",
        total_mlp_ns as f64 / steps / 1e6,
        pct(total_mlp_ns),
        total_mlp_ns as f64 / steps / (gdn_layers + attn_layers).max(1) as f64 / 1e6,
    );
    println!(
        "  norm/resid: {:>7.2}ms {:>5.1}%",
        total_norm_resid_ns as f64 / steps / 1e6,
        pct(total_norm_resid_ns),
    );
    println!(
        "  lm_head:    {:>7.2}ms {:>5.1}%",
        total_lm_ns as f64 / steps / 1e6,
        pct(total_lm_ns),
    );
}

#[test]
#[ignore = "requires model files on disk"]
fn bench_actual_qwen3_5_mtp_decode() {
    use std::time::Instant;

    let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
    });
    if !std::path::Path::new(&model_path).exists() {
        println!("Model not found at {model_path}, skipping");
        return;
    }

    let prompt_len: i32 = std::env::var("BENCH_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let target_tokens: usize = std::env::var("BENCH_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let mut model = load_qwen3_5_model(&model_path).unwrap();
    if !model.has_mtp() {
        println!("Model at {model_path} has no MTP head, skipping");
        return;
    }

    let tokens: Vec<u32> = (0..prompt_len as u32)
        .map(|i| i % model.args.vocab_size as u32)
        .collect();
    let prompt = Array::from_slice(&tokens, &[1, prompt_len]);

    let mut cache: Vec<Option<LayerCache>> = Vec::new();
    let prefill_out = if prompt_len > 512 {
        model
            .forward_chunked(&prompt, None, &mut cache, 512)
            .unwrap()
    } else {
        model.forward(&prompt, None, &mut cache).unwrap()
    };

    let mut prefill_eval: Vec<&Array> = vec![&prefill_out];
    for lc in &cache {
        if let Some(lc) = lc {
            match lc {
                LayerCache::Arrays(ac) => {
                    if let Some(ref s) = ac.ssm_state {
                        prefill_eval.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        prefill_eval.push(c);
                    }
                }
                LayerCache::KV(_) => {}
            }
        }
    }
    mlx_rs::transforms::eval(prefill_eval).unwrap();

    let logits = prefill_out.index((.., -1, ..));
    let first_token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
    mlx_rs::transforms::eval([&first_token]).unwrap();
    let first_token_id: u32 = first_token.item_cast();

    let mut mtp_cache = model.make_mtp_cache().unwrap();

    // Warm up speculative decode by confirming the first sampled token.
    let first_input = Array::from_slice(&[first_token_id as i32], &[1, 1]);
    let (hidden, logits) = model
        .forward_with_hidden(&first_input, None, &mut cache)
        .unwrap();
    let next_arr = ops::indexing::argmax_axis(&logits.index((.., -1, ..)), -1, false).unwrap();
    let h = hidden.index((.., -1.., ..));
    mlx_rs::transforms::eval([&next_arr, &h]).unwrap();

    let mut current_hidden = h;
    let mut confirmed_token_id: u32 = next_arr.item_cast();
    let mut emitted_tokens = 0usize;
    let mut accepted_cycles = 0usize;
    let mut total_cycles = 0usize;
    let mut total_ns = 0u128;

    while emitted_tokens < target_tokens {
        let t0 = Instant::now();

        let draft_logits = model
            .mtp_draft(&current_hidden, confirmed_token_id, &mut mtp_cache)
            .unwrap();
        let draft_token_arr =
            ops::indexing::argmax_axis(&draft_logits.index((.., -1, ..)), -1, false).unwrap();
        mlx_rs::transforms::eval([&draft_token_arr]).unwrap();
        let draft_token_id: u32 = draft_token_arr.item_cast();

        let confirmed_input = Array::from_slice(&[confirmed_token_id as i32], &[1, 1]);
        let (confirmed_hidden, confirmed_logits) = model
            .forward_with_hidden(&confirmed_input, None, &mut cache)
            .unwrap();
        let target_arr =
            ops::indexing::argmax_axis(&confirmed_logits.index((.., -1, ..)), -1, false).unwrap();
        let h_confirmed = confirmed_hidden.index((.., -1.., ..));
        mlx_rs::transforms::eval([&target_arr, &h_confirmed]).unwrap();
        let target_id: u32 = target_arr.item_cast();

        let emitted_this_cycle = if target_id == draft_token_id {
            accepted_cycles += 1;
            model
                .mtp_advance(&h_confirmed, draft_token_id, &mut mtp_cache)
                .unwrap();

            let draft_input = Array::from_slice(&[draft_token_id as i32], &[1, 1]);
            let (draft_hidden, draft_logits) = model
                .forward_with_hidden(&draft_input, None, &mut cache)
                .unwrap();
            let bonus_token =
                ops::indexing::argmax_axis(&draft_logits.index((.., -1, ..)), -1, false).unwrap();
            let h_last = draft_hidden.index((.., -1.., ..));
            mlx_rs::transforms::eval([&bonus_token, &h_last]).unwrap();
            current_hidden = h_last;
            confirmed_token_id = bonus_token.item_cast();
            2usize
        } else {
            current_hidden = h_confirmed;
            confirmed_token_id = target_id;
            1usize
        };

        total_cycles += 1;
        emitted_tokens += emitted_this_cycle;
        let cycle_ns = t0.elapsed().as_nanos();
        total_ns += cycle_ns;

        println!(
            "cycle={total_cycles:>2} emitted={emitted_this_cycle} total_tokens={emitted_tokens} total_ms={:.2} tok/s={:.2} accept_rate={:.1}%",
            cycle_ns as f64 / 1e6,
            emitted_this_cycle as f64 * 1e9 / cycle_ns as f64,
            accepted_cycles as f64 * 100.0 / total_cycles as f64,
        );
    }

    println!(
        "AVG MTP decode: cycles={} emitted={} avg_cycle_ms={:.2} tok/s={:.2} accept_rate={:.1}%",
        total_cycles,
        emitted_tokens,
        total_ns as f64 / total_cycles as f64 / 1e6,
        emitted_tokens as f64 * 1e9 / total_ns as f64,
        accepted_cycles as f64 * 100.0 / total_cycles as f64,
    );
}

/// Benchmark chunked vs full prefill TTFT.
///
/// Set env vars to control the benchmark:
/// - `BENCH_SEQ`: comma-separated sequence lengths (default: 512,1024,2048,5120,10240)
/// - `BENCH_CHUNK`: comma-separated chunk sizes (default: 128,256,512,1024)
/// - `BENCH_FULL_MAX`: max sequence length for full prefill baseline (default: 10240)
///
/// ```bash
/// cargo test -p higgs-models --release -- bench_chunked_prefill --nocapture --ignored
///
/// # Long sequences only:
/// BENCH_SEQ=10240,20480,40960 BENCH_CHUNK=256,512 BENCH_FULL_MAX=20480 \
///   cargo test -p higgs-models --release -- bench_chunked_prefill --nocapture --ignored
/// ```
#[test]
#[ignore = "requires model files on disk"]
fn bench_chunked_prefill() {
    use mlx_rs::transforms::eval;
    use std::time::Instant;

    let mut model = load_test_model();
    eprintln!(
        "Model: {} layers, hidden={}\n",
        model.args.num_hidden_layers, model.args.hidden_size,
    );

    let seq_lengths: Vec<i32> = std::env::var("BENCH_SEQ")
        .unwrap_or_else(|_| "512,1024,2048,5120,10240".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let chunk_sizes: Vec<i32> = std::env::var("BENCH_CHUNK")
        .unwrap_or_else(|_| "128,256,512,1024".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let full_max: i32 = std::env::var("BENCH_FULL_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10240);

    println!(
        "{:>7}  {:>6}  {:>10}  {:>10}  {:>8}",
        "T", "chunk", "full(ms)", "chunked(ms)", "ratio"
    );
    println!("{}", "-".repeat(50));

    for &seq_len in &seq_lengths {
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        let full_ms = if seq_len <= full_max {
            let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
            let t0 = Instant::now();
            let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
            eval([&logits_full]).unwrap();
            Some(t0.elapsed().as_secs_f64() * 1000.0)
        } else {
            None
        };

        for &chunk in &chunk_sizes {
            if chunk >= seq_len {
                continue;
            }

            let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
            let t0 = Instant::now();
            let logits_chunked = model
                .forward_chunked(&input, None, &mut cache_chunked, chunk)
                .unwrap();
            eval([&logits_chunked]).unwrap();
            let chunked_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let full_str = match full_ms {
                Some(ms) => format!("{ms:>10.0}"),
                None => format!("{:>10}", "—"),
            };
            let ratio_str = match full_ms {
                Some(ms) => format!("{:>7.2}x", ms / chunked_ms),
                None => format!("{:>8}", "—"),
            };

            println!("{seq_len:>7}  {chunk:>6}  {full_str}  {chunked_ms:>10.0}  {ratio_str}");
        }
        println!();
    }
}

/// Profile per-component TTFT breakdown for different sequence lengths.
///
/// Measures wall-clock TTFT (single eval) and per-component time with eval
/// barriers between embed, GDN, attention, MLP/MoE, norms, and lm_head.
///
/// ```bash
/// # Default model path: ~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit
/// cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
///
/// # Override model path:
/// HIGGS_MODEL_PATH=/path/to/model cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
/// ```
#[test]
#[ignore = "requires model files on disk"]
fn bench_prefill_breakdown() {
    use mlx_rs::transforms::eval;
    use std::time::Instant;

    let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
    });
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Model not found at {model_path}");
        eprintln!("Set HIGGS_MODEL_PATH env var to your model directory");
        return;
    }

    eprintln!("Loading model from {model_path} ...");
    let mut model = load_qwen3_5_moe_model(&model_path).unwrap();
    let n_layers = model.args.num_hidden_layers;
    let fa_interval = model.args.full_attention_interval;
    eprintln!(
        "Loaded: {n_layers} layers, hidden={}, fa_interval={fa_interval}",
        model.args.hidden_size,
    );

    // Warmup: prime Metal shaders + lazy dtype conversions
    {
        let w = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
        let mut wc: Vec<Option<LayerCache>> = Vec::new();
        let out = model.forward(&w, None, &mut wc).unwrap();
        eval([&out].into_iter()).unwrap();
    }

    let seq_lengths: &[i32] = &[128, 512, 1024, 2048, 5120];

    for &seq_len in seq_lengths {
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();

        // ----- Pass 1: real-world TTFT (no eval barriers) -----
        let input_a = Array::from_slice(&tokens, &[1, seq_len]);
        let mut cache_a: Vec<Option<LayerCache>> = Vec::new();

        let wall_start = Instant::now();
        let logits_a = model.forward(&input_a, None, &mut cache_a).unwrap();
        let mut eval_tgts: Vec<&Array> = vec![&logits_a];
        for lc in &cache_a {
            if let Some(LayerCache::Arrays(ac)) = lc {
                if let Some(ref s) = ac.ssm_state {
                    eval_tgts.push(s);
                }
                if let Some(ref c) = ac.conv_state {
                    eval_tgts.push(c);
                }
            }
        }
        eval(eval_tgts).unwrap();
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        // ----- Pass 2: per-component with eval barriers -----
        let input_b = Array::from_slice(&tokens, &[1, seq_len]);
        let mut cache_b: Vec<Option<LayerCache>> = model.make_cache();

        let fa_mask: Option<AttentionMask> = if seq_len > 1 {
            Some(AttentionMask::Causal)
        } else {
            None
        };

        // Embed
        let t0 = Instant::now();
        let mut h = model.model.embed_tokens.forward(&input_b).unwrap();
        eval([&h].into_iter()).unwrap();
        let ns_embed = t0.elapsed().as_nanos();

        let mut ns_gdn = 0u128;
        let mut ns_attn = 0u128;
        let mut ns_mlp = 0u128;
        let mut ns_norm = 0u128;
        let mut n_gdn = 0u32;
        let mut n_attn = 0u32;

        for (layer, layer_cache) in model.model.layers.iter_mut().zip(cache_b.iter_mut()) {
            let lc = layer_cache.as_mut().unwrap();
            let mask_ref = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            // Pre-attention norm
            let t0 = Instant::now();
            let normed = layer.input_layernorm.forward(&h).unwrap();
            eval([&normed].into_iter()).unwrap();
            ns_norm += t0.elapsed().as_nanos();

            // GDN or full attention
            let t0 = Instant::now();
            let r = if layer.is_linear {
                let gdn = layer.linear_attn.as_mut().unwrap();
                let LayerCache::Arrays(sc) = lc else {
                    panic!("Expected ArraysCache");
                };
                let out = gdn.forward(&normed, mask_ref, sc).unwrap();
                let mut tgts: Vec<&Array> = vec![&out];
                if let Some(ref s) = sc.ssm_state {
                    tgts.push(s);
                }
                if let Some(ref c) = sc.conv_state {
                    tgts.push(c);
                }
                eval(tgts).unwrap();
                n_gdn += 1;
                ns_gdn += t0.elapsed().as_nanos();
                out
            } else {
                let attn = layer.self_attn.as_mut().unwrap();
                let LayerCache::KV(kvc) = lc else {
                    panic!("Expected KVCache");
                };
                let out = attn.forward(&normed, mask_ref, kvc).unwrap();
                eval([&out].into_iter()).unwrap();
                n_attn += 1;
                ns_attn += t0.elapsed().as_nanos();
                out
            };

            // Residual + post-attention norm
            let t0 = Instant::now();
            let h2 = h.add(r).unwrap();
            let normed_post = layer.post_attention_layernorm.forward(&h2).unwrap();
            eval([&normed_post].into_iter()).unwrap();
            ns_norm += t0.elapsed().as_nanos();

            // MLP / MoE
            let t0 = Instant::now();
            let mlp_out = layer.mlp.forward(&normed_post).unwrap();
            eval([&mlp_out].into_iter()).unwrap();
            ns_mlp += t0.elapsed().as_nanos();

            // Final residual
            let t0 = Instant::now();
            h = h2.add(mlp_out).unwrap();
            eval([&h].into_iter()).unwrap();
            ns_norm += t0.elapsed().as_nanos();
        }

        // Final norm
        let t0 = Instant::now();
        h = model.model.norm.forward(&h).unwrap();
        eval([&h].into_iter()).unwrap();
        ns_norm += t0.elapsed().as_nanos();

        // LM head
        let t0 = Instant::now();
        let _logits = match model.lm_head.as_ref() {
            Some(head) => head.forward(&h).unwrap(),
            None => model.model.embed_tokens.as_linear(&h).unwrap(),
        };
        eval([&_logits].into_iter()).unwrap();
        let ns_lm = t0.elapsed().as_nanos();

        // ----- Report -----
        let barrier_total = ns_embed + ns_gdn + ns_attn + ns_mlp + ns_norm + ns_lm;
        let ms = |ns: u128| ns as f64 / 1e6;
        let pct = |ns: u128| ns as f64 / barrier_total as f64 * 100.0;
        let n_total = n_gdn + n_attn;

        println!();
        println!("==== T = {seq_len} ====");
        println!("  Wall TTFT (no barriers):  {:>8.1}ms", wall_ms);
        println!(
            "  Sum  (eval barriers):     {:>8.1}ms  (barrier overhead: {:.1}ms)",
            ms(barrier_total),
            ms(barrier_total) - wall_ms,
        );
        println!();
        println!(
            "  embed:            {:>8.1}ms  {:>5.1}%",
            ms(ns_embed),
            pct(ns_embed),
        );
        println!(
            "  GDN ({n_gdn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
            ms(ns_gdn),
            pct(ns_gdn),
            ms(ns_gdn) / n_gdn.max(1) as f64,
        );
        println!(
            "  Attn ({n_attn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
            ms(ns_attn),
            pct(ns_attn),
            ms(ns_attn) / n_attn.max(1) as f64,
        );
        println!(
            "  MLP/MoE:          {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
            ms(ns_mlp),
            pct(ns_mlp),
            ms(ns_mlp) / n_total.max(1) as f64,
        );
        println!(
            "  norms+residual:   {:>8.1}ms  {:>5.1}%",
            ms(ns_norm),
            pct(ns_norm),
        );
        println!(
            "  lm_head:          {:>8.1}ms  {:>5.1}%",
            ms(ns_lm),
            pct(ns_lm),
        );
        println!(
            "  ---- GDN share of wall TTFT: {:.1}%",
            ms(ns_gdn) / wall_ms * 100.0,
        );
    }
}

/// Benchmark helper: time GEMV vs quantized_matmul for given dims.
fn bench_gemv_at(n: i32, k: i32, group_size: i32, iters: usize) {
    use mlx_rs::Dtype;

    let x = mlx_rs::with_stream(&Stream::default(), || {
        mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 1, k], None)
    })
    .unwrap()
    .as_dtype(Dtype::Bfloat16)
    .unwrap();

    let w_dense = mlx_rs::with_stream(&Stream::default(), || {
        mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[n, k], None)
    })
    .unwrap();
    let (w_q, scales, biases) =
        crate::test_support::quantize_affine(&w_dense, group_size, 4).unwrap();
    mlx_rs::transforms::eval([&w_q, &scales, &biases, &x]).unwrap();

    // Warmup
    for _ in 0..5 {
        let r = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
        let g = qgemv_4bit(&x, &w_q, &scales, &biases, group_size).unwrap();
        mlx_rs::transforms::eval([&r, &g]).unwrap();
    }

    // Bench quantized_matmul
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
    }
    let qmm_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Bench custom GEMV
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = qgemv_4bit(&x, &w_q, &scales, &biases, group_size).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
    }
    let gemv_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let ratio = qmm_us / gemv_us;
    println!("  N={n:>5} K={k:>5} | qmm={qmm_us:>7.0}μs  gemv={gemv_us:>7.0}μs  ratio={ratio:.2}x");
}

#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_qgemv_vs_quantized_matmul() {
    println!("=== GEMV vs quantized_matmul (gs=64) ===");
    // Small
    bench_gemv_at(128, 512, 64, 50);
    bench_gemv_at(256, 1024, 64, 50);
    // Medium
    bench_gemv_at(1024, 2048, 64, 50);
    bench_gemv_at(2048, 2048, 64, 50);
    bench_gemv_at(5504, 2048, 64, 50);
    bench_gemv_at(5120, 5120, 64, 30);
    // 27B dense MLP actual dims (hidden=5120, intermediate=17408)
    println!("--- 27B dense MLP dims ---");
    bench_gemv_at(34816, 5120, 64, 20); // gate+up fused
    bench_gemv_at(5120, 17408, 64, 20); // down projection
    bench_gemv_at(248_320, 5120, 64, 5); // tied lm_head / embedding projection
}
