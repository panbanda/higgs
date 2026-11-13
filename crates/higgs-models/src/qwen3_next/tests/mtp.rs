//! Tests for MTP draft and advance.

use super::valid_causal_lm_args;
use crate::{cache::KeyValueCache, qwen3_next::model::Qwen3NextCausalLM};
use mlx_rs::{Array, Stream};

#[test]
#[ignore = "requires real model weights; placeholder test tensors cannot run MTP forward"]
fn test_mtp_draft_advances_cache_for_confirmed_token() {
    let stream = Stream::new();
    mlx_rs::with_stream(&stream, || {
        let mut args = valid_causal_lm_args();
        args.mtp_num_hidden_layers = 1;
        let mut model = Qwen3NextCausalLM::new(args).unwrap();
        let mut mtp_cache = model.make_mtp_cache().unwrap();
        let hidden = Array::zeros::<f32>(&[1, 1, model.args.hidden_size]).unwrap();

        let draft = model.mtp_draft(&hidden, 0, &mut mtp_cache).unwrap();
        draft.eval().unwrap();

        assert_eq!(mtp_cache[0].offset(), 1);
    });
}

#[test]
#[ignore = "requires real model weights; placeholder test tensors cannot run MTP forward"]
fn test_mtp_advance_appends_accepted_token_state() {
    let stream = Stream::new();
    mlx_rs::with_stream(&stream, || {
        let mut args = valid_causal_lm_args();
        args.mtp_num_hidden_layers = 1;
        let mut model = Qwen3NextCausalLM::new(args).unwrap();
        let mut mtp_cache = model.make_mtp_cache().unwrap();
        let hidden = Array::zeros::<f32>(&[1, 1, model.args.hidden_size]).unwrap();

        let draft = model.mtp_draft(&hidden, 0, &mut mtp_cache).unwrap();
        draft.eval().unwrap();
        model.mtp_advance(&hidden, 1, &mut mtp_cache).unwrap();

        assert_eq!(mtp_cache[0].offset(), 2);
    });
}

#[test]
#[ignore = "requires real model weights; placeholder test tensors cannot run MTP forward"]
fn test_mtp_advance_many_appends_accepted_token_states() {
    let stream = Stream::new();
    mlx_rs::with_stream(&stream, || {
        let mut args = valid_causal_lm_args();
        args.mtp_num_hidden_layers = 1;
        let mut model = Qwen3NextCausalLM::new(args).unwrap();
        let mut mtp_cache = model.make_mtp_cache().unwrap();
        let hidden = Array::zeros::<f32>(&[1, 2, model.args.hidden_size]).unwrap();

        model
            .mtp_advance_many(&hidden, &[1, 2], &mut mtp_cache)
            .unwrap();

        assert_eq!(mtp_cache[0].offset(), 2);
    });
}

#[test]
fn test_mtp_advance_many_rejects_cache_layer_mismatch() {
    let err =
        Qwen3NextCausalLM::validate_mtp_advance_many_shape(&[1, 1, 256], 0, 1, 1).unwrap_err();

    assert!(
        err.to_string().contains("mtp_cache length"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_mtp_advance_many_rejects_hidden_sequence_mismatch() {
    let err =
        Qwen3NextCausalLM::validate_mtp_advance_many_shape(&[1, 2, 256], 1, 1, 1).unwrap_err();

    assert!(
        err.to_string().contains("hidden sequence length"),
        "unexpected error: {err}"
    );
}
