//! Activation helpers, including the compiled fused variants.

use super::flags::compiled_gating_enabled;
use mlx_rs::{Array, error::Exception, fast, nn};
use std::cell::RefCell;

pub fn swiglu(gate: &Array, x: &Array) -> Result<Array, Exception> {
    gate.multiply(nn::sigmoid(gate)?)?.multiply(x)
}

pub(super) fn silu_direct(x: &Array) -> Result<Array, Exception> {
    x.multiply(nn::sigmoid(x)?)
}

fn compiled_silu_mul((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

type CompiledSiluMulFn = dyn for<'a> FnMut((&'a Array, &'a Array)) -> Result<Array, Exception>;
type CompiledSigmoidMulFn = dyn for<'a> FnMut((&'a Array, &'a Array)) -> Result<Array, Exception>;
type CompiledGdnOutputGateFn =
    dyn for<'a> FnMut((&'a Array, &'a Array, &'a Array)) -> Result<Array, Exception>;

thread_local! {
    static COMPILED_SILU_MUL_FN: RefCell<Option<Box<CompiledSiluMulFn>>> = RefCell::new(None);
    static COMPILED_SIGMOID_MUL_FN: RefCell<Option<Box<CompiledSigmoidMulFn>>> = RefCell::new(None);
    static COMPILED_GDN_OUTPUT_GATE_FN: RefCell<Option<Box<CompiledGdnOutputGateFn>>> =
        RefCell::new(None);
}

pub(super) fn silu_mul(gate: &Array, x: &Array) -> Result<Array, Exception> {
    if compiled_gating_enabled() {
        COMPILED_SILU_MUL_FN.with(|cell| {
            let mut guard = cell.borrow_mut();
            let compiled = guard.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    compiled_silu_mul,
                    None,
                ))
            });
            compiled((gate, x))
        })
    } else {
        nn::silu(gate)?.multiply(x)
    }
}

fn compiled_sigmoid_mul((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::sigmoid(gate)?.multiply(x)
}

pub(super) fn sigmoid_mul(gate: &Array, x: &Array) -> Result<Array, Exception> {
    if compiled_gating_enabled() {
        COMPILED_SIGMOID_MUL_FN.with(|cell| {
            let mut guard = cell.borrow_mut();
            let compiled = guard.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    compiled_sigmoid_mul,
                    None,
                ))
            });
            compiled((gate, x))
        })
    } else {
        nn::sigmoid(gate)?.multiply(x)
    }
}

fn compiled_gdn_output_gate((y, weight, z): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let normed = fast::rms_norm(y, Some(weight), 1e-6)?;
    nn::silu(z)?.multiply(&normed)
}

pub(super) fn gdn_output_gate(
    y: &Array,
    weight: &Array,
    eps: f32,
    z: &Array,
) -> Result<Array, Exception> {
    if compiled_gating_enabled() && (eps - 1e-6).abs() <= f32::EPSILON {
        COMPILED_GDN_OUTPUT_GATE_FN.with(|cell| {
            let mut guard = cell.borrow_mut();
            let compiled = guard.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    compiled_gdn_output_gate,
                    None,
                ))
            });
            compiled((y, weight, z))
        })
    } else {
        let normed = fast::rms_norm(y, Some(weight), eps)?;
        nn::silu(z)?.multiply(&normed)
    }
}
