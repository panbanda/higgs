//! Environment-driven feature switches and hardware-dependent defaults.

use std::sync::OnceLock;

static COMPILED_GATING_ENABLED: OnceLock<bool> = OnceLock::new();
static APPLE_CPU_BRAND: OnceLock<Option<String>> = OnceLock::new();
static COMPILED_GDN_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();
static ASYNC_LAYER_STATE_EVAL_ENABLED: OnceLock<bool> = OnceLock::new();

fn parse_compiled_gating_enabled(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

pub(super) fn apple_cpu_brand() -> Option<&'static str> {
    APPLE_CPU_BRAND
        .get_or_init(|| {
            #[cfg(target_os = "macos")]
            {
                std::process::Command::new("sysctl")
                    .args(["-n", "machdep.cpu.brand_string"])
                    .output()
                    .ok()
                    .filter(|out| out.status.success())
                    .and_then(|out| String::from_utf8(out.stdout).ok())
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        })
        .as_deref()
}

pub(super) fn should_force_dense_decode_safe_defaults_for_brand(brand: Option<&str>) -> bool {
    matches!(brand.map(str::trim), Some("Apple M4"))
}

pub(super) fn compiled_gating_enabled() -> bool {
    *COMPILED_GATING_ENABLED.get_or_init(|| {
        std::env::var("HIGGS_COMPILED_GATING")
            .ok()
            .is_none_or(|raw| parse_compiled_gating_enabled(Some(raw.as_str())))
    })
}

pub(super) fn compiled_gdn_decode_enabled() -> bool {
    *COMPILED_GDN_DECODE_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIGGS_COMPILED_GDN_DECODE")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref(),
            Some("1" | "true" | "on" | "yes")
        )
    })
}

pub(super) fn async_layer_state_eval_enabled() -> bool {
    *ASYNC_LAYER_STATE_EVAL_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIGGS_ASYNC_LAYER_STATE_EVAL")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref(),
            Some("1" | "true" | "on" | "yes")
        )
    })
}

static QGEMV_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static GATED_DELTA_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_GEMV_ENABLED: OnceLock<bool> = OnceLock::new();
static QGEMV_NSG_OVERRIDE: OnceLock<Option<i32>> = OnceLock::new();
static DENSE_FFN_GEMV_MODE: OnceLock<DenseFfnGemvMode> = OnceLock::new();
static DENSE_FFN_FUSE_GATE_UP: OnceLock<bool> = OnceLock::new();
static MOE_FFN_FUSE_GATE_UP: OnceLock<bool> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DenseFfnGemvMode {
    Both,
    FusedOnly,
    DownOnly,
    Off,
}

pub(super) fn decode_gemv_enabled() -> bool {
    *DECODE_GEMV_ENABLED.get_or_init(|| std::env::var("HIGGS_ENABLE_SELECTED_DECODE_GEMV").is_ok())
}

fn truthy_env_var(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

pub(super) fn parse_dense_ffn_gemv_mode(raw: Option<&str>) -> DenseFfnGemvMode {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("fused" | "fused_only") => DenseFfnGemvMode::FusedOnly,
        Some("down" | "down_only") => DenseFfnGemvMode::DownOnly,
        Some("off" | "none") => DenseFfnGemvMode::Off,
        _ => DenseFfnGemvMode::Both,
    }
}

pub(super) fn dense_ffn_gemv_mode() -> DenseFfnGemvMode {
    *DENSE_FFN_GEMV_MODE.get_or_init(|| {
        parse_dense_ffn_gemv_mode(std::env::var("HIGGS_QGEMV_FFN_MODE").ok().as_deref())
    })
}

pub(super) fn dense_ffn_fuse_gate_up() -> bool {
    *DENSE_FFN_FUSE_GATE_UP.get_or_init(|| {
        std::env::var("HIGGS_DENSE_FFN_GATE_UP").ok().map_or_else(
            || !should_force_dense_decode_safe_defaults_for_brand(apple_cpu_brand()),
            |raw| {
                !matches!(
                    Some(raw.trim().to_ascii_lowercase()).as_deref(),
                    Some("separate" | "split" | "0" | "false" | "off")
                )
            },
        )
    })
}

pub(super) fn moe_ffn_fuse_gate_up() -> bool {
    *MOE_FFN_FUSE_GATE_UP.get_or_init(|| truthy_env_var("HIGGS_MOE_FFN_GATE_UP"))
}

pub(super) fn qgemv_config_cache_enabled() -> bool {
    *QGEMV_CONFIG_CACHE_ENABLED.get_or_init(|| truthy_env_var("HIGGS_CACHE_QGEMV_CONFIGS"))
}

pub(super) fn gated_delta_config_cache_enabled() -> bool {
    *GATED_DELTA_CONFIG_CACHE_ENABLED.get_or_init(|| {
        std::env::var("HIGGS_CACHE_GATED_DELTA_CONFIGS")
            .ok()
            .is_none_or(|raw| {
                matches!(
                    Some(raw.trim().to_ascii_lowercase()).as_deref(),
                    Some("1" | "true" | "on" | "yes")
                )
            })
    })
}

pub(super) fn qgemv_nsg_override() -> Option<i32> {
    *QGEMV_NSG_OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_QGEMV_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&n| matches!(n, 4 | 8 | 16 | 32))
    })
}
