//! Process-wide MLX memory settings: the allocator cache and the wired limit.

#[allow(unsafe_code)]
pub fn clear_allocator_cache_if(enabled: bool, reason: &str) {
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

/// The device's `max_recommended_working_set_size`, if MLX reports one.
#[allow(unsafe_code)]
fn max_recommended_working_set() -> Option<usize> {
    unsafe {
        let mut info = mlx_sys::mlx_device_info_new();
        let mut dev = mlx_sys::mlx_device_new();
        mlx_sys::mlx_get_default_device(&raw mut dev);
        let mut max_rec: usize = 0;
        let key = c"max_recommended_working_set_size";
        let found = mlx_sys::mlx_device_info_get(&raw mut info, dev) == 0
            && mlx_sys::mlx_device_info_get_size(&raw mut max_rec, info, key.as_ptr()) == 0
            && max_rec > 0;
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
        found.then_some(max_rec)
    }
}

fn legacy_limits_requested() -> bool {
    matches!(
        std::env::var("HIGGS_WIRED_LIMIT_MODE").ok().as_deref(),
        Some("legacy" | "safe" | "caps")
    )
}

fn mem_limits_disabled() -> bool {
    std::env::var("HIGGS_NO_MEM_LIMIT").is_ok()
}

/// Bound the memory that MLX keeps for reuse after it frees a buffer.
///
/// MLX's own default is the size of the memory limit, so a long-running server keeps almost
/// every buffer that it ever freed. `explicit` comes from `memory.mlx_cache_limit`; without
/// it the limit follows the device (see [`default_cache_limit`]). `HIGGS_NO_MEM_LIMIT=1`
/// leaves MLX's default in place.
#[allow(unsafe_code)]
pub fn set_cache_limit(explicit: Option<usize>) {
    if mem_limits_disabled() {
        tracing::info!("MLX cache limit left at the MLX default (HIGGS_NO_MEM_LIMIT is set)");
        return;
    }
    let Some(max_rec) = max_recommended_working_set() else {
        if let Some(limit) = explicit {
            apply_cache_limit(limit);
        }
        return;
    };
    let limit = explicit.unwrap_or_else(|| default_cache_limit(max_rec, legacy_limits_requested()));
    apply_cache_limit(limit);
}

#[allow(unsafe_code)]
fn apply_cache_limit(limit: usize) {
    let mut previous: usize = 0;
    unsafe { mlx_sys::mlx_set_cache_limit(&raw mut previous, limit) };
    tracing::info!(
        cache_limit_mb = limit / (1024 * 1024),
        previous_mb = previous / (1024 * 1024),
        "Configured MLX cache limit",
    );
}

/// The cache limit when the config does not set one.
pub const fn default_cache_limit(max_recommended: usize, legacy: bool) -> usize {
    if legacy {
        return max_recommended / 2;
    }
    let floor = AUTOMATIC_CACHE_FLOOR;
    let share = max_recommended / AUTOMATIC_CACHE_DIVISOR;
    if share > floor { share } else { floor }
}

const AUTOMATIC_CACHE_FLOOR: usize = 1 << 30;
const AUTOMATIC_CACHE_DIVISOR: usize = 16;

/// Configure MLX's large-model working set on Apple Silicon.
///
/// By default we mirror upstream `mlx-lm` and raise MLX's wired limit to the
/// device's `max_recommended_working_set_size`. Set
/// `HIGGS_WIRED_LIMIT_MODE=legacy` to restore the older conservative memory cap, or
/// `HIGGS_NO_MEM_LIMIT=1` to skip both.
#[allow(unsafe_code)]
pub fn raise_wired_limit_if(enabled: bool) {
    if !enabled {
        tracing::info!("MLX wired-limit escalation disabled by config");
        return;
    }
    let Some(max_rec) = max_recommended_working_set() else {
        return;
    };
    let legacy = legacy_limits_requested();
    if mem_limits_disabled() {
        tracing::info!(
            mode = if legacy { "legacy" } else { "mlx_wired_limit" },
            max_recommended_mb = max_rec / (1024 * 1024),
            "Skipped MLX memory-limit configuration",
        );
        return;
    }
    if legacy {
        let mem_limit = max_rec * 3 / 4;
        let mut previous: usize = 0;
        unsafe { mlx_sys::mlx_set_memory_limit(&raw mut previous, mem_limit) };
        tracing::info!(
            mode = "legacy",
            max_recommended_mb = max_rec / (1024 * 1024),
            memory_limit_mb = mem_limit / (1024 * 1024),
            prev_mem_mb = previous / (1024 * 1024),
            "Configured MLX legacy memory cap",
        );
    } else {
        let mut previous: usize = 0;
        unsafe { mlx_sys::mlx_set_wired_limit(&raw mut previous, max_rec) };
        tracing::info!(
            mode = "mlx_wired_limit",
            max_recommended_mb = max_rec / (1024 * 1024),
            wired_limit_mb = max_rec / (1024 * 1024),
            prev_wired_mb = previous / (1024 * 1024),
            "Configured MLX wired limit",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::default_cache_limit;

    const GIB: usize = 1 << 30;

    #[test]
    fn the_automatic_limit_is_a_share_of_the_working_set_with_a_floor() {
        assert_eq!(default_cache_limit(96 * GIB, false), 6 * GIB);
        assert_eq!(default_cache_limit(10 * GIB + GIB * 7 / 10, false), GIB);
        assert_eq!(default_cache_limit(GIB, false), GIB);
    }

    #[test]
    fn the_legacy_limit_is_half_the_working_set() {
        assert_eq!(default_cache_limit(96 * GIB, true), 48 * GIB);
    }
}
