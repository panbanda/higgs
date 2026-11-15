//! Process-wide handle wrapper for compiled Metal kernels.

/// Wrapper for the cached `GatedDeltaNet` Metal kernel object.
pub(super) struct CachedMetalKernel(pub(super) mlx_sys::mlx_fast_metal_kernel);

// SAFETY: The kernel handle is created once during initialization and used
// read-only thereafter (only passed as an argument to `mlx_fast_metal_kernel_apply`).
// No mutable state is shared across threads.
#[allow(unsafe_code)]
unsafe impl Send for CachedMetalKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for CachedMetalKernel {}

impl Drop for CachedMetalKernel {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}
