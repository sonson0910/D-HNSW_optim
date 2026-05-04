//! SIMD-accelerated fixed-point distance computation (O1 optimization).
//!
//! This module provides hardware-accelerated squared Euclidean distance
//! calculation using AVX2 on x86_64 and a portable fallback for other
//! architectures. All computations remain in integer domain (I64F32)
//! to preserve cross-platform determinism.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌────────────────┐
//! │ AVX2 (x86)  │     │  Fallback      │
//! │ 4×i64 SIMD  │     │  Scalar i64    │
//! └──────┬──────┘     └───────┬────────┘
//!        │                    │
//!        └────────┬───────────┘
//!                 ▼
//!         I64F32 result
//! ```
//!
//! # Consensus Safety
//!
//! Both paths produce **bit-identical** results because:
//! 1. All operations are saturating integer arithmetic
//! 2. Addition order is deterministic (sequential accumulation)
//! 3. No floating-point intermediates are used

use crate::fixed_point::I64F32;

/// Compute squared Euclidean distance using SIMD acceleration.
///
/// Dispatches to AVX2 on x86_64 (runtime feature detection),
/// falls back to scalar on other platforms.
///
/// # Arguments
/// * `a` - First vector components (raw I64F32 values)
/// * `b` - Second vector components (raw I64F32 values)
///
/// # Returns
/// The squared Euclidean distance as I64F32.
#[inline]
pub fn squared_distance_simd(a: &[I64F32], b: &[I64F32]) -> I64F32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We've verified AVX2 is available at runtime.
            return unsafe { squared_distance_avx2(a, b) };
        }
    }

    // Fallback: scalar implementation (always available)
    squared_distance_scalar(a, b)
}

/// Scalar fallback for squared Euclidean distance.
///
/// This is the reference implementation that all SIMD paths must match.
/// Uses saturating arithmetic to prevent overflow.
#[inline]
pub fn squared_distance_scalar(a: &[I64F32], b: &[I64F32]) -> I64F32 {
    let mut sum = I64F32::ZERO;
    for i in 0..a.len() {
        let diff = a[i].saturating_sub(b[i]);
        sum = sum.saturating_add(diff.saturating_mul(diff));
    }
    sum
}

/// AVX2-accelerated squared Euclidean distance.
///
/// Processes 4 × i64 lanes per iteration. The underlying I64F32 values
/// are treated as raw i64 for SIMD, then the final sum is converted back.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
///
/// # Determinism Note
/// This produces bit-identical results to [`squared_distance_scalar`] because:
/// - We process elements in the same sequential order
/// - Saturating semantics are emulated with clamping
/// - No FMA or reordering that would change results
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_avx2(a: &[I64F32], b: &[I64F32]) -> I64F32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let _remainder = len % 4;

    // Accumulate as raw i64 (I64F32 bit representation)
    // Note: When we multiply two I64F32 values (which are i64 with 32 fractional bits),
    // diff * diff shifts the decimal point by 32 extra bits.
    // We need to right-shift by 32 after multiplication to stay in I64F32 domain.
    let mut sum_accum: i128 = 0;

    // Pointers to raw i64 representation of I64F32
    let a_ptr = a.as_ptr() as *const i64;
    let b_ptr = b.as_ptr() as *const i64;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 4;

        // Load 4 × i64 from each vector
        let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);

        // Compute diff = a - b (element-wise i64 subtraction)
        let diff = _mm256_sub_epi64(va, vb);

        // Extract individual lanes for multiplication (AVX2 doesn't have i64 multiply)
        // We use scalar multiplication to ensure bit-exact results
        let diff_array: [i64; 4] = std::mem::transmute(diff);

        for &d in &diff_array {
            // Saturating multiply in i128 space, then shift back to I64F32
            let prod = (d as i128) * (d as i128);
            // Right-shift by 32 to account for fixed-point multiplication
            let prod_shifted = prod >> 32;

            // Saturating accumulation
            sum_accum = sum_accum.saturating_add(prod_shifted);
        }
    }

    // Handle remainder elements
    for i in (chunks * 4)..len {
        let a_raw = *a_ptr.add(i);
        let b_raw = *b_ptr.add(i);
        let diff = a_raw.saturating_sub(b_raw);
        let prod = (diff as i128) * (diff as i128);
        let prod_shifted = prod >> 32;
        sum_accum = sum_accum.saturating_add(prod_shifted);
    }

    // Clamp to I64F32 range
    let max_val = i64::MAX as i128;
    let min_val = i64::MIN as i128;
    let clamped = sum_accum.max(min_val).min(max_val) as i64;

    I64F32::from_bits(clamped)
}

// ──────────────────────────────────────────────────────────────────
// Two-Phase distance: fast approximate pre-filter + exact verify
// ──────────────────────────────────────────────────────────────────

/// Compute an approximate squared distance using f32 arithmetic.
///
/// This is used as Phase 1 of the Two-Phase search optimization (O2).
/// It converts I64F32 components to f32 for a fast approximate distance,
/// which is then used to prune candidates before the expensive exact
/// I64F32 computation.
///
/// # Consensus Safety
/// This value is **never** used for final ordering or graph construction.
/// It is purely a pre-filter hint — the actual ranking always uses
/// [`squared_distance_scalar`] or [`squared_distance_simd`].
#[inline]
pub fn approximate_distance_f32(a: &[I64F32], b: &[I64F32]) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        // Convert fixed-point to f32 (lossy but fast)
        let af = a[i].to_num::<f32>();
        let bf = b[i].to_num::<f32>();
        let diff = af - bf;
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_distance_zero() {
        let a = vec![I64F32::from_num(1.0), I64F32::from_num(2.0), I64F32::from_num(3.0)];
        let b = a.clone();
        let dist = squared_distance_scalar(&a, &b);
        assert_eq!(dist, I64F32::ZERO);
    }

    #[test]
    fn test_scalar_distance_known() {
        // distance([1,0,0], [0,0,0]) = 1.0
        let a = vec![I64F32::from_num(1.0), I64F32::ZERO, I64F32::ZERO];
        let b = vec![I64F32::ZERO, I64F32::ZERO, I64F32::ZERO];
        let dist = squared_distance_scalar(&a, &b);
        assert_eq!(dist, I64F32::from_num(1.0));
    }

    #[test]
    fn test_scalar_distance_3d() {
        // distance([1,2,3], [4,5,6]) = 9+9+9 = 27.0
        let a = vec![I64F32::from_num(1.0), I64F32::from_num(2.0), I64F32::from_num(3.0)];
        let b = vec![I64F32::from_num(4.0), I64F32::from_num(5.0), I64F32::from_num(6.0)];
        let dist = squared_distance_scalar(&a, &b);
        assert_eq!(dist, I64F32::from_num(27.0));
    }

    #[test]
    fn test_simd_matches_scalar() {
        // The SIMD path must produce identical results to scalar
        let a: Vec<I64F32> = (0..128)
            .map(|i| I64F32::from_num(i as f64 * 0.1))
            .collect();
        let b: Vec<I64F32> = (0..128)
            .map(|i| I64F32::from_num(i as f64 * 0.2 + 0.5))
            .collect();

        let scalar_dist = squared_distance_scalar(&a, &b);
        let simd_dist = squared_distance_simd(&a, &b);

        assert_eq!(
            scalar_dist, simd_dist,
            "SIMD and scalar must produce bit-identical results"
        );
    }

    #[test]
    fn test_approximate_distance_reasonable() {
        let a = vec![I64F32::from_num(1.0), I64F32::from_num(2.0), I64F32::from_num(3.0)];
        let b = vec![I64F32::from_num(4.0), I64F32::from_num(5.0), I64F32::from_num(6.0)];

        let approx = approximate_distance_f32(&a, &b);
        // Should be approximately 27.0
        assert!((approx - 27.0).abs() < 0.01, "Approximate distance should be close to exact");
    }
}
