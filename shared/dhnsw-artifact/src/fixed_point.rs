// ============================================================
//  Q32.32 Fixed-Point Arithmetic Module
//  Central to D-HNSW's bit-exact determinism guarantee.
//
//  All distance computations are performed in I64F32 format:
//    - 32-bit integer part
//    - 32-bit fractional part
//    - Total: 64 bits per coordinate component
//
//  This eliminates IEEE-754 rounding nondeterminism across
//  different CPU architectures (x86-64, ARM64, RISC-V).
// ============================================================

use fixed::FixedI64;
use fixed::types::extra::U32;
use std::fmt;

/// The concrete Q32.32 type: 32-bit integer + 32-bit fraction.
type I64F32 = FixedI64<U32>;

/// A fixed-point scalar using Q32.32 representation.
/// Wraps `FixedI64<U32>` from the `fixed` crate.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedScalar(pub I64F32);

impl FixedScalar {
    /// Create from a floating-point value (quantization step).
    /// This is the ONLY point where floating-point enters the system.
    /// After construction, all operations are pure integer arithmetic.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        FixedScalar(I64F32::from_num(val))
    }

    /// Create from a floating-point value (f64 precision for import).
    #[inline]
    pub fn from_f64(val: f64) -> Self {
        FixedScalar(I64F32::from_num(val))
    }

    /// Create from raw integer bits.
    #[inline]
    pub fn from_bits(bits: i64) -> Self {
        FixedScalar(I64F32::from_bits(bits))
    }

    /// Get the raw integer bits.
    #[inline]
    pub fn to_bits(self) -> i64 {
        self.0.to_bits()
    }

    /// Saturating addition — prevents overflow panics.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        FixedScalar(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        FixedScalar(self.0.saturating_sub(rhs.0))
    }

    /// Saturating multiplication.
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        FixedScalar(self.0.saturating_mul(rhs.0))
    }

    /// Convert back to f64 (for reporting / logging only).
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0.to_num::<f64>()
    }

    /// Zero constant.
    pub const ZERO: FixedScalar = FixedScalar(I64F32::ZERO);
}

impl fmt::Debug for FixedScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q32.32({:.6})", self.to_f64())
    }
}

impl fmt::Display for FixedScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
}

// ---- Vector type ----

/// A fixed-point vector. All components are Q32.32.
#[derive(Clone, Debug)]
pub struct FixedVector {
    pub components: Vec<FixedScalar>,
}

impl FixedVector {
    /// Quantize an f32 slice into a fixed-point vector.
    /// This is the ingestion boundary — after this call,
    /// no floating-point arithmetic is ever used.
    pub fn from_f32_slice(data: &[f32]) -> Self {
        FixedVector {
            components: data.iter().map(|&v| FixedScalar::from_f32(v)).collect(),
        }
    }

    /// Dimension count.
    #[inline]
    pub fn dim(&self) -> usize {
        self.components.len()
    }

    /// Deterministic Squared Euclidean distance.
    /// Uses only saturating integer arithmetic.
    ///
    /// d(u, v) = Σ (u_i − v_i)²
    ///
    /// Theorem 9 (Proposition) in the paper:
    ///   For any two Q32.32 vectors u, v of dimension d,
    ///   dist_fixed(u, v) is computed via integer subtract
    ///   and multiply, producing an identical bit pattern
    ///   on every ISA that implements two's-complement i64.
    pub fn squared_euclidean(&self, other: &FixedVector) -> FixedScalar {
        debug_assert_eq!(self.dim(), other.dim(), "Dimension mismatch");

        let mut sum = FixedScalar::ZERO;
        for (a, b) in self.components.iter().zip(other.components.iter()) {
            let diff = a.saturating_sub(*b);
            let sq = diff.saturating_mul(diff);
            sum = sum.saturating_add(sq);
        }
        sum
    }

    /// Deterministic inner product distance.
    /// d(u, v) = 1 − Σ u_i · v_i  (for normalized vectors)
    pub fn inner_product_distance(&self, other: &FixedVector) -> FixedScalar {
        debug_assert_eq!(self.dim(), other.dim(), "Dimension mismatch");

        let one = FixedScalar::from_f32(1.0);
        let mut dot = FixedScalar::ZERO;
        for (a, b) in self.components.iter().zip(other.components.iter()) {
            let prod = a.saturating_mul(*b);
            dot = dot.saturating_add(prod);
        }
        one.saturating_sub(dot)
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_roundtrip() {
        let val = 3.14159f32;
        let fixed = FixedScalar::from_f32(val);
        let back = fixed.to_f64();
        // Q32.32 has ~2^-32 precision ≈ 2.3e-10
        assert!((back - val as f64).abs() < 1e-6,
                "Roundtrip error too large: {} vs {}", back, val);
    }

    #[test]
    fn test_squared_euclidean_zero() {
        let v = FixedVector::from_f32_slice(&[1.0, 2.0, 3.0]);
        let dist = v.squared_euclidean(&v);
        assert_eq!(dist.to_bits(), 0, "Distance to self must be exactly zero");
    }

    #[test]
    fn test_squared_euclidean_known() {
        let a = FixedVector::from_f32_slice(&[1.0, 0.0]);
        let b = FixedVector::from_f32_slice(&[0.0, 1.0]);
        let dist = a.squared_euclidean(&b);
        // Expected: (1-0)^2 + (0-1)^2 = 2.0
        let expected = FixedScalar::from_f32(2.0);
        assert_eq!(dist.to_bits(), expected.to_bits(),
                   "Known distance mismatch: {} vs {}", dist, expected);
    }

    #[test]
    fn test_determinism_bit_exact() {
        // Run the same computation 1000 times — must produce
        // identical bits every single time.
        let a = FixedVector::from_f32_slice(&[0.123, -0.456, 0.789, -1.234]);
        let b = FixedVector::from_f32_slice(&[-0.987, 0.654, -0.321, 0.111]);

        let reference = a.squared_euclidean(&b).to_bits();
        for _ in 0..1000 {
            let result = a.squared_euclidean(&b).to_bits();
            assert_eq!(result, reference, "Bit-exact determinism violated!");
        }
    }

    #[test]
    fn test_saturating_overflow() {
        let max = FixedScalar::from_bits(i64::MAX);
        let one = FixedScalar::from_f32(1.0);
        let result = max.saturating_add(one);
        assert_eq!(result.to_bits(), i64::MAX,
                   "Saturating add must clamp at MAX");
    }
}
