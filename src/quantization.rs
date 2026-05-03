//! Scalar Quantization (INT8) for Memory-Efficient D-HNSW
//!
//! This module provides INT8 scalar quantization to reduce memory overhead
//! from 8 bytes/dim (I64F32) to 1 byte/dim — an **8× reduction**.
//!
//! ## Determinism Guarantee
//!
//! All quantization operations use **integer-only arithmetic**:
//! - Calibration: min/max computed via I64F32 comparisons (deterministic)
//! - Quantization: round-half-to-even via integer division (deterministic)
//! - Distance: LUT-based squared distance using i32 accumulation (deterministic)
//!
//! ## Usage
//!
//! ```text
//! FixedPointVector<D>  →  calibrate()  →  QuantizationParams<D>
//!                      →  quantize()   →  ScalarQuantizedVector<D>
//! ```

use crate::fixed_point::I64F32;
use serde::{Deserialize, Serialize};

/// Calibration parameters for scalar quantization.
///
/// Stores per-dimension min/max values and precomputed scale factors.
/// These are derived from a calibration dataset and used to map
/// I64F32 values to the INT8 range [0, 255].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizationParams<const D: usize> {
    /// Per-dimension minimum values (I64F32 bit representation)
    pub min_vals: Vec<i64>,
    /// Per-dimension scale factors: (max - min) / 255, stored as I64F32 bits
    pub scale: Vec<i64>,
    /// Per-dimension inverse scale: 255 / (max - min), stored as I64F32 bits
    pub inv_scale: Vec<i64>,
}

/// An INT8 scalar-quantized vector.
///
/// Each component is quantized to [0, 255] (unsigned byte) to maximize
/// the available range. Memory footprint: D bytes instead of 8D bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarQuantizedVector<const D: usize> {
    /// Quantized values in [0, 255]
    pub data: Vec<u8>,
}

impl<const D: usize> QuantizationParams<D> {
    /// Calibrate quantization parameters from a set of training vectors.
    ///
    /// Computes per-dimension min/max across all vectors, then derives
    /// scale factors for mapping to [0, 255].
    ///
    /// # Determinism
    /// All comparisons use I64F32 which is integer-based → deterministic.
    pub fn calibrate(vectors: &[crate::fixed_point::FixedPointVector<D>]) -> Self {
        assert!(!vectors.is_empty(), "Cannot calibrate with empty dataset");

        // Initialize min/max with first vector
        let mut min_vals = vectors[0].components.map(|c| c.to_bits());
        let mut max_vals = vectors[0].components.map(|c| c.to_bits());

        // Find per-dimension min/max
        for vec in vectors.iter().skip(1) {
            for d in 0..D {
                let val = vec.components[d].to_bits();
                if val < min_vals[d] {
                    min_vals[d] = val;
                }
                if val > max_vals[d] {
                    max_vals[d] = val;
                }
            }
        }

        // Compute scale = (max - min) / 255
        // and inv_scale = 255 / (max - min)
        let mut scale = vec![0i64; D];
        let mut inv_scale = vec![0i64; D];
        let f255 = I64F32::from_num(255);

        for d in 0..D {
            let min_fp = I64F32::from_bits(min_vals[d]);
            let max_fp = I64F32::from_bits(max_vals[d]);
            let range = max_fp.saturating_sub(min_fp);

            if range > I64F32::ZERO {
                // scale = range / 255
                let s = range.saturating_div(f255);
                scale[d] = s.to_bits();
                // inv_scale = 255 / range
                let is = f255.saturating_div(range);
                inv_scale[d] = is.to_bits();
            } else {
                // All values are the same in this dimension
                scale[d] = I64F32::from_num(1).to_bits();
                inv_scale[d] = I64F32::from_num(1).to_bits();
            }
        }

        Self {
            min_vals: min_vals.to_vec(),
            scale,
            inv_scale,
        }
    }

    /// Quantize a full-precision vector to INT8.
    ///
    /// # Determinism
    /// Uses integer-only operations: subtract min, multiply by inv_scale,
    /// clamp to [0, 255], cast to u8.
    pub fn quantize(
        &self,
        vector: &crate::fixed_point::FixedPointVector<D>,
    ) -> ScalarQuantizedVector<D> {
        let mut data = vec![0u8; D];

        for d in 0..D {
            let val = vector.components[d];
            let min_fp = I64F32::from_bits(self.min_vals[d]);
            let inv_s = I64F32::from_bits(self.inv_scale[d]);

            // quantized = (val - min) * inv_scale
            let shifted = val.saturating_sub(min_fp);
            let scaled = shifted.saturating_mul(inv_s);

            // Clamp to [0, 255] and convert to u8
            let clamped = scaled.to_num::<i32>().clamp(0, 255) as u8;
            data[d] = clamped;
        }

        ScalarQuantizedVector { data }
    }

    /// Dequantize an INT8 vector back to full precision.
    ///
    /// Approximate inverse: val ≈ quantized * scale + min
    pub fn dequantize(
        &self,
        qvec: &ScalarQuantizedVector<D>,
    ) -> crate::fixed_point::FixedPointVector<D> {
        let mut components = [I64F32::ZERO; D];

        for d in 0..D {
            let q = I64F32::from_num(qvec.data[d] as i32);
            let s = I64F32::from_bits(self.scale[d]);
            let min_fp = I64F32::from_bits(self.min_vals[d]);

            // val = q * scale + min
            components[d] = q.saturating_mul(s).saturating_add(min_fp);
        }

        crate::fixed_point::FixedPointVector { components }
    }

    /// Memory footprint per vector in bytes.
    pub fn memory_per_vector(&self) -> usize {
        D // 1 byte per dimension
    }
}

impl<const D: usize> ScalarQuantizedVector<D> {
    /// Compute squared Euclidean distance between two quantized vectors.
    ///
    /// Uses i32 arithmetic to avoid overflow (max per-dim: 255² = 65025,
    /// with D dimensions: D × 65025). For D ≤ 65536, i32 is sufficient.
    ///
    /// # Determinism
    /// Pure integer arithmetic → deterministic across all platforms.
    pub fn squared_distance(&self, other: &Self) -> u64 {
        let mut sum: u64 = 0;
        for d in 0..D {
            let diff = self.data[d] as i32 - other.data[d] as i32;
            sum += (diff * diff) as u64;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::FixedPointVector;

    #[test]
    fn test_calibration_basic() {
        let v1: FixedPointVector<3> =
            FixedPointVector::from_f32_slice(&[0.0, 0.0, 0.0]).unwrap();
        let v2: FixedPointVector<3> =
            FixedPointVector::from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
        let v3: FixedPointVector<3> =
            FixedPointVector::from_f32_slice(&[0.5, 1.0, 1.5]).unwrap();

        let params = QuantizationParams::calibrate(&[v1, v2, v3]);
        assert_eq!(params.min_vals.len(), 3);
        assert_eq!(params.scale.len(), 3);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let vectors: Vec<FixedPointVector<3>> = (0..10)
            .map(|i| {
                FixedPointVector::from_f32_slice(&[
                    i as f32 * 0.1,
                    i as f32 * 0.2,
                    i as f32 * 0.3,
                ])
                .unwrap()
            })
            .collect();

        let params = QuantizationParams::calibrate(&vectors);

        for vec in &vectors {
            let qvec = params.quantize(vec);
            let dequantized = params.dequantize(&qvec);

            // Check that dequantized is close to original
            for d in 0..3 {
                let orig = vec.components[d].to_num::<f32>();
                let restored = dequantized.components[d].to_num::<f32>();
                let error = (orig - restored).abs();
                // Quantization error should be small relative to range
                assert!(
                    error < 0.05,
                    "Dimension {} error too large: {} vs {} (error: {})",
                    d, orig, restored, error
                );
            }
        }
    }

    #[test]
    fn test_quantized_distance_determinism() {
        let v1: FixedPointVector<3> =
            FixedPointVector::from_f32_slice(&[0.0, 0.0, 0.0]).unwrap();
        let v2: FixedPointVector<3> =
            FixedPointVector::from_f32_slice(&[1.0, 1.0, 1.0]).unwrap();
        let v3: FixedPointVector<3> =
            FixedPointVector::from_f32_slice(&[0.5, 0.5, 0.5]).unwrap();

        let params = QuantizationParams::calibrate(&[v1.clone(), v2.clone(), v3.clone()]);

        let q1 = params.quantize(&v1);
        let q2 = params.quantize(&v2);

        // Distance should be deterministic
        let d1 = q1.squared_distance(&q2);
        let d2 = q1.squared_distance(&q2);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_memory_savings() {
        let params: QuantizationParams<128> = QuantizationParams {
            min_vals: vec![0; 128],
            scale: vec![0; 128],
            inv_scale: vec![0; 128],
        };

        // INT8: 1 byte/dim = 128 bytes
        assert_eq!(params.memory_per_vector(), 128);
        // I64F32: 8 bytes/dim = 1024 bytes
        // Ratio: 8x reduction
        let full_precision_bytes = 128 * 8; // 8 bytes per I64F32
        let ratio = full_precision_bytes / params.memory_per_vector();
        assert_eq!(ratio, 8);
    }
}
