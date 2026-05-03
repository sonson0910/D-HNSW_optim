//! Error types for D-HNSW.
//!
//! All error variants used across the crate are consolidated here to provide
//! a single, consistent error type for the public API.

use std::fmt;

/// Result type alias using [`HnswError`].
pub type Result<T> = std::result::Result<T, HnswError>;

/// Errors produced by D-HNSW operations.
#[derive(Debug)]
pub enum HnswError {
    /// Graph has reached its maximum node capacity.
    CapacityExceeded,

    /// The specified node ID does not exist in the graph.
    InvalidNodeId(usize),

    /// A construction or query parameter is invalid.
    InvalidParameter(String),

    /// The graph is empty (no entry point available for search).
    EmptyGraph,

    /// Vector dimension mismatch (expected vs. actual).
    DimensionMismatch { expected: usize, actual: usize },

    /// Fixed-point arithmetic overflow or underflow.
    ArithmeticOverflow(String),

    /// Deserialization failed (corrupt data, size limit, or invalid structure).
    DeserializationError(String),

    /// Serialization failed.
    SerializationError(String),

    /// Input data is invalid (e.g., NaN or Infinity in float conversion).
    InvalidData,
}

impl fmt::Display for HnswError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded => write!(f, "Graph capacity exceeded (max {} nodes)", 5_000_000),
            Self::InvalidNodeId(id) => write!(f, "Invalid node ID: {}", id),
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::EmptyGraph => write!(f, "Graph is empty (no entry point)"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::ArithmeticOverflow(msg) => write!(f, "Arithmetic overflow: {}", msg),
            Self::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            Self::InvalidData => write!(f, "Input data is invalid (NaN or Infinity)"),
        }
    }
}

impl std::error::Error for HnswError {}

/// Conversion from bincode errors for serialization/deserialization.
impl From<Box<bincode::ErrorKind>> for HnswError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        HnswError::SerializationError(e.to_string())
    }
}
