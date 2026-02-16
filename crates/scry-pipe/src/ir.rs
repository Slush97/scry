//! Intermediate representation for feature engineering pipelines.
//!
//! The IR captures every transform operation with all fitted parameters
//! baked in, enabling exact numerical parity between interactive (`PyO3`)
//! and compiled (codegen) execution modes.

use serde::{Deserialize, Serialize};

use crate::error::PipeError;

// ---------------------------------------------------------------------------
// Scalar enums
// ---------------------------------------------------------------------------

/// Data type of a feature column.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DType {
    /// 64-bit floating point.
    Float64,
    /// 64-bit signed integer.
    Int64,
    /// Variable-length string.
    String,
    /// Boolean.
    Bool,
}

/// Strategy for imputing missing values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImputeStrategy {
    /// Replace with column mean.
    Mean,
    /// Replace with column median.
    Median,
    /// Replace with most frequent value.
    MostFrequent,
    /// Replace with a caller-supplied constant.
    Constant,
}

// ---------------------------------------------------------------------------
// Transform operations
// ---------------------------------------------------------------------------

/// A single, self-contained transform operation with all fitted parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransformOp {
    /// Z-score normalization: `(x - mean) / std_dev`.
    StandardScale {
        /// Fitted mean.
        mean: f64,
        /// Fitted standard deviation.
        std_dev: f64,
    },
    /// Min-max scaling to [0, 1]: `(x - min) / (max - min)`.
    MinMaxScale {
        /// Fitted minimum.
        min: f64,
        /// Fitted maximum.
        max: f64,
    },
    /// Robust scaling using median and IQR: `(x - median) / iqr`.
    RobustScale {
        /// Fitted median.
        median: f64,
        /// Fitted interquartile range (Q3 - Q1).
        iqr: f64,
    },
    /// Clamp values to `[lower, upper]`.
    Clip {
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
    },
    /// Natural log of `(x + 1)`.
    Log1p,
    /// Replace missing (`NaN`) values.
    Impute {
        /// Strategy used to compute the fill value during fitting.
        strategy: ImputeStrategy,
        /// Pre-computed fill value.
        fill_value: f64,
    },
    /// Map string categories to integer indices (pre-encoded as `f64`).
    LabelEncode {
        /// Ordered list of class labels.
        classes: Vec<String>,
    },
    /// Expand a single column into N binary indicator columns.
    OneHotEncode {
        /// Ordered list of category names.
        categories: Vec<String>,
    },
    /// Discretize a continuous feature into bins via binary search.
    BinDiscretize {
        /// Sorted bin edges (N edges → N+1 bins, but we output the bin index).
        bin_edges: Vec<f64>,
    },
    /// Generate polynomial features up to `degree`.
    ///
    /// For an input value `x`, produces `[x^2, x^3, …, x^degree]`
    /// (the original `x^1` is kept in-place).
    Polynomial {
        /// Maximum polynomial degree.
        degree: u8,
    },
}

// ---------------------------------------------------------------------------
// Schema & pipeline types
// ---------------------------------------------------------------------------

/// Description of a single input feature column.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FeatureSpec {
    /// Human-readable column name.
    pub name: String,
    /// Data type.
    pub dtype: DType,
    /// Zero-based column index in the input row.
    pub index: usize,
}

/// A single step in the pipeline: apply `op` to the feature at `feature_idx`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineStep {
    /// Index of the feature this step operates on.
    pub feature_idx: usize,
    /// The transform operation to apply.
    pub op: TransformOp,
}

/// A complete, serializable pipeline definition.
///
/// Contains all fitted parameters so the pipeline can be executed or
/// compiled without access to the original training data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineDef {
    /// Human-readable pipeline name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// ISO-8601 timestamp of when the pipeline was fitted / frozen.
    pub created_at: String,
    /// Ordered list of transform steps.
    pub steps: Vec<PipelineStep>,
    /// Schema of the raw input features.
    pub input_schema: Vec<FeatureSpec>,
}

impl PipelineDef {
    /// Deserialize a `PipelineDef` from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, PipeError> {
        serde_json::from_str(json).map_err(PipeError::from)
    }

    /// Serialize this `PipelineDef` to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, PipeError> {
        serde_json::to_string_pretty(self).map_err(PipeError::from)
    }

    /// Compute the output dimensionality by walking every step.
    ///
    /// Most ops are 1→1 (they replace the value in-place). Expansions:
    /// - `OneHotEncode { categories }` → adds `categories.len()` columns
    ///   (the original column is consumed).
    /// - `Polynomial { degree }` → adds `degree - 1` extra columns.
    pub fn output_dim(&self) -> usize {
        let mut dim = self.input_schema.len();
        for step in &self.steps {
            match &step.op {
                TransformOp::OneHotEncode { categories } => {
                    // Original column is replaced by N indicator columns.
                    // Net change: +categories.len() - 1
                    dim += categories.len().saturating_sub(1);
                }
                TransformOp::Polynomial { degree } => {
                    // Original column stays; adds degree-1 new columns.
                    dim += (*degree as usize).saturating_sub(1);
                }
                _ => { /* 1-to-1 transform, no dimensionality change */ }
            }
        }
        dim
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal pipeline for testing.
    fn sample_pipeline() -> PipelineDef {
        PipelineDef {
            name: "test_pipe".to_string(),
            version: "0.1.0".to_string(),
            created_at: "2026-02-14T00:00:00Z".to_string(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::StandardScale {
                        mean: 35.2,
                        std_dev: 12.1,
                    },
                },
                PipelineStep {
                    feature_idx: 1,
                    op: TransformOp::MinMaxScale {
                        min: 20_000.0,
                        max: 500_000.0,
                    },
                },
                PipelineStep {
                    feature_idx: 2,
                    op: TransformOp::LabelEncode {
                        classes: vec!["LA".into(), "NYC".into(), "SF".into()],
                    },
                },
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::Clip {
                        lower: 0.0,
                        upper: 120.0,
                    },
                },
                PipelineStep {
                    feature_idx: 1,
                    op: TransformOp::Log1p,
                },
            ],
            input_schema: vec![
                FeatureSpec {
                    name: "age".into(),
                    dtype: DType::Float64,
                    index: 0,
                },
                FeatureSpec {
                    name: "income".into(),
                    dtype: DType::Float64,
                    index: 1,
                },
                FeatureSpec {
                    name: "city".into(),
                    dtype: DType::String,
                    index: 2,
                },
            ],
        }
    }

    #[test]
    fn roundtrip_json() {
        let original = sample_pipeline();
        let json = original.to_json().expect("serialize");
        let restored = PipelineDef::from_json(&json).expect("deserialize");
        assert_eq!(original, restored);
    }

    #[test]
    fn output_dim_no_expansion() {
        let pipe = sample_pipeline();
        // No OneHot or Polynomial → dim == input_schema.len()
        assert_eq!(pipe.output_dim(), 3);
    }

    #[test]
    fn output_dim_with_onehot() {
        let mut pipe = sample_pipeline();
        pipe.steps.push(PipelineStep {
            feature_idx: 2,
            op: TransformOp::OneHotEncode {
                categories: vec!["LA".into(), "NYC".into(), "SF".into()],
            },
        });
        // 3 base + (3 - 1) onehot expansion = 5
        assert_eq!(pipe.output_dim(), 5);
    }

    #[test]
    fn output_dim_with_polynomial() {
        let mut pipe = sample_pipeline();
        pipe.steps.push(PipelineStep {
            feature_idx: 0,
            op: TransformOp::Polynomial { degree: 3 },
        });
        // 3 base + (3 - 1) polynomial expansion = 5
        assert_eq!(pipe.output_dim(), 5);
    }

    #[test]
    fn output_dim_empty_steps() {
        let pipe = PipelineDef {
            name: "empty".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            steps: vec![],
            input_schema: vec![
                FeatureSpec {
                    name: "a".into(),
                    dtype: DType::Float64,
                    index: 0,
                },
                FeatureSpec {
                    name: "b".into(),
                    dtype: DType::Int64,
                    index: 1,
                },
            ],
        };
        assert_eq!(pipe.output_dim(), 2);
    }

    #[test]
    fn serialize_each_transform_variant() {
        let ops = vec![
            TransformOp::StandardScale {
                mean: 0.0,
                std_dev: 1.0,
            },
            TransformOp::MinMaxScale {
                min: 0.0,
                max: 1.0,
            },
            TransformOp::RobustScale {
                median: 5.0,
                iqr: 2.0,
            },
            TransformOp::Clip {
                lower: -1.0,
                upper: 1.0,
            },
            TransformOp::Log1p,
            TransformOp::Impute {
                strategy: ImputeStrategy::Mean,
                fill_value: 3.14,
            },
            TransformOp::LabelEncode {
                classes: vec!["a".into()],
            },
            TransformOp::OneHotEncode {
                categories: vec!["x".into(), "y".into()],
            },
            TransformOp::BinDiscretize {
                bin_edges: vec![0.0, 1.0, 2.0],
            },
            TransformOp::Polynomial { degree: 2 },
        ];
        for op in &ops {
            let json = serde_json::to_string(op).expect("serialize op");
            let restored: TransformOp = serde_json::from_str(&json).expect("deserialize op");
            assert_eq!(*op, restored);
        }
    }

    #[test]
    fn parse_design_doc_example_json() {
        // Matches the JSON example from SCRY_PIPE_PROPOSAL.md (adapted to
        // our struct layout where steps use PipelineStep with tagged enum).
        let json = r#"{
            "name": "user_features",
            "version": "0.1.0",
            "created_at": "2026-02-14T07:00:00Z",
            "steps": [
                { "feature_idx": 0, "op": { "StandardScale": { "mean": 35.2, "std_dev": 12.1 } } },
                { "feature_idx": 1, "op": { "MinMaxScale": { "min": 20000.0, "max": 500000.0 } } },
                { "feature_idx": 2, "op": { "LabelEncode": { "classes": ["LA", "NYC", "SF"] } } },
                { "feature_idx": 0, "op": { "Clip": { "lower": 0.0, "upper": 120.0 } } },
                { "feature_idx": 1, "op": "Log1p" }
            ],
            "input_schema": [
                { "name": "age", "dtype": "Float64", "index": 0 },
                { "name": "income", "dtype": "Float64", "index": 1 },
                { "name": "city", "dtype": "String", "index": 2 }
            ]
        }"#;
        let pipe = PipelineDef::from_json(json).expect("parse example JSON");
        assert_eq!(pipe.name, "user_features");
        assert_eq!(pipe.steps.len(), 5);
        assert_eq!(pipe.input_schema.len(), 3);
    }
}
