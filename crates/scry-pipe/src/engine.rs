// SPDX-License-Identifier: MIT OR Apache-2.0
//! Runtime transform engine for executing feature pipelines.
//!
//! [`PipelineEngine`] takes a [`PipelineDef`](crate::ir::PipelineDef) and
//! applies its transform steps to input rows, producing model-ready feature
//! vectors. Supports single-row, batch, and parallel (rayon) execution.

use rayon::prelude::*;

use crate::error::PipeError;
use crate::ir::{PipelineDef, TransformOp};

/// Executes a compiled pipeline definition against input data.
///
/// # Example
///
/// ```
/// use scry_pipe::ir::*;
/// use scry_pipe::engine::PipelineEngine;
///
/// let def = PipelineDef {
///     name: "demo".into(),
///     version: "0.1.0".into(),
///     created_at: "2026-01-01".into(),
///     steps: vec![PipelineStep {
///         feature_idx: 0,
///         op: TransformOp::Log1p,
///     }],
///     input_schema: vec![FeatureSpec {
///         name: "x".into(),
///         dtype: DType::Float64,
///         index: 0,
///     }],
/// };
///
/// let engine = PipelineEngine::new(def);
/// let out = engine.transform_row(&[0.0]).unwrap();
/// assert!((out[0] - 0.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct PipelineEngine {
    /// The pipeline definition containing all steps and fitted parameters.
    def: PipelineDef,
}

impl PipelineEngine {
    /// Create a new engine from a pipeline definition.
    pub fn new(def: PipelineDef) -> Self {
        Self { def }
    }

    /// Return a reference to the underlying pipeline definition.
    pub fn def(&self) -> &PipelineDef {
        &self.def
    }

    /// Transform a single input row, returning a new feature vector.
    ///
    /// The input slice length must match `input_schema.len()`, or a
    /// [`PipeError::Schema`] is returned.
    pub fn transform_row(&self, input: &[f64]) -> Result<Vec<f64>, PipeError> {
        let expected = self.def.input_schema.len();
        if input.len() != expected {
            return Err(PipeError::Schema(format!(
                "expected {} features, got {}",
                expected,
                input.len()
            )));
        }

        // Start with a copy of the input row. Expansion ops (OneHot,
        // Polynomial) append to this vector and shift subsequent indices.
        let mut row: Vec<f64> = input.to_vec();

        // Track cumulative index offsets caused by expansion ops so that
        // later steps referencing the *original* feature indices still
        // resolve correctly.
        let mut offset: Vec<isize> = vec![0; expected];

        for step in &self.def.steps {
            let base = step.feature_idx;
            let off = offset.get(base).copied().unwrap_or(0);
            // Safety: feature indices are small non-negative values; wrapping
            // would require > isize::MAX features which is physically impossible.
            #[allow(clippy::cast_possible_wrap)]
            let adjusted_idx = (base as isize + off) as usize;

            if adjusted_idx >= row.len() {
                return Err(PipeError::Transform {
                    feature_idx: base,
                    message: format!(
                        "adjusted index {} out of bounds (row len {})",
                        adjusted_idx,
                        row.len()
                    ),
                });
            }

            apply_op(&step.op, &mut row, adjusted_idx, &mut offset, base);
        }

        Ok(row)
    }

    /// Transform a batch of rows sequentially.
    pub fn transform_batch(
        &self,
        inputs: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, PipeError> {
        inputs.iter().map(|row| self.transform_row(row)).collect()
    }

    /// Transform a batch of rows in parallel using rayon.
    pub fn transform_batch_parallel(
        &self,
        inputs: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, PipeError> {
        inputs
            .par_iter()
            .map(|row| self.transform_row(row))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Transform dispatch
// ---------------------------------------------------------------------------

/// Apply a single [`TransformOp`] to the working `row` at `idx`.
///
/// For expansion ops (`OneHotEncode`, `Polynomial`), new columns are
/// inserted *after* `idx` and the `offset` table is updated so that
/// later steps targeting original feature indices still resolve correctly.
fn apply_op(
    op: &TransformOp,
    row: &mut Vec<f64>,
    idx: usize,
    offset: &mut [isize],
    original_feature_idx: usize,
) {
    match op {
        TransformOp::StandardScale { mean, std_dev } => {
            row[idx] = (row[idx] - mean) / std_dev;
        }
        TransformOp::MinMaxScale { min, max } => {
            row[idx] = (row[idx] - min) / (max - min);
        }
        TransformOp::RobustScale { median, iqr } => {
            row[idx] = (row[idx] - median) / iqr;
        }
        TransformOp::Clip { lower, upper } => {
            row[idx] = row[idx].clamp(*lower, *upper);
        }
        TransformOp::Log1p => {
            row[idx] = row[idx].ln_1p();
        }
        TransformOp::Impute { fill_value, .. } => {
            if row[idx].is_nan() {
                row[idx] = *fill_value;
            }
        }
        TransformOp::LabelEncode { .. } => {
            // Input is assumed to be a pre-encoded f64 index.
            // Passthrough — no transformation needed.
        }
        TransformOp::OneHotEncode { categories } => {
            let value = row[idx] as usize;
            let n = categories.len();
            // Replace the original column with N binary indicator columns.
            let mut indicators: Vec<f64> = vec![0.0; n];
            if value < n {
                indicators[value] = 1.0;
            }
            // Remove the original value and splice in the indicators.
            row.splice(idx..=idx, indicators);

            // Shift offsets for all features after this one.
            #[allow(clippy::cast_possible_wrap)]
            let expansion = n as isize - 1; // net new columns
            for off in offset.iter_mut().skip(original_feature_idx + 1) {
                *off += expansion;
            }
        }
        TransformOp::BinDiscretize { bin_edges } => {
            let x = row[idx];
            // Binary search: find the first edge > x → bin index.
            let bin = bin_edges.partition_point(|&edge| edge <= x);
            row[idx] = bin as f64;
        }
        TransformOp::Polynomial { degree } => {
            let x = row[idx];
            let d = *degree as usize;
            // Generate x^2, x^3, …, x^degree and insert after idx.
            let mut terms: Vec<f64> = Vec::with_capacity(d.saturating_sub(1));
            for p in 2..=d {
                // degree is a u8, so p fits comfortably in i32.
                #[allow(clippy::cast_possible_wrap)]
                let exp = p as i32;
                terms.push(x.powi(exp));
            }
            let extra = terms.len();
            // Insert the polynomial terms right after the original column.
            let insert_pos = idx + 1;
            for (i, t) in terms.into_iter().enumerate() {
                row.insert(insert_pos + i, t);
            }
            // Shift offsets for subsequent features.
            #[allow(clippy::cast_possible_wrap)]
            let expansion = extra as isize;
            for off in offset.iter_mut().skip(original_feature_idx + 1) {
                *off += expansion;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    /// Helper: pipeline with a single step on feature 0.
    fn single_step_pipeline(op: TransformOp) -> PipelineEngine {
        PipelineEngine::new(PipelineDef {
            name: "test".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![PipelineStep {
                feature_idx: 0,
                op,
            }],
            input_schema: vec![FeatureSpec {
                name: "x".into(),
                dtype: DType::Float64,
                index: 0,
            }],
        })
    }

    // -- Individual op correctness ----------------------------------------

    #[test]
    fn standard_scale() {
        let e = single_step_pipeline(TransformOp::StandardScale {
            mean: 35.2,
            std_dev: 12.1,
        });
        let out = e.transform_row(&[35.2]).unwrap();
        assert!((out[0] - 0.0).abs() < 1e-10, "mean maps to 0");
    }

    #[test]
    fn min_max_scale() {
        let e = single_step_pipeline(TransformOp::MinMaxScale {
            min: 20_000.0,
            max: 500_000.0,
        });
        let out = e.transform_row(&[260_000.0]).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-10, "midpoint maps to 0.5");
    }

    #[test]
    fn robust_scale() {
        let e = single_step_pipeline(TransformOp::RobustScale {
            median: 10.0,
            iqr: 4.0,
        });
        let out = e.transform_row(&[14.0]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-10, "(14 - 10) / 4 = 1.0");
    }

    #[test]
    fn clip() {
        let e = single_step_pipeline(TransformOp::Clip {
            lower: 0.0,
            upper: 120.0,
        });
        let out = e.transform_row(&[150.0]).unwrap();
        assert!((out[0] - 120.0).abs() < 1e-10, "clipped to upper");
    }

    #[test]
    fn log1p() {
        let e = single_step_pipeline(TransformOp::Log1p);
        let out = e.transform_row(&[0.0]).unwrap();
        assert!((out[0] - 0.0).abs() < 1e-10, "ln(1) = 0");
    }

    #[test]
    fn impute_nan() {
        let e = single_step_pipeline(TransformOp::Impute {
            strategy: ImputeStrategy::Mean,
            fill_value: 42.0,
        });
        let out = e.transform_row(&[f64::NAN]).unwrap();
        assert!((out[0] - 42.0).abs() < 1e-10, "NaN replaced by fill_value");
    }

    #[test]
    fn impute_non_nan_passthrough() {
        let e = single_step_pipeline(TransformOp::Impute {
            strategy: ImputeStrategy::Constant,
            fill_value: 42.0,
        });
        let out = e.transform_row(&[7.0]).unwrap();
        assert!((out[0] - 7.0).abs() < 1e-10, "non-NaN passes through");
    }

    #[test]
    fn label_encode_passthrough() {
        let e = single_step_pipeline(TransformOp::LabelEncode {
            classes: vec!["A".into(), "B".into(), "C".into()],
        });
        let out = e.transform_row(&[1.0]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-10, "pre-encoded index passes");
    }

    #[test]
    fn one_hot_encode() {
        let e = single_step_pipeline(TransformOp::OneHotEncode {
            categories: vec!["LA".into(), "NYC".into(), "SF".into()],
        });
        // Input index 1 → [0, 1, 0]
        let out = e.transform_row(&[1.0]).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn bin_discretize() {
        let e = single_step_pipeline(TransformOp::BinDiscretize {
            bin_edges: vec![10.0, 20.0, 30.0],
        });
        // x=15 → between edge[0]=10 and edge[1]=20 → bin 1
        let out = e.transform_row(&[15.0]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bin_discretize_boundaries() {
        let e = single_step_pipeline(TransformOp::BinDiscretize {
            bin_edges: vec![10.0, 20.0, 30.0],
        });
        // x < all edges → bin 0
        let out = e.transform_row(&[5.0]).unwrap();
        assert!((out[0] - 0.0).abs() < 1e-10);
        // x > all edges → bin 3
        let out = e.transform_row(&[35.0]).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn polynomial() {
        let e = single_step_pipeline(TransformOp::Polynomial { degree: 3 });
        let out = e.transform_row(&[2.0]).unwrap();
        // Original x=2, then x^2=4, x^3=8
        assert_eq!(out.len(), 3);
        assert!((out[0] - 2.0).abs() < 1e-10);
        assert!((out[1] - 4.0).abs() < 1e-10);
        assert!((out[2] - 8.0).abs() < 1e-10);
    }

    // -- Pipeline chains & validation -------------------------------------

    #[test]
    fn full_pipeline_chain() {
        // StandardScale → Clip → Log1p on the same feature.
        let engine = PipelineEngine::new(PipelineDef {
            name: "chain".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::StandardScale {
                        mean: 50.0,
                        std_dev: 10.0,
                    },
                },
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::Clip {
                        lower: 0.0,
                        upper: 5.0,
                    },
                },
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::Log1p,
                },
            ],
            input_schema: vec![FeatureSpec {
                name: "x".into(),
                dtype: DType::Float64,
                index: 0,
            }],
        });

        // x=70 → scale=(70-50)/10=2.0 → clip=2.0 → log1p=ln(3)
        let out = engine.transform_row(&[70.0]).unwrap();
        assert!((out[0] - 3.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn batch_matches_individual() {
        let engine = single_step_pipeline(TransformOp::StandardScale {
            mean: 0.0,
            std_dev: 1.0,
        });
        let inputs: Vec<Vec<f64>> = (0..1000).map(|i| vec![i as f64]).collect();
        let batch = engine.transform_batch(&inputs).unwrap();
        for (i, row) in batch.iter().enumerate() {
            let single = engine.transform_row(&inputs[i]).unwrap();
            assert_eq!(*row, single);
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let engine = single_step_pipeline(TransformOp::MinMaxScale {
            min: 0.0,
            max: 100.0,
        });
        let inputs: Vec<Vec<f64>> = (0..500).map(|i| vec![i as f64]).collect();
        let sequential = engine.transform_batch(&inputs).unwrap();
        let parallel = engine.transform_batch_parallel(&inputs).unwrap();
        assert_eq!(sequential, parallel);
    }

    #[test]
    fn input_length_mismatch_returns_schema_error() {
        let engine = single_step_pipeline(TransformOp::Log1p);
        let result = engine.transform_row(&[1.0, 2.0]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PipeError::Schema(_)),
            "expected Schema error, got: {err}"
        );
    }

    #[test]
    fn empty_pipeline_passthrough() {
        let engine = PipelineEngine::new(PipelineDef {
            name: "noop".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![],
            input_schema: vec![
                FeatureSpec {
                    name: "a".into(),
                    dtype: DType::Float64,
                    index: 0,
                },
                FeatureSpec {
                    name: "b".into(),
                    dtype: DType::Float64,
                    index: 1,
                },
            ],
        });
        let out = engine.transform_row(&[3.14, 2.72]).unwrap();
        assert_eq!(out, vec![3.14, 2.72]);
    }

    #[test]
    fn multi_step_same_feature() {
        // Two clips on the same feature in sequence.
        let engine = PipelineEngine::new(PipelineDef {
            name: "multi".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::Clip {
                        lower: 0.0,
                        upper: 100.0,
                    },
                },
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::Clip {
                        lower: 10.0,
                        upper: 50.0,
                    },
                },
            ],
            input_schema: vec![FeatureSpec {
                name: "x".into(),
                dtype: DType::Float64,
                index: 0,
            }],
        });
        // 200 → clip(0,100)=100 → clip(10,50)=50
        let out = engine.transform_row(&[200.0]).unwrap();
        assert!((out[0] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn onehot_then_transform_another_feature() {
        // Feature 0 = category index, Feature 1 = numeric.
        // OneHot feature 0 (3 cats) → then StandardScale feature 1.
        // After OneHot: row = [0, 1, 0, <original feat 1>]
        // StandardScale should target the original feature 1, which is now at index 3.
        let engine = PipelineEngine::new(PipelineDef {
            name: "mixed".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::OneHotEncode {
                        categories: vec!["A".into(), "B".into(), "C".into()],
                    },
                },
                PipelineStep {
                    feature_idx: 1,
                    op: TransformOp::StandardScale {
                        mean: 10.0,
                        std_dev: 2.0,
                    },
                },
            ],
            input_schema: vec![
                FeatureSpec {
                    name: "cat".into(),
                    dtype: DType::Float64,
                    index: 0,
                },
                FeatureSpec {
                    name: "num".into(),
                    dtype: DType::Float64,
                    index: 1,
                },
            ],
        });
        // cat=1 (B), num=14
        let out = engine.transform_row(&[1.0, 14.0]).unwrap();
        // OneHot(B) → [0, 1, 0], then StandardScale(14) = (14-10)/2 = 2.0
        assert_eq!(out.len(), 4); // 3 onehot + 1 numeric
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 0.0).abs() < 1e-10);
        assert!((out[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn empty_input() {
        let engine = PipelineEngine::new(PipelineDef {
            name: "empty".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![],
            input_schema: vec![],
        });
        let out = engine.transform_row(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn standard_scale_division_by_zero() {
        // std_dev=0 causes division by zero → produces inf, not a panic.
        let e = single_step_pipeline(TransformOp::StandardScale {
            mean: 0.0,
            std_dev: 0.0,
        });
        let out = e.transform_row(&[5.0]).unwrap();
        assert!(out[0].is_infinite() || out[0].is_nan());
    }

    #[test]
    fn min_max_scale_identical_bounds() {
        // min == max → division by zero → inf/nan, not a panic.
        let e = single_step_pipeline(TransformOp::MinMaxScale {
            min: 10.0,
            max: 10.0,
        });
        let out = e.transform_row(&[10.0]).unwrap();
        assert!(out[0].is_nan() || out[0].is_infinite());
    }

    #[test]
    fn robust_scale_zero_iqr() {
        let e = single_step_pipeline(TransformOp::RobustScale {
            median: 5.0,
            iqr: 0.0,
        });
        let out = e.transform_row(&[10.0]).unwrap();
        assert!(out[0].is_infinite());
    }

    #[test]
    fn log1p_negative_input() {
        // ln(1 + x) for x < -1 → NaN.
        let e = single_step_pipeline(TransformOp::Log1p);
        let out = e.transform_row(&[-2.0]).unwrap();
        assert!(out[0].is_nan());
    }

    #[test]
    fn bin_discretize_empty_edges() {
        // No edges → partition_point returns 0 for everything.
        let e = single_step_pipeline(TransformOp::BinDiscretize {
            bin_edges: vec![],
        });
        let out = e.transform_row(&[42.0]).unwrap();
        assert!((out[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn one_hot_out_of_range_index() {
        // Category index >= n → all indicators are 0.
        let e = single_step_pipeline(TransformOp::OneHotEncode {
            categories: vec!["A".into(), "B".into()],
        });
        let out = e.transform_row(&[5.0]).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn polynomial_degree_1_no_expansion() {
        let e = single_step_pipeline(TransformOp::Polynomial { degree: 1 });
        let out = e.transform_row(&[3.0]).unwrap();
        // degree=1: no extra terms generated (loop 2..=1 is empty).
        assert_eq!(out.len(), 1);
        assert!((out[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn def_accessor() {
        let e = single_step_pipeline(TransformOp::Log1p);
        assert_eq!(e.def().name, "test");
        assert_eq!(e.def().steps.len(), 1);
    }

    #[test]
    fn roundtrip_serialize_then_execute() {
        // Define pipeline → serialize → deserialize → execute → same result.
        let def = PipelineDef {
            name: "roundtrip".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::StandardScale { mean: 10.0, std_dev: 5.0 },
                },
                PipelineStep {
                    feature_idx: 1,
                    op: TransformOp::Clip { lower: 0.0, upper: 1.0 },
                },
            ],
            input_schema: vec![
                FeatureSpec { name: "a".into(), dtype: DType::Float64, index: 0 },
                FeatureSpec { name: "b".into(), dtype: DType::Float64, index: 1 },
            ],
        };

        let engine1 = PipelineEngine::new(def.clone());
        let out1 = engine1.transform_row(&[20.0, 0.5]).unwrap();

        let json = def.to_json().unwrap();
        let def2 = PipelineDef::from_json(&json).unwrap();
        let engine2 = PipelineEngine::new(def2);
        let out2 = engine2.transform_row(&[20.0, 0.5]).unwrap();

        assert_eq!(out1, out2);
    }

    #[test]
    fn polynomial_then_another_feature() {
        // Polynomial on feature 0 (degree 3), then scale feature 1.
        // After poly: [x, x^2, x^3, y] → scale targets adjusted index.
        let engine = PipelineEngine::new(PipelineDef {
            name: "poly_shift".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::Polynomial { degree: 3 },
                },
                PipelineStep {
                    feature_idx: 1,
                    op: TransformOp::StandardScale { mean: 0.0, std_dev: 2.0 },
                },
            ],
            input_schema: vec![
                FeatureSpec { name: "x".into(), dtype: DType::Float64, index: 0 },
                FeatureSpec { name: "y".into(), dtype: DType::Float64, index: 1 },
            ],
        });
        // x=2, y=6 → poly: [2, 4, 8, 6] → scale y: 6/2=3
        let out = engine.transform_row(&[2.0, 6.0]).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 2.0).abs() < 1e-10);
        assert!((out[1] - 4.0).abs() < 1e-10);
        assert!((out[2] - 8.0).abs() < 1e-10);
        assert!((out[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn batch_parallel_1000_rows() {
        let engine = PipelineEngine::new(PipelineDef {
            name: "big".into(),
            version: "0.1.0".into(),
            created_at: "2026-01-01".into(),
            steps: vec![
                PipelineStep {
                    feature_idx: 0,
                    op: TransformOp::StandardScale {
                        mean: 50.0,
                        std_dev: 10.0,
                    },
                },
                PipelineStep {
                    feature_idx: 1,
                    op: TransformOp::MinMaxScale {
                        min: 0.0,
                        max: 100.0,
                    },
                },
            ],
            input_schema: vec![
                FeatureSpec {
                    name: "a".into(),
                    dtype: DType::Float64,
                    index: 0,
                },
                FeatureSpec {
                    name: "b".into(),
                    dtype: DType::Float64,
                    index: 1,
                },
            ],
        });
        let inputs: Vec<Vec<f64>> = (0..1000)
            .map(|i| vec![i as f64, (i * 2) as f64])
            .collect();
        let seq = engine.transform_batch(&inputs).unwrap();
        let par = engine.transform_batch_parallel(&inputs).unwrap();
        assert_eq!(seq, par);
        assert_eq!(seq.len(), 1000);
    }
}
