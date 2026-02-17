//! Integration tests for the scry-pipe code generator.
//!
//! These tests verify that the generated Rust code actually compiles with
//! `rustc` and that snapshot output remains stable.

use scry_pipe::codegen::RustCodegen;
use scry_pipe::engine::PipelineEngine;
use scry_pipe::ir::*;
use std::process::Command;
use tempfile::TempDir;

/// Build the example pipeline from SCRY_PIPE_PROPOSAL.md.
fn proposal_pipeline() -> PipelineDef {
    PipelineDef {
        name: "user_features".into(),
        version: "0.1.0".into(),
        created_at: "2026-02-14T07:00:00Z".into(),
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

/// Build a pipeline with all op types for comprehensive compilation testing.
fn all_ops_pipeline() -> PipelineDef {
    PipelineDef {
        name: "all_ops".into(),
        version: "0.1.0".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        steps: vec![
            PipelineStep {
                feature_idx: 0,
                op: TransformOp::StandardScale {
                    mean: 1.0,
                    std_dev: 2.0,
                },
            },
            PipelineStep {
                feature_idx: 1,
                op: TransformOp::MinMaxScale {
                    min: 0.0,
                    max: 10.0,
                },
            },
            PipelineStep {
                feature_idx: 2,
                op: TransformOp::RobustScale {
                    median: 5.0,
                    iqr: 3.0,
                },
            },
            PipelineStep {
                feature_idx: 0,
                op: TransformOp::Clip {
                    lower: -1.0,
                    upper: 1.0,
                },
            },
            PipelineStep {
                feature_idx: 1,
                op: TransformOp::Log1p,
            },
            PipelineStep {
                feature_idx: 3,
                op: TransformOp::Impute {
                    strategy: ImputeStrategy::Mean,
                    fill_value: 42.0,
                },
            },
            PipelineStep {
                feature_idx: 4,
                op: TransformOp::BinDiscretize {
                    bin_edges: vec![10.0, 20.0, 30.0],
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
            FeatureSpec {
                name: "c".into(),
                dtype: DType::Float64,
                index: 2,
            },
            FeatureSpec {
                name: "d".into(),
                dtype: DType::Float64,
                index: 3,
            },
            FeatureSpec {
                name: "e".into(),
                dtype: DType::Float64,
                index: 4,
            },
        ],
    }
}

/// Helper: compile generated Rust code with rustc and assert success.
fn assert_compiles(code: &str, label: &str) {
    let dir = TempDir::new().expect("create temp dir");
    let src = dir.path().join("generated.rs");
    let out = dir.path().join("libgenerated.rlib");
    std::fs::write(&src, code).unwrap();

    let output = Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--crate-name",
            "generated_pipeline",
        ])
        .arg("-o")
        .arg(&out)
        .arg(&src)
        .output()
        .expect("failed to invoke rustc");

    assert!(
        output.status.success(),
        "rustc failed for {label}:\nstderr: {}\nGenerated code:\n{code}",
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn generated_code_compiles_with_rustc() {
    let codegen = RustCodegen::new().no_std(false).emit_batch(false);
    let code = codegen.generate(&proposal_pipeline()).unwrap();
    assert_compiles(&code, "proposal (no_std=false, batch=false)");
}

#[test]
fn generated_code_with_batch_compiles() {
    let codegen = RustCodegen::new().no_std(false).emit_batch(true);
    let code = codegen.generate(&proposal_pipeline()).unwrap();
    assert_compiles(&code, "proposal (batch=true)");
}

#[test]
fn all_ops_generated_code_compiles() {
    let codegen = RustCodegen::new().no_std(false).emit_batch(false);
    let code = codegen.generate(&all_ops_pipeline()).unwrap();
    assert_compiles(&code, "all_ops");
}

#[test]
fn onehot_generated_code_compiles() {
    let def = PipelineDef {
        name: "ohe".into(),
        version: "0.1.0".into(),
        created_at: "2026-01-01".into(),
        steps: vec![PipelineStep {
            feature_idx: 0,
            op: TransformOp::OneHotEncode {
                categories: vec!["A".into(), "B".into(), "C".into()],
            },
        }],
        input_schema: vec![FeatureSpec {
            name: "cat".into(),
            dtype: DType::Float64,
            index: 0,
        }],
    };
    let codegen = RustCodegen::new().no_std(false).emit_batch(false);
    let code = codegen.generate(&def).unwrap();
    assert_compiles(&code, "onehot");
}

#[test]
fn polynomial_generated_code_compiles() {
    let def = PipelineDef {
        name: "poly".into(),
        version: "0.1.0".into(),
        created_at: "2026-01-01".into(),
        steps: vec![PipelineStep {
            feature_idx: 0,
            op: TransformOp::Polynomial { degree: 3 },
        }],
        input_schema: vec![FeatureSpec {
            name: "x".into(),
            dtype: DType::Float64,
            index: 0,
        }],
    };
    let codegen = RustCodegen::new().no_std(false).emit_batch(false);
    let code = codegen.generate(&def).unwrap();
    assert_compiles(&code, "polynomial");
}

#[test]
fn snapshot_proposal_pipeline() {
    let codegen = RustCodegen::new();
    let code = codegen.generate(&proposal_pipeline()).unwrap();
    insta::assert_snapshot!("proposal_pipeline", code);
}

#[test]
fn engine_codegen_parity_proposal() {
    let def = proposal_pipeline();
    let engine = PipelineEngine::new(def);

    let test_inputs: Vec<Vec<f64>> = vec![
        vec![40.0, 100_000.0, 1.0],
        vec![20.0, 20_000.0, 0.0],
        vec![80.0, 500_000.0, 2.0],
        vec![35.2, 260_000.0, 1.0],
    ];

    for input in &test_inputs {
        let engine_out = engine.transform_row(input).unwrap();

        // Compute expected codegen result manually:
        let scaled_age: f64 = (input[0] - 35.2) / 12.1;
        let scaled_income: f64 = (input[1] - 20_000.0) / 480_000.0;
        let city: f64 = input[2];
        let clipped_age = scaled_age.clamp(0.0, 120.0);
        let log_income = scaled_income.ln_1p();

        let eps = f64::EPSILON * 100.0;
        assert!(
            (engine_out[0] - clipped_age).abs() < eps,
            "age mismatch for input {input:?}: engine={}, expected={clipped_age}",
            engine_out[0]
        );
        assert!(
            (engine_out[1] - log_income).abs() < eps,
            "income mismatch for input {input:?}: engine={}, expected={log_income}",
            engine_out[1]
        );
        assert!(
            (engine_out[2] - city).abs() < eps,
            "city mismatch for input {input:?}: engine={}, expected={city}",
            engine_out[2]
        );
    }
}
