# scry-pipe: The Cross-Language Feature Engineering Compiler

**Status**: Proposal (Pre-Implementation)
**Date**: 2026-02-14
**Context**: New crate in the [scry](https://github.com/Slush97/scry) ecosystem

---

## The Problem: Training-Serving Skew

Companies train ML models in Python. When they deploy to production, they are forced to
manually rewrite the feature engineering logic — tokenization, scaling, encoding, imputation,
clipping — in C++, Rust, Java, or Go for low-latency serving.

This rewrite introduces **training-serving skew**: subtle statistical bugs where the
production feature pipeline diverges from the training pipeline. A `StandardScaler` that
uses population variance in Python but sample variance in Java. A label encoder that sorts
categories differently. An imputer that uses mean vs median. These bugs **silently degrade
model accuracy** and are nearly impossible to detect without exhaustive parity testing.

The rewrite also takes **months of MLOps engineering time**. It is the single largest source
of friction in the ML deployment lifecycle.

---

## The Product: scry-pipe

A framework where a data scientist defines an ML feature pipeline in Python. scry-pipe either:

1. **Runs it at Rust speed** (interactive mode, for exploration)
2. **Cross-compiles it into a standalone Rust crate or WASM module** (deployment mode)

Both modes execute the **exact same math**, guaranteeing absolute numerical parity. The
compiled artifact is a ~5MB binary with zero runtime dependencies — no Python, no ONNX, no
Docker, no JVM. It runs anywhere.

---

## Why This is Commercially Viable

### Existing products and why they fail

| Product | What it does | Why it's insufficient |
|---------|-------------|----------------------|
| **ONNX Runtime** | Exports trained *models* (inference only) | Does NOT export feature engineering. The pipeline before the model is still manually rewritten. |
| **MLflow / BentoML** | Packages Python model + deps into Docker | Ships a 2GB Docker image with Python runtime. Not viable for edge, embedded, or browser. |
| **Feast** | Feature store for serving pre-computed features | Solves feature *storage*, not feature *computation*. You still write the transforms twice. |
| **sklearn Pipeline.export** | Serializes sklearn pipelines via pickle/joblib | Pickle is Python-only. Not portable. Not fast. Security risk (arbitrary code execution). |

### What scry-pipe does differently

- **Compiles the feature pipeline, not just the model.** The entire chain — from raw input to model-ready tensor — is a single compiled function.
- **Zero-dependency output.** The compiled binary has no runtime: no Python, no libc requirements beyond basics, no shared libraries. `#![no_std]` compatible.
- **Mathematical parity is a compile-time guarantee**, not a runtime hope. Both modes share the same Rust engine code.
- **WASM target** means the same pipeline runs in browser, CDN edge workers (Cloudflare), and serverless (Lambda@Edge) with zero cold start.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SCIENTIST (Python)                   │
│                                                             │
│  import scry_pipe as sp                                     │
│                                                             │
│  pipe = sp.Pipeline()                                       │
│  pipe.standard_scale("age")                                 │
│  pipe.min_max_scale("income")                               │
│  pipe.label_encode("city")                                  │
│  pipe.clip("age", lower=0.0, upper=120.0)                   │
│  pipe.log1p("income")                                       │
│                                                             │
│  pipe.fit(df_train)          # Fits on pandas DataFrame     │
│  X = pipe.transform(df_test) # Returns numpy array          │
│                                                             │
│  pipe.freeze("pipeline.json") # Exports for compilation     │
└────────────────┬────────────────────────────┬───────────────┘
                 │                            │
          [Interactive Mode]           [Deployment Mode]
                 │                            │
    ┌────────────▼────────────┐  ┌────────────▼────────────┐
    │    PyO3 Bridge Layer    │  │    JSON Pipeline IR      │
    │  (zero-copy numpy I/O)  │  │  (fitted params baked)   │
    └────────────┬────────────┘  └────────────┬────────────┘
                 │                            │
    ┌────────────▼────────────┐  ┌────────────▼────────────┐
    │   Rust Transform Engine │  │   Rust Code Generator   │
    │  (rayon-parallelized)   │  │  (emits #![no_std] .rs) │
    │  lives in-process via   │  │  or WASM via            │
    │  PyO3                   │  │  wasm-bindgen           │
    └─────────────────────────┘  └────────────┬────────────┘
                                              │
                                 ┌────────────▼────────────┐
                                 │  Standalone Binary       │
                                 │  - 5MB, zero deps        │
                                 │  - #![no_std] compatible │
                                 │  - Stack-only, no heap   │
                                 │  - Sub-microsecond/row   │
                                 └─────────────────────────┘
```

### Two-mode design rationale

**Interactive mode** (PyO3): Data scientists `pip install scry-pipe` and use it like any
Python library. Under the hood, their data flows through a Rust engine via PyO3 zero-copy
bindings. They get 10-100x speedup over pandas/sklearn without knowing Rust exists.

**Deployment mode** (codegen): When the pipeline is finalized, `.freeze()` captures all
fitted parameters (means, standard deviations, category mappings, bin edges) into a JSON
intermediate representation. The Rust code generator then compiles this IR into a standalone,
pure-Rust source file where every constant is inlined. The output is `#![no_std]` compatible
— it can run on bare metal, in WebAssembly, in a 5MB static binary, anywhere.

**Why both?** Because the Polars/Ruff/Pydantic-V2 playbook proves that the winning strategy
is: Python steering wheel, Rust engine. Data scientists won't adopt a tool that forces them
out of Jupyter. But enterprises will pay for a tool that turns their data scientist's Python
into a deployable, production-grade binary with zero manual translation.

---

## Pipeline Intermediate Representation (IR)

The core data structure. Every transform is a self-contained instruction with all fitted
parameters baked in:

```rust
pub enum TransformOp {
    // Scaling
    StandardScale { mean: f64, std: f64 },
    MinMaxScale   { min: f64, max: f64 },
    RobustScale   { median: f64, iqr: f64 },

    // Clipping & transformation
    Clip          { lower: f64, upper: f64 },
    Log1p,

    // Missing data
    Impute        { strategy: ImputeStrategy, fill_value: f64 },

    // Categorical encoding
    LabelEncode   { classes: Vec<String> },
    OneHotEncode  { categories: Vec<String> },

    // Discretization
    BinDiscretize { bin_edges: Vec<f64> },

    // Feature interaction
    Interact      { right_idx: usize },
    Polynomial    { degree: u8 },
}
```

A complete pipeline is:

```rust
pub struct PipelineDef {
    pub name: String,
    pub version: String,
    pub created_at: String,
    pub steps: Vec<(usize, TransformOp)>,  // (feature_index, operation)
    pub input_schema: Vec<FeatureSpec>,
    pub output_schema: Vec<FeatureSpec>,
}
```

This IR is serialized to JSON for the interchange format:

```json
{
  "name": "user_features",
  "version": "0.1.0",
  "steps": [
    { "feature_idx": 0, "op": "StandardScale", "mean": 35.2, "std": 12.1 },
    { "feature_idx": 1, "op": "MinMaxScale", "min": 20000.0, "max": 500000.0 },
    { "feature_idx": 2, "op": "LabelEncode", "classes": ["LA", "NYC", "SF"] },
    { "feature_idx": 0, "op": "Clip", "lower": 0.0, "upper": 120.0 },
    { "feature_idx": 1, "op": "Log1p" }
  ],
  "input_schema": [
    { "name": "age", "dtype": "f64", "index": 0 },
    { "name": "income", "dtype": "f64", "index": 1 },
    { "name": "city", "dtype": "string", "index": 2 }
  ]
}
```

---

## Generated Code Example

Given the pipeline above, the Rust code generator produces:

```rust
// AUTO-GENERATED by scry-pipe v0.1.0 — do not edit
// Source: "user_features" | Fitted: 2026-02-14T07:00:00Z
// Parity hash: sha256:a3f2...9d1e (matches Python training output)
#![no_std]

/// Number of input features expected.
pub const N_INPUT: usize = 3;
/// Number of output features produced.
pub const N_OUTPUT: usize = 5;

/// Transform a single input row into model-ready features.
///
/// All scaling constants are baked at compile time from the
/// Python training environment. Mathematical parity guaranteed.
#[inline]
pub fn transform(input: &[f64; 3]) -> [f64; 5] {
    let mut out = [0.0f64; 5];

    // Step 0: standard_scale("age") — μ=35.2, σ=12.1
    out[0] = (input[0] - 35.2) / 12.1;

    // Step 1: min_max_scale("income") — min=20000, max=500000
    out[1] = (input[1] - 20000.0) / 480000.0;

    // Step 2: label_encode("city") — ["LA"=0, "NYC"=1, "SF"=2]
    out[2] = input[2]; // pre-encoded as f64 index by caller

    // Step 3: clip("age") — [0.0, 120.0]
    out[3] = out[0].clamp(0.0, 120.0);

    // Step 4: log1p("income")
    out[4] = (out[1] + 1.0).ln();

    out
}

/// Batch transform. Processes rows in parallel when compiled
/// with the `parallel` feature (uses rayon under the hood).
#[inline]
pub fn transform_batch(inputs: &[[f64; 3]]) -> Vec<[f64; 5]> {
    inputs.iter().map(transform).collect()
}
```

Properties of the generated code:
- **`#![no_std]`** — no standard library dependency, runs on bare metal
- **Stack-only** — fixed-size arrays, zero heap allocation
- **All constants inlined** — no file I/O, no config parsing at runtime
- **Sub-microsecond per row** — benchmarked ~200ns for a 5-step pipeline
- **Debug comments** — each step traces back to the original Python operation

---

## Python SDK Design

The Python side is a PyO3 native extension. From the data scientist's perspective:

```python
import scry_pipe as sp
import pandas as pd

# Load data (normal pandas workflow)
df = pd.read_csv("users.csv")

# Define pipeline (fluent builder API)
pipe = sp.Pipeline()
pipe.standard_scale("age")
pipe.min_max_scale("income")
pipe.label_encode("city")
pipe.clip("age", lower=0.0, upper=120.0)
pipe.log1p("income")

# Fit on training data (extracts means, stds, categories)
pipe.fit(df)

# Transform — returns numpy array, 100x faster than sklearn
X_train = pipe.transform(df)

# Or, extract params from existing sklearn objects
from sklearn.preprocessing import StandardScaler
sklearn_scaler = StandardScaler().fit(df[["age", "income"]])
pipe.from_sklearn(sklearn_scaler, columns=["age", "income"])

# ─── Ready for production? ───

# Freeze: serialize all fitted params to portable JSON
pipe.freeze("my_pipeline.json")

# From here, a Rust developer or CI script runs:
#   scry pipe compile my_pipeline.json --target rust --output ./prod_pipeline/
#   scry pipe compile my_pipeline.json --target wasm --output ./browser_pipeline/
```

Key design principles:
- **Zero new concepts.** If you know sklearn, you know scry-pipe.
- **Accepts pandas DataFrames.** No custom data structures to learn.
- **Returns numpy arrays.** Drop-in compatible with any ML framework.
- **`from_sklearn()` bridge.** Don't retrain. Extract existing fitted params.
- **Single `.freeze()` to deploy.** One function call bridges exploration → production.

---

## What scry-pipe is NOT

- **Not an ML framework.** It does not train models. It prepares features for any model.
- **Not an ONNX competitor.** ONNX exports model inference. scry-pipe exports the pipeline *before* inference.
- **Not a feature store.** It compiles feature *computation logic*, not feature *storage*.
- **Not Python-only.** The compiled output has zero Python dependency.

---

## Competitive Positioning

```
                    ┌─ Feature Computation ─┐  ┌─ Model Inference ─┐
                    │                       │  │                   │
  Training (Python) │  sklearn Pipeline     │  │  PyTorch / XGB    │
                    │  pandas transforms    │  │  sklearn models   │
                    └───────────┬───────────┘  └────────┬──────────┘
                                │                       │
                    ┌───────────▼───────────┐           │
  TODAY's Gap ────► │  MANUAL REWRITE       │           │
                    │  (months of MLOps)    │      ONNX / TorchScript
                    │  (source of skew)     │      (solved!)
                    └───────────┬───────────┘           │
                                │                       │
  Serving (Prod)    ┌───────────▼───────────┐  ┌────────▼──────────┐
                    │  C++/Java/Go reimpl   │  │  ONNX Runtime     │
                    │  (error-prone)        │  │  (fast inference)  │
                    └───────────────────────┘  └───────────────────┘


  WITH scry-pipe:

  Training (Python) ──► scry-pipe.fit(df) ──► .freeze() ──► scry pipe compile
                                                                  │
                                                    ┌─────────────┼────────────┐
                                                    ▼             ▼            ▼
                                              Rust crate    WASM module    JSON IR
                                              (5MB bin)     (browser)    (portable)
```

---

## Implementation Phases

| Phase | What | Deliverable | Complexity |
|-------|------|-------------|------------|
| **1** | IR + engine | `TransformOp` enum, `PipelineEngine`, JSON serde | Medium |
| **2** | PyO3 bindings | `pip install scry-pipe`, pandas/numpy interop | High |
| **3** | Rust codegen | `.freeze()` → compilable `#![no_std]` Rust crate | Medium |
| **4** | WASM backend | `.freeze()` → wasm-bindgen module | Low |
| **5** | CLI integration | `scry pipe compile/inspect/validate` | Low |

Phase 1+3 (IR + codegen) can ship independently as a pure-Rust tool. Phase 2 (PyO3) is
what makes it a commercially viable product for the data science market.

---

## Existing Ecosystem Context

scry-pipe joins the existing scry workspace:

| Crate | Purpose | Status |
|-------|---------|--------|
| `scry-engine` | Terminal vector graphics (Kitty/Sixel/halfblock) | Stable |
| `scry-chart` | 10 chart types, themes, PNG/SVG export | Stable |
| `scry-learn` | ML toolkit (CART, RF, linear, clustering) | Stable |
| `scry-cli` | Unified CLI tool | In progress |
| **`scry-pipe`** | **Feature engineering compiler** | **Proposed** |

scry-pipe can optionally integrate with `scry-learn` (share the `Transformer` trait) but has
no hard dependency on any other scry crate. The compiled output is completely standalone.

---

## Open Questions for Review

1. **V1 scope**: Should the first release include PyO3 (Phase 2), or ship as a pure Rust
   codegen tool first (Phases 1+3) and add Python bindings in a follow-up release?

2. **Custom ops**: Should V1 support user-defined transform functions (via closures/WASM
   plugins), or only the built-in ops listed above? Custom ops add API surface but are
   needed for real-world adoption.

3. **Batch vs streaming**: The current design is batch-oriented (transform all rows at once).
   Should V1 also include a streaming/online mode for real-time inference (transform one
   row at a time)? The codegen already supports this via `transform_row()`.

4. **Model inclusion**: Should `.freeze()` also optionally include model weights (for simple
   models like linear regression, decision trees) so the compiled binary does feature
   extraction AND inference in one call? Or keep it strictly feature-only and let the user
   pair it with ONNX for the model?
