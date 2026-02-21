# Production Competitiveness Assessment Report
## Library: `scry-learn` + `scry-llm` (workspace `scry`)
## Date: February 21, 2026
## Assessor: Codex (code-and-evidence review)

## 1) Executive Summary

**Verdict:** **Not yet competitive as a full production Rust ML platform** against leading current Rust ML implementations, but **materially competitive in selected tabular/classical ML scenarios**.

- `scry-learn` shows unusually broad classical ML feature coverage and deep test/benchmark investment.
- `scry-llm` is a focused GPT-2 framework with CUDA/BF16 support, but lacks several production-critical capabilities expected today for LLM systems.
- The workspace currently looks strongest as an **advanced ML experimentation and tabular modeling toolkit**, not a complete production ML stack.

**Confidence level:** Medium-high (source audit + local compile/test evidence + ecosystem comparison to current Rust ML primary sources).

## 2) Scope and Methodology

### Scope
- In-scope code:
  - `crates/scry-learn`
  - `crates/scry-llm`
- Out-of-scope code:
  - visualization/rendering crates except where they affect ML production posture

### Evidence used
- Source and manifest audit of ML crates.
- Local build/tests:
  - `cargo check -p scry-learn` (pass)
  - `cargo check -p scry-llm --no-default-features` (pass)
  - `cargo check -p scry-llm` (pass)
  - `cargo test --test quick_vitals -p scry-learn --release -- --nocapture` (pass)
- External baseline comparison against current Rust ML ecosystem documentation (Burn, Candle, Linfa, SmartCore, ORT/tract ecosystems).

## 3) What Is Strong Today

### 3.1 Classical ML breadth is genuinely large
Evidence:
- Wide module surface in `scry-learn` including tree ensembles, linear models, SVMs, clustering, anomaly, neural nets, search/CV, preprocessing, sparse/text, explainability.
  - `crates/scry-learn/src/lib.rs:57`
  - `crates/scry-learn/src/lib.rs:153`
- Feature flags for CSV/serde/GPU/polars/mmap/viz.
  - `crates/scry-learn/Cargo.toml:15`

### 3.2 Pipeline + streaming support exist
Evidence:
- Composable pipeline abstraction with transformer chain + model.
  - `crates/scry-learn/src/pipeline.rs:21`
- Explicit online-learning trait and documented supported models.
  - `crates/scry-learn/src/partial_fit.rs:23`

### 3.3 Testing and benchmarking investment is substantial
Evidence:
- Large internal test surface in both source and integration test dirs.
- Dedicated competitor/industry/scaling benchmark suites.
  - `crates/scry-learn/Cargo.toml:65`
  - `crates/scry-learn/benches/README.md`
- Quick-vitals suite executed successfully in this review.

### 3.4 LLM stack has real CUDA core and BF16 path
Evidence:
- CUDA backend with cuBLAS/NVRTC integration and BF16 mode.
  - `crates/scry-llm/src/backend/cuda.rs:4`
  - `crates/scry-llm/src/backend/cuda.rs:60`
- GPT-2 model implementation with KV cache path.
  - `crates/scry-llm/src/nn/gpt2.rs:41`
  - `crates/scry-llm/src/nn/gpt2.rs:147`

## 4) Material Gaps vs Production Competitors

### 4.1 Model portability/export has a structural issue
- `onnx.rs` is present and extensive, but `lib.rs` does not expose `pub mod onnx;`, so users cannot access `scry_learn::onnx::ToOnnx` through the public API as written.
  - ONNX trait exists: `crates/scry-learn/src/onnx.rs:43`
  - Module not exported from crate root: `crates/scry-learn/src/lib.rs:57`

**Impact:** portability/interoperability appears weaker in practice than on paper.

### 4.2 Production serving/MLOps surface is limited
- No native model serving layer (HTTP/gRPC inference service patterns), model registry integration, or observability integrations are evident in the ML crates.
- Security policy scope is terminal/rendering-centric rather than ML-specific.
  - `SECURITY.md:19`

**Impact:** significant custom platform work required for real production operations.

### 4.3 LLM production readiness lags current expectations
- LLM crate is explicitly GPT-2 centered and still at `0.1.0`.
  - `crates/scry-llm/Cargo.toml:3`
  - `crates/scry-llm/src/nn/gpt2.rs:14`
- Default feature is CUDA-only (`default = ["cuda"]`), which reduces portability/ergonomics for many deployment paths.
  - `crates/scry-llm/Cargo.toml:11`
- No explicit distributed/multi-node/multi-GPU orchestration abstractions are present in training config or trainer API.
  - `crates/scry-llm/src/training.rs:16`

**Impact:** not yet competitive with the strongest Rust LLM stacks for production-scale training/inference.

### 4.4 API/runtime hardening gaps
- Some critical paths still panic instead of returning recoverable errors (example: CUDA initialization and context access).
  - `crates/scry-llm/src/backend/cuda.rs:45`
  - `crates/scry-llm/src/backend/cuda.rs:73`
- Dataset construction relies on assertions for shape checks.
  - `crates/scry-learn/src/dataset.rs:69`

**Impact:** this raises operational risk in long-running services.

## 5) Competitive Positioning vs Current Rust ML Ecosystem (2026)

### 5.1 Relative to tabular/classical Rust ML (`smartcore`, `linfa`)
- `scry-learn` is competitive on breadth and appears to invest heavily in internal benchmarking.
- It can be competitive in teams that want an all-Rust tabular toolkit with integrated visualization.
- Production edge still depends on hardening around interoperability, API guarantees, and service integration.

### 5.2 Relative to deep learning/LLM frameworks (`burn`, `candle`)
- Current state is behind leading Rust DL frameworks on ecosystem maturity and production deployment pathways.
- `scry-llm` is promising technically, but still a focused training framework rather than a production-grade full platform.

### 5.3 Relative to production inference stacks (`ort`, `tract` ecosystems)
- These ecosystems emphasize deployment/interoperability/runtime execution concerns that are not yet first-class here.

## 6) Scored Assessment (0-5)

| Dimension | Score | Notes |
|---|---:|---|
| Classical algorithm breadth | 4.5 | Very broad tabular coverage in source exports |
| LLM/deep learning breadth | 2.5 | Strong GPT-2 core, narrow overall model family |
| Interoperability (ONNX/portable serving) | 2.0 | ONNX module exposure issue; serving layer absent |
| Performance evidence quality | 3.0 | Many benchmarks; mostly internal/self-run |
| Reliability/testing rigor | 4.0 | Large test/fuzz surface and benchmark discipline |
| Production ops (serving/monitoring/governance) | 1.5 | Major gaps for enterprise operation |
| Deployment flexibility | 2.5 | CUDA path strong; broader deployment story limited |
| API/platform maturity | 2.5 | Rich features but still 0.x-style maturity profile |

**Overall competitiveness score:** **2.8 / 5.0**

Interpretation:
- **Competitive in parts** (classical ML workloads).
- **Not yet competitive as a production-complete Rust ML platform.**

## 7) Go/No-Go Decision

### For production use today
- **Go (conditional)** for controlled tabular ML workloads where your team can own serving and MLOps infrastructure.
- **No-go** if you require out-of-the-box production-grade interoperability, standardized serving, and hardened LLM production workflows.

## 8) 90-Day Plan to Become Competitive

### Phase 1 (0-30 days) — unblock critical gaps
1. Export ONNX module from public API (`pub mod onnx;`) and add end-to-end ONNX runtime parity tests.
2. Replace panic-prone infra paths in ML-critical code with typed recoverable errors.
3. Publish reproducible benchmark protocol + CI-generated benchmark artifacts.

### Phase 2 (31-60 days) — production foundation
1. Add minimal inference-serving crate (HTTP/gRPC) with health checks and structured metrics.
2. Add model artifact versioning conventions and migration docs.
3. Add inference-time observability hooks (latency/error/drift-friendly metrics).

### Phase 3 (61-90 days) — ecosystem competitiveness
1. Add first-class interop runners (ONNX Runtime/tract where applicable).
2. Expand `scry-llm` beyond GPT-2 baseline and formalize deployment targets.
3. Add multi-device/distributed training roadmap with explicit milestones.

## 9) Final Determination

If the question is "does it already have enough features to be broadly competitive with production Rust ML implementations right now?" the answer is:

**Not yet.**

If the question is "is the core substantial enough to become competitive soon with focused productization?" the answer is:

**Yes, especially for tabular/classical ML first.**

## 10) External Sources (Current Ecosystem Baselines)

- Burn docs/crate (current published API surface): https://docs.rs/crate/burn/latest
- Burn official site/book: https://burn.dev/
- Candle repository (Rust ML framework by Hugging Face): https://github.com/huggingface/candle
- Linfa crate docs: https://docs.rs/linfa/latest/linfa/
- SmartCore crate docs: https://docs.rs/smartcore/latest/smartcore/
- ORT Rust bindings docs: https://docs.rs/ort/latest/ort/
- tract ONNX crate docs: https://docs.rs/tract-onnx/latest/tract_onnx/

