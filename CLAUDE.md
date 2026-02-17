# scry

Vector graphics engine for the terminal + charting library + ML toolkit.

## Workspace Layout

| Crate | Path | Purpose |
|-------|------|---------|
| `scry-engine` | `src/` | Core engine: scene builder, rasterizer (tiny-skia), transport (Kitty/Sixel/iTerm2/halfblock) |
| `scry-chart` | `crates/scry-chart/` | 19 chart types, 6 themes, PNG/SVG export, 3D interactive viz |
| `scry-learn` | `crates/scry-learn/` | 23+ ML models, preprocessing, metrics, GridSearchCV, GPU accel |
| `scry-cli` | `crates/scry-cli/` | CLI tool (`scry` binary) |
| `scry-pipe` | `crates/scry-pipe/` | Feature pipeline IR + codegen |
| `examples/` | `examples/` | 25+ demo programs for the core engine |
| `fuzz/` | `fuzz/` | libfuzzer targets (cart, scaler, neural, chart) |

## Commands

```bash
# Build & verify
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all -- --check

# Crate-specific testing
cargo test -p scry-learn --release                    # all scry-learn tests
cargo test -p scry-learn --release -- --nocapture     # with stdout output
cargo test --test benchmark_audit -p scry-learn --release -- --nocapture
cargo test --test numerical_stability -p scry-learn --release -- --nocapture
cargo test --test convergence -p scry-learn --release -- --nocapture
cargo test --test regression_audit -p scry-learn --release -- --nocapture

# Benchmarks (criterion)
cargo bench --bench ml_algorithms -p scry-learn
cargo bench --bench industry_benchmark -p scry-learn
cargo bench --bench competitor_bench -p scry-learn

# Fuzz & Miri (nightly required)
cargo +nightly fuzz run fuzz_cart_predict -- -max_total_time=30
cargo +nightly fuzz run fuzz_scaler_chain -- -max_total_time=30
cargo +nightly miri test -p scry-learn -- --skip gpu --skip viz

# Documentation
cargo doc -p scry-engine --all-features --open
```

## Stack

- **Rust** (MSRV 1.83.0)
- **tiny-skia** — 2D rasterization
- **fontdue** — text rendering (feature `text`)
- **ratatui** — widget integration (feature `widget`)
- **rayon** — parallel training (RF, GBT)
- **wgpu** — GPU compute shaders (feature `gpu`)
- **serde/clap** — CLI parsing

## Architecture Rules

### Code Conventions
- `#[non_exhaustive]` on all public types
- Builder pattern: `model.n_estimators(100).max_depth(8)`
- `unsafe_code = "deny"` in scry-learn — no unsafe allowed
- Column-major data: `Dataset.features[feature_idx][sample_idx]`
- Deterministic RNG: `fastrand::Rng::with_seed(42)` for all data generation in tests/benchmarks
- Feature flags: `kitty` (default), `sixel`, `iterm2`, `widget` (default), `text`, `shm`, `svg`, `gpu`, `csv`, `serde`

### Test & Benchmark Integrity
- **No marketing language in test/benchmark files.** Output only measured numbers. No feature comparison tables with checkmarks.
- **Always use proper train/test splits** for accuracy measurement. Never report accuracy on training data without explicit `"(train=test, timing only — NOT generalization)"` labeling.
- **Prediction checksums** (`prediction_checksum()` FNV-1a on f64 bits) must accompany all cross-library accuracy comparisons for reproducibility.
- **Cross-library comparisons must be like-for-like.** If comparing different algorithms (e.g., GBT vs RF), add a prominent `NOT a like-for-like comparison` warning.
- **Use `std::hint::black_box()`** for all timing measurements to prevent compiler elision.
- **Warmup iterations** (2+) before timing loops.

### scry-learn Specifics
- Models follow sklearn API pattern: `model.fit(&dataset)` → `model.predict(&rows)`
- `Dataset::new(features_col_major, target, feature_names, target_name)`
- Train/test: `train_test_split(&dataset, test_ratio, seed)`
- Metrics: `accuracy()`, `f1_score()`, `r2_score()`, `mean_squared_error()`
- All iterative models accept `.max_iter()` builder
- Test fixtures in `crates/scry-learn/tests/fixtures/` (iris, wine, breast_cancer, california, digits + 10 more UCI datasets as CSV pairs)

## Verification Checklist

Before any commit touching scry-learn:

```bash
cargo test -p scry-learn --release                 # all tests pass
cargo clippy -p scry-learn --all-targets           # no new warnings
cargo test --test benchmark_audit -p scry-learn --release -- --nocapture  # audit clean
```

Before any commit touching benchmarks:
- Verify no marketing tables (✅/❌) in output
- Verify all accuracy numbers use train/test split or are labeled as train=test
- Verify checksums are printed for cross-machine verification

## Concurrent Agent Safety

Multiple agents may work on this repo simultaneously. Conflict zones:

| Zone | Owner | Do NOT touch |
|------|-------|-------------|
| `src/sparse.rs` | Agent 13 (sparse polish) | Fixing from_triplets dedup bug |
| `src/linear/*.rs` | Agent 13 (sparse polish) | Adding auto-dispatch in fit() |
| `src/naive_bayes/gaussian.rs` | Agent 13 (sparse polish) | Adding auto-dispatch in fit() |
| `src/preprocess/scaler.rs` | Agent 13 (sparse polish) | Adding auto-dispatch in fit() |
| `src/neighbors/knn.rs` | Agent 13 (sparse polish) | True sparse distance + auto-dispatch |
| `benches/predict_latency.rs` | Agent 14 (predict bench) | New benchmark file |
| `crates/*/Cargo.toml`, `CHANGELOG.md` | Agent 15 (pre-publish) | Metadata + headers |
| `crates/scry-chart/src/streaming.rs` | Agent 16 (streaming charts) | New module |

Safe zones (no concurrent work):
- `crates/scry-learn/tests/` — test files
- `crates/scry-learn/src/tree/` — no current agents
- `crates/scry-learn/src/accel/` — no current agents
- `crates/scry-learn/src/neural/` — no current agents
- `crates/scry-pipe/src/` — no current agents

## Known Issues

- DenseMatrix migration done (Sprint 12B) — needs scaling benchmark validation (Sprint 12C, Agent 11).
- CART builder optimized (Sprint 12.5) — needs scaling benchmark validation.
- linfa-elasticnet Lasso/ElasticNet shows R²=0.249 vs scry's 0.999 — likely parameter interpretation mismatch, needs investigation.
