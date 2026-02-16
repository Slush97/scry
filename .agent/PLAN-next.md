# Sprint 11 Completion + 8C Large File Refactoring

> **For next agent** | 2026-02-16 | Depends on: nothing (independent of benchmark agent on 8A)

## Situation

- **Sprint 11 (Neural Networks)** is 80% done: 2175 LOC, 34 unit tests, `Tunable` + `Visualize` impls exist
- The entire neural module (`crates/scry-learn/src/neural/`) and `src/rng.rs` are **untracked** (not committed)
- Many other scry-learn changes are staged but uncommitted (see `git status`)
- **8C Large File Refactoring** hasn't started — 5 files past maintainability thresholds

**Another agent is concurrently working on 8A (industry benchmarks).** Do not touch benchmark files.

---

## Phase 1: Sprint 11 Finishing Touches

### 1A. Add `PipelineModel` impls for MLP

**File: `crates/scry-learn/src/pipeline.rs`**

Every other model has `impl PipelineModel`. MLP is missing. Add:

```rust
impl PipelineModel for crate::neural::MLPClassifier {
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}

impl PipelineModel for crate::neural::MLPRegressor {
    fn fit(&mut self, data: &Dataset) -> Result<()> { self.fit(data) }
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> { self.predict(features) }
}
```

Check: `MLPRegressor::predict` takes `&[Vec<f64>]` and returns `Result<Vec<f64>>`. Read `regressor.rs` to confirm the signature matches.

### 1B. Add MLP integration tests to `tests/correctness.rs`

**File: `crates/scry-learn/tests/correctness.rs`**

This file has correctness tests for every other model family. Add MLP tests:

1. **XOR test** — MLPClassifier with `hidden_layers(&[4])` should learn XOR (4 samples, 2 features, 2 classes). Assert accuracy = 100% after enough iterations.
2. **Regression test** — MLPRegressor on `y = sin(x)` or `y = x^2` with 50 samples. Assert MSE < threshold.
3. **Multi-class test** — MLPClassifier on a 3-class synthetic dataset (like the iris-like helper in cart.rs tests). Assert accuracy > 80%.

### 1C. Verify all neural module tests pass

```bash
cargo test -p scry-learn -- neural
cargo test -p scry-learn -- correctness
```

---

## Phase 2: 8C Large File Refactoring

The roadmap identifies 5 files past maintainability thresholds. Prioritize by impact:

### 2A. Split `cart.rs` (2040 lines → 4 files)

**Priority: HIGH** — most-modified file in codebase, complex unsafe code

| New File | Contents | Approx Lines |
|----------|----------|:------------:|
| `tree/cart/node.rs` | `TreeNode` enum + methods (predict, depth, n_leaves, prune_ccp, cost_complexity_pruning_path) | ~350 |
| `tree/cart/flat.rs` | `FlatNode`, `FlatTree`, `LEAF_SENTINEL`, flatten_dfs, all predict methods, depth/n_leaves | ~320 |
| `tree/cart/builder.rs` | `DecisionTreeClassifier` + `DecisionTreeRegressor` structs, fit methods, build_tree_presorted*, find_best_split* | ~1200 |
| `tree/cart/mod.rs` | Re-exports, `SplitCriterion`, `BestSplit`, helper fns (compute_impurity, majority_class, etc.) | ~200 |

**Rules:**
- Zero public API change — all existing `use scry_learn::tree::{FlatNode, FlatTree, ...}` must continue working
- Re-export everything from `tree/cart/mod.rs` that `tree/mod.rs` currently re-exports from `cart`
- All 428+ tests pass, clippy clean
- The `pub(crate)` visibility on methods like `apply_sample`, `fit_on_indices`, `fit_on_indices_presorted` must be preserved

### 2B. Split `search.rs` (~1400 lines → 4 files)

**Priority: MEDIUM** — many `impl Tunable` blocks, mechanical split

| New File | Contents |
|----------|----------|
| `search/grid.rs` | `GridSearchCV` struct + impl |
| `search/random.rs` | `RandomizedSearchCV` struct + impl |
| `search/tunable.rs` | `Tunable` trait + all 23 `impl Tunable for ...` blocks |
| `search/mod.rs` | `ParamValue`, `ParamGrid`, `CvResult`, re-exports |

### 2C. Others (lower priority, skip if time-constrained)

- `scry-chart/src/formatter.rs` (~1300 lines)
- `scry-chart/src/layout/mod.rs` (~1200 lines)
- `src/rasterize/skia.rs` (1187 lines)

---

## Phase 3: Verification

```bash
# All tests pass
cargo test --workspace

# Clippy clean
cargo clippy --workspace --all-targets

# Fuzz targets still build (the split shouldn't affect fuzz, but verify)
cargo +nightly fuzz build

# Doc build
cargo doc -p scry-learn --all-features --no-deps
```

---

## Files Modified

| File | Change |
|------|--------|
| `crates/scry-learn/src/pipeline.rs` | Add 2 `PipelineModel` impls for MLP |
| `crates/scry-learn/tests/correctness.rs` | Add MLP integration tests |
| `crates/scry-learn/src/tree/cart.rs` | **SPLIT** → `cart/{mod,node,flat,builder}.rs` |
| `crates/scry-learn/src/tree/mod.rs` | Update `mod cart` to point to directory module |
| `crates/scry-learn/src/search.rs` | **SPLIT** → `search/{mod,grid,random,tunable}.rs` |
| `crates/scry-learn/src/lib.rs` | Update `mod search` if it becomes a directory |

## Do NOT Touch

- `benches/` — another agent is working on benchmarks
- `BENCHMARKS.md` — same
- `examples/benchmark_comparison.rs`, `examples/industry_report.rs` — same
- Anything in `scry-engine` or `scry-chart` unless required by the refactor
