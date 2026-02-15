---
description: Sprint 1 hardening — CI, clippy, oob_score, README update
---

# Sprint 1 Hardening Workflow

Quick wins that compound into trust and professional signal.

**Status: ✅ COMPLETE** (2026-02-14)

// turbo-all

## Steps

### 1. ✅ Kill All Clippy Warnings — Done 2026-02-14

scry-chart (0 errors), scry-pipe (0 errors), and scry-learn (0 errors — 67 warnings fixed 2026-02-14) are all clean.

### 2. ✅ Implement RF OOB Score — Done 2026-02-14

**File:** `crates/scry-learn/src/tree/random_forest.rs`

Implemented real OOB scoring in `fit()`:
1. Parallel closure returns `(tree, bootstrap_indices)`
2. New `compute_oob_score()` uses `HashSet` to identify OOB samples per tree
3. Accumulates majority-vote predictions, computes accuracy
4. Stored in `self.oob_score_` (only when `bootstrap=true`)

Tests: `test_oob_score_with_bootstrap` (≥ 0.80), `test_oob_score_without_bootstrap` (`None`). Both pass.

```bash
cargo test -p scry-learn --lib -- random_forest::tests::test_oob
```

### 3. ✅ Create CI/CD Pipeline — Done 2026-02-14

**File:** `.github/workflows/ci.yml`

Pre-existing CI already had check, fmt, clippy, test, doc, MSRV, feature combos, publish-check, Miri, and fuzz. Added `test-crates` matrix job:

- `cargo test -p scry-learn`
- `cargo test -p scry-learn --features serde`
- `cargo test -p scry-chart`

### 4. ✅ Update README — Done 2026-02-14

**File:** `README.md`

- scry-chart: "10 chart types" → "17 chart types, 6 themes"
- scry-learn: expanded to full model list (13 algorithms + preprocessing + cross-validation)
- Added scry-pipe row to workspace table
- Added Crate Maturity stability table

### 5. ✅ Verify — Done 2026-02-14

| Check | Result |
|-------|--------|
| `cargo check --workspace` | ✅ Clean |
| `cargo test -p scry-learn --lib` | ✅ 189/189 pass |
| `cargo clippy -p scry-learn -- -D warnings` | ✅ 0 errors (67 fixed 2026-02-14) |
| `cargo test -p scry-learn --test correctness` | ✅ 37/37 pass |
| `cargo bench --bench ml_algorithms -p scry-learn -- --test` | ✅ All pass |
