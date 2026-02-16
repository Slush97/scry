---
description: How to commit work after completing a task or session
---

# Git Commit Workflow

Run this after completing a meaningful unit of work (feature, fix, refactor session).

## 1. Verify before committing

// turbo-all

```bash
# Workspace compiles clean
cargo check --workspace

# Zero clippy warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Tests pass (use --release for scry-learn speed)
cargo test --workspace --exclude scry-learn
cargo test -p scry-learn --release

# Insta snapshots are up to date (no pending reviews)
cargo insta test --workspace --review
```

## 2. Stage changes

```bash
# Review what changed
git status
git diff --stat

# Stage everything (or selectively)
git add -A
```

## 3. Write a conventional commit message

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `chore`
**Scopes:** `engine`, `chart`, `learn`, `cli`, `pipe`, `workspace`

Examples:
```
feat(learn): add BernoulliNB and MultinomialNB classifiers
fix(chart): correct Y-axis tick spacing for log scale
perf(engine): batch consecutive same-style shape commands
refactor(learn): split cart.rs into cart/{node,builder,predict}.rs
test(chart): add insta snapshots for all 19 chart types
docs(workspace): update BENCHMARKS.md with HistGBT vs XGBoost results
ci: add Miri job for scry-learn crate
```

## 4. Commit

```bash
git commit -m "<type>(<scope>): <description>"
```

## 5. Push (if remote is configured)

```bash
git push origin main
```

## Rules

- **Never commit with failing tests** — all tests must pass first
- **Never commit with clippy warnings** — zero tolerance
- **One logical change per commit** — don't mix features with refactors
- **Update CHANGELOG.md** for user-facing changes (feat, fix, perf)
- **Update .agent/ROADMAP.md** if completing a sprint item — mark it ✅
- **Update .agent/workflows/next-agent-handoff.md** at end of session
