# Unsafe Code Audit — `scry-engine`

This document catalogues every `unsafe` block in the crate, following the
conventions of [the Rust standard library's SAFETY comments](https://std-dev-guide.rust-lang.org/policy/safety-comments.html).

## Policy

The crate root declares `#![deny(unsafe_code)]`. Individual modules that
require unsafe FFI must opt in with `#![allow(unsafe_code)]` and document
every block.

---

## `transport/shm.rs` — POSIX Shared Memory

This module is gated behind `--features shm` and provides zero-copy
Kitty protocol transmission via `shm_open` / `mmap`.

### `unsafe impl Send for ShmBuffer`

**Location:** `shm.rs:55`

**Justification:** `ShmBuffer` contains a `*mut u8` (from `mmap`) which
prevents auto-`Send`. The pointer is exclusively owned by the struct—no
aliases exist—and the mapped region is process-local. Moving the struct
between threads is safe because only one thread ever writes to the mapping
at a time (enforced by `&self` / `&mut self` receiver rules).

### `ShmBuffer::new()` — `shm_open`, `ftruncate`, `mmap`

**Location:** `shm.rs:68–115`

| FFI Call | Pre-conditions | Post-conditions | Error handling |
|----------|---------------|-----------------|----------------|
| `shm_open` | `c_name` is a valid NUL-terminated C string (constructed via `CString::new`) | Returns fd ≥ 0 on success | `fd < 0` → return `io::Error` |
| `ftruncate` | `fd` is valid from `shm_open` | Sets object size | `ret < 0` → close fd, return error |
| `mmap` | `fd` valid, `capacity > 0` | Returns non-`MAP_FAILED` pointer | `MAP_FAILED` → return error |
| `close(fd)` | Always called after mmap | Releases fd; mapping survives | Infallible (errors ignored per POSIX convention) |

**Panic conditions:** None in the unsafe blocks. The function returns `Result`.

### `ShmBuffer::write()` — `ptr::copy_nonoverlapping`

**Location:** `shm.rs:130–132`

**Pre-conditions:**
- `self.ptr` is non-null — enforced by an `assert!` before the unsafe block.
- `data.len() <= self.capacity` — enforced by an `assert!` before the unsafe block.
- `self.ptr` is valid (established in `new()` / `resize()`, nulled on failure).
- Source and destination do not overlap (src = stack/heap slice, dst = mmap region).

**Panic conditions:** The `assert!` panics if the pointer is null or data
exceeds capacity. The unsafe block itself cannot panic.

### `ShmBuffer::resize()` — `munmap`, re-`shm_open`, `ftruncate`, `mmap`

**Location:** `shm.rs:156–188`

Same FFI calls as `new()` with the addition of `munmap` to release the
old mapping before remapping. Error handling follows the same pattern:
each syscall is checked and errors are propagated as `io::Result`.

**Critical invariant:** After `munmap`, `self.ptr` is immediately set to
`null_mut()` and `self.capacity` to `0`. If the subsequent `shm_open` or
`mmap` fails, the struct is left in a safe null state. `write()` will panic
on the null-pointer assertion, and `Drop` skips `munmap` for null pointers.

> **Resolved:** The improvement opportunity from the original audit
> (setting `ptr = null_mut()` after `munmap`) has been implemented.

### `ShmBuffer::drop()` — `munmap`, `shm_unlink`

**Location:** `shm.rs:203–206`

- `munmap(self.ptr, self.capacity)` — releases the mapping (skipped if `self.ptr` is null).
- `shm_unlink(self.c_name)` — removes the named object from the filesystem.

**Double-free protection:** `Drop` runs exactly once (Rust ownership).
Null-pointer check guards against `munmap` on an invalidated buffer.
**Use-after-unlink:** Not possible because the struct is consumed.

---

## `transport/picker.rs` — Terminal Size Detection

### `libc_ioctl` — TIOCGWINSZ

**Location:** `picker.rs:205–214`

```rust
let mut ws = MaybeUninit::<Winsize>::uninit();
let result = unsafe { libc_ioctl(1, TIOCGWINSZ_VAL, ws.as_mut_ptr()) };
// ... check result ...
let ws = unsafe { ws.assume_init() };
```

**Pre-conditions:**
- `MaybeUninit` provides properly aligned, sufficiently sized storage.
- `TIOCGWINSZ` is a well-known ioctl that writes a `Winsize` struct.
- `assume_init()` is only called after verifying `result >= 0` (success).

**Risk:** Low. This is a standard pattern used by every terminal library
(`crossterm`, `termion`, `termsize`).

---

## Summary

| Module | Unsafe blocks | Risk | Miri testable? |
|--------|:---:|:---:|:---:|
| `shm.rs` | 12 | Medium (FFI + raw pointers) | ❌ (FFI) |
| `picker.rs` | 2 | Low (standard ioctl) | ❌ (FFI) |
| All other core modules | 0 | N/A | ✅ |
| `scry-chart` (entire crate) | 0 (`#![deny(unsafe_code)]`) | N/A | ✅ |

**Miri coverage (core crate):** 125/126 tests pass under Miri.
The `full_pipeline_sixel` test is skipped — it involves heavy image encoding
(`flate2`/`png`) that exceeds Miri's interpreted execution speed. This is not
undefined behavior; it is a Miri performance limitation.

**Miri coverage (scry-chart crate):** 9/9 tests pass under Miri.
Two log-scale tests were updated to use tolerance-based assertions
(`1e-9` epsilon) because Miri's software-float emulation rounds
`10.0_f64.powf(3.0)` to `999.999…95` instead of exactly `1000.0`.

**Fuzz targets (6 total):**
| Target | Crate | Coverage |
|--------|-------|----------|
| `fuzz_kitty_escape` | core | Kitty escape sequence parser |
| `fuzz_rasterize` | core | Scene rasterization pipeline |
| `fuzz_scene_hash` | core | Content hash determinism |
| `fuzz_scale` | scry-chart | LinearScale, LogScale, CategoricalScale |
| `fuzz_chart_render` | scry-chart | Full render pipeline, all 7 chart types |
| `fuzz_chart_builder` | scry-chart | Builder chain with arbitrary method combos |
