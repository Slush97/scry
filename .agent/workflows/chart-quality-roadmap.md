---
description: Multi-session implementation roadmap for scry-chart rendering quality hardening
---

# scry-chart Quality Roadmap — Cross-Session Workflow

## Status

**Current Phase: Complete** (2026-02-15)

| Session | Focus | Status |
|---------|-------|--------|
| 1 | Heatmap RenderContext Migration + Contrast Text | ✅ Complete |
| 2 | Font Scaling Consistency + Proportional Offsets | ✅ Complete |
| 3 | Code Hygiene + SVG Standards Compliance | ✅ Complete |
| 4 | Theme-Aware Color System + New Tests | ✅ Complete |

## Background

This roadmap addresses 15 findings from a comprehensive quality audit of the scry-chart
rendering pipeline (~5,000 lines). The audit graded the library at **B+** — solid architecture
with gaps in text contrast, font metric consistency, and heatmap layout isolation. These
sessions close those gaps to reach **A-tier** quality parity with D3/matplotlib/Plotly.

**Audit report**: See `walkthrough.md` in the most recent chart-audit conversation.

**Every agent working on this roadmap MUST:**
1. Read this workflow file first
2. Read the session they're implementing IN FULL before writing any code
3. Run verification commands after each major change
4. Update the Status table after completing each session
5. Read Context Files listed below before touching any module

## Context Files to Read

### Core Architecture
- `crates/scry-chart/src/lib.rs` — Module structure, prelude re-exports
- `crates/scry-chart/src/layout/mod.rs` — Layout engine, `RenderContext`, `TextOverlay`, `scaled_font_size()`
- `crates/scry-chart/src/theme.rs` — 6 themes, `TextStyle`, `AxisTheme`, `SeriesTheme`
- `crates/scry-chart/src/axis.rs` — Axis rendering, `AVG_CHAR_WIDTH`, auto-skip

### Per-Chart Layout Files (read the one you're modifying)
- `crates/scry-chart/src/layout/heatmap.rs` — Bypasses RenderContext (Session 1 target)
- `crates/scry-chart/src/layout/pie.rs` — Hardcoded white text (Session 1 target)
- `crates/scry-chart/src/layout/gauge.rs` — Fixed 16px offset (Session 2 target)
- `crates/scry-chart/src/layout/radar.rs` — Fixed 12px offset (Session 2 target)
- `crates/scry-chart/src/layout/funnel.rs` — Manual color lightening (Session 4 target)
- `crates/scry-chart/src/layout/scatter.rs` — Reference for proper RenderContext usage
- `crates/scry-chart/src/layout/bar.rs` — Reference for proper RenderContext usage

### Export & Widget
- `crates/scry-chart/src/svg_export.rs` — SVG generation, `font-size` units (Session 3)
- `crates/scry-chart/src/widget.rs` — Terminal text rendering, alignment (Session 3)

### Tests
- `crates/scry-chart/tests/render_tests.rs` — 70+ snapshot tests (1505 lines)
- `crates/scry-chart/tests/acid_test.rs` — Stress tests
- `crates/scry-chart/tests/edge_cases.rs` — Edge case coverage

## Verification Commands

// turbo-all

1. Run unit tests:
```bash
cargo test -p scry-chart --lib
```

2. Run render snapshot tests:
```bash
cargo test -p scry-chart --test render_tests
```

3. Run all chart tests:
```bash
cargo test -p scry-chart
```

4. Run clippy:
```bash
cargo clippy -p scry-chart -- -D warnings
```

5. Workspace check:
```bash
cargo check --workspace
```

6. Update snapshots after intentional layout changes:
```bash
cargo insta test -p scry-chart --review
```

---

## Session 1: Heatmap RenderContext Migration + Contrast Text Helper

**Goal:** Fix the two highest-severity audit findings — heatmap's isolated rendering path
and hardcoded white text that breaks on light themes.

**Estimated effort:** 1 session (2-3 hours)

**Audit findings addressed:** #1 (P0), #2 (P1)

### 1A. Contrast Text Color Helper

**New function in:** `src/theme.rs`

**Implementation:**
- Add `contrast_text_color(background: Color) -> Color` that returns black or white based
  on WCAG 2.0 relative luminance:
  ```rust
  pub fn contrast_text_color(bg: Color) -> Color {
      // sRGB → linear → relative luminance
      let lum = 0.2126 * linearize(bg.r) + 0.7152 * linearize(bg.g) + 0.0722 * linearize(bg.b);
      if lum > 0.179 { Color::BLACK } else { Color::WHITE }
  }
  
  fn linearize(c: f32) -> f32 {
      if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
  }
  ```
- This follows the W3C WCAG 2.0 contrast algorithm exactly

**Unit tests:**
- `contrast_text_color(Color::WHITE) == Color::BLACK`
- `contrast_text_color(Color::BLACK) == Color::WHITE`
- `contrast_text_color(dark_blue) == Color::WHITE`
- `contrast_text_color(yellow) == Color::BLACK`

### 1B. Update Pie Chart Text Color

**Modify:** `src/layout/pie.rs`

**Change:** Replace hardcoded `Color::from_rgba8(255, 255, 255, 230)` for percentage labels
with `contrast_text_color(slice_color)`.

```diff
- color: Color::from_rgba8(255, 255, 255, 230),
+ color: contrast_text_color(series_color),
```

### 1C. Update Heatmap Text Color

**Modify:** `src/layout/heatmap.rs`

**Change:** Replace the `if t > 0.5 { white } else { light_gray }` logic with
`contrast_text_color(cell_color)` — computed from the actual mapped cell color.

```diff
- color: if t > 0.5 {
-     Color::from_rgba8(255, 255, 255, 220)
- } else {
-     Color::from_rgba8(200, 200, 200, 220)
- },
+ color: contrast_text_color(cell_color),
```

### 1D. Migrate Heatmap to RenderContext

**Modify:** `src/layout/heatmap.rs`

This is the largest change. Currently, `draw_heatmap` creates a raw `PixelCanvas` and
`Vec<TextOverlay>` directly, bypassing `RenderContext`. It also uses hardcoded layout
constants (row label width `7.0 * max_chars`, column label height `20.0`).

**Steps:**
1. Change `draw_heatmap` signature to accept `&mut RenderContext` instead of returning
   `(PixelCanvas, Vec<TextOverlay>)`
2. Use `ctx.canvas` for rectangle drawing (currently does `canvas.add_command(...)`)
3. Use `ctx.overlays` for text overlays (currently pushes to local `Vec`)
4. Replace hardcoded `7.0` char width with `char_width_for_size(tick_font_size)` from `layout/mod.rs`
5. Replace hardcoded `20.0` column label height with `proportional_x_tick_height()` or similar

**Also modify:** `src/layout/mod.rs` — Update `render_chart()` match arm for `ChartKind::Heatmap`
to use RenderContext flow (same as scatter, bar, line). This ensures `add_common_overlays()`
is called, so subtitles and footers work on heatmaps.

**Key risk:** Heatmap snapshots will change. Update with `cargo insta test -p scry-chart --review`.

### Session 1 Verification Checklist
- [ ] `cargo test -p scry-chart --lib` passes
- [ ] `cargo test -p scry-chart --test render_tests` passes (after snapshot update)
- [ ] `cargo clippy -p scry-chart -- -D warnings` clean
- [ ] `cargo check --workspace` clean
- [ ] Contrast helper has unit tests in `theme.rs`
- [ ] Heatmap now receives subtitles/footers via `add_common_overlays()`

---

## Session 2: Font Scaling Consistency + Proportional Offsets

**Goal:** Ensure all font metric computations respect dynamic scaling and replace all
fixed-pixel offsets with proportional ones.

**Estimated effort:** 1 session (1.5-2 hours)

**Audit findings addressed:** #4 (P1), #5 (P2), #6 (P2), #8 (P2)

### 2A. Fix `estimate_y_label_width` to Use Scaled Font Size

**Modify:** `src/layout/mod.rs`

**Change:** The function currently uses a hardcoded `7.5` pixel-per-char. It should accept
canvas dimensions and use `scaled_font_size(theme.text.label_size, w, h)` to compute
the actual character width.

```diff
- fn estimate_y_label_width(label: Option<&str>) -> f32 {
-     label.map_or(0.0, |l| {
-         let chars = l.chars().count() as f32;
-         chars * 7.5 + 12.0
-     })
- }
+ fn estimate_y_label_width(label: Option<&str>, w: u32, h: u32, label_size: f32) -> f32 {
+     label.map_or(0.0, |l| {
+         let fs = scaled_font_size(label_size, w, h);
+         let char_w = fs * INTER_ADVANCE_RATIO;
+         let chars = l.chars().count() as f32;
+         chars * char_w + fs  // padding = 1em
+     })
+ }
```

**Also update** all call sites of `estimate_y_label_width` to pass `(w, h, theme.text.label_size)`.

### 2B. Proportional Gauge Value Label Offset

**Modify:** `src/layout/gauge.rs`

**Change:** Replace fixed `center_y + 16.0` with proportional offset.

```diff
- y_px: center_y + 16.0,
+ y_px: center_y + radius * 0.15,
```

### 2C. Proportional Radar Label Offset

**Modify:** `src/layout/radar.rs`

**Change:** Replace fixed `radius + 12.0` with proportional offset.

```diff
- let label_r = radius + 12.0;
+ let label_r = radius + scaled_font_size(theme.text.tick_size, w, h) * 1.2;
```

This uses 1.2× the tick font size as the gap, which scales naturally with canvas size.

### 2D. Standardize `data_fs` Base Size

**Modify:** `src/layout/heatmap.rs`

**Change:** Align heatmap's `data_fs` base from `10.0` to `9.0` to match scatter, bar,
gauge, and funnel. This is a minor consistency fix.

```diff
- let data_fs = scaled_font_size(10.0, w, h);
+ let data_fs = scaled_font_size(9.0, w, h);
```

### Session 2 Verification Checklist
- [ ] `cargo test -p scry-chart --lib` passes
- [ ] `cargo test -p scry-chart --test render_tests` passes (after snapshot update)
- [ ] `cargo clippy -p scry-chart -- -D warnings` clean
- [ ] `cargo check --workspace` clean
- [ ] Y-label width test: render chart at 400×300 and 2000×1200, verify labels don't overlap axis

---

## Session 3: Code Hygiene + SVG Standards Compliance

**Goal:** Consolidate duplicated constants, fix SVG unit specification, document terminal
font-size limitation, and fix minor alignment precision.

**Estimated effort:** 1 session (1-1.5 hours)

**Audit findings addressed:** #3 (P2), #9 (P3), #10 (P2 doc), #11 (P3), #12 (P3), #13 (Info)

### 3A. Consolidate `AVG_CHAR_WIDTH`

**Modify:** `src/layout/mod.rs`, `src/axis.rs`

**Steps:**
1. Keep `INTER_ADVANCE_RATIO` in `layout/mod.rs` as the canonical constant (it has the
   better name)
2. Make it `pub(crate)` so `axis.rs` can import it
3. Remove `AVG_CHAR_WIDTH` from `axis.rs`, replace with `use crate::layout::INTER_ADVANCE_RATIO`
4. Also add a `char_width_for_size(font_size: f32) -> f32` helper:
   ```rust
   pub(crate) fn char_width_for_size(font_size: f32) -> f32 {
       font_size * INTER_ADVANCE_RATIO
   }
   ```

### 3B. SVG `font-size` Units

**Modify:** `src/svg_export.rs`

**Change:** Add explicit `px` unit to SVG text `font-size` attribute.

```diff
- write!(svg, " font-size=\"{}\"", overlay.font_size)
+ write!(svg, " font-size=\"{}px\"", overlay.font_size)
```

### 3C. SVG `dominant-baseline` Per Role

**Modify:** `src/svg_export.rs`

**Change:** Use `dominant-baseline="hanging"` for title text and `"central"` for labels/ticks.
This requires checking `overlay.align` or adding a `role` field to `TextOverlay`.

**Simpler approach:** Since the layout engine already positions text accounting for baseline,
just document that `dominant-baseline="central"` is used uniformly and all positions are
adjusted accordingly. Add a comment to the SVG emit code.

### 3D. Widget Center-Align Precision Fix

**Modify:** `src/widget.rs`

**Change:** Use rounding instead of truncation for center-aligned text.

```diff
- TextAlign::Center => abs_x.saturating_sub(text_len / 2),
+ TextAlign::Center => abs_x.saturating_sub((text_len + 1) / 2),
```

### 3E. Document Terminal Font-Size Limitation

**Modify:** `src/layout/mod.rs`

**Add doc comment** to `TextOverlay::font_size` explaining that this field is only
meaningful for SVG/PNG export and is ignored in the terminal widget path (character-cell
terminals cannot vary glyph size).

### 3F. Document Diagonal Text Limitation

**Modify:** `src/widget.rs`

**Add doc comment** to the diagonal rotation branch explaining the staircase effect
is inherent to character-cell rendering.

### Session 3 Verification Checklist
- [ ] `cargo test -p scry-chart --lib` passes
- [ ] `cargo test -p scry-chart --test render_tests` passes
- [ ] `cargo clippy -p scry-chart -- -D warnings` clean
- [ ] `cargo check --workspace` clean
- [ ] `grep -r "AVG_CHAR_WIDTH" crates/scry-chart/src/` returns 0 results (constant removed)
- [ ] SVG output includes `px` unit in font-size

---

## Session 4: Theme-Aware Color System + Quality Tests

**Goal:** Fix funnel's manual color lightening and add dedicated quality audit tests that
verify contrast, scaling, and layout correctness programmatically.

**Estimated effort:** 1 session (2 hours)

**Audit findings addressed:** #7 (P3), plus new test infrastructure

### 4A. Funnel Theme-Aware Stage Colors

**Modify:** `src/layout/funnel.rs`

**Change:** Replace manual `mix(white, fraction)` color lightening with `color.with_alpha()`
multiplied by stage index, or use `contrast_text_color()` for the label text.

```diff
- let stage_color = base_color.mix(Color::WHITE, i as f32 / n as f32 * 0.4);
+ let alpha = 1.0 - (i as f32 / n as f32 * 0.4);
+ let stage_color = Color { r: base_color.r, g: base_color.g, b: base_color.b, a: alpha };
```

Also update funnel label text to use `contrast_text_color(stage_color)`.

### 4B. Quality Audit Test Suite

**New file:** `crates/scry-chart/tests/quality_audit.rs`

**Tests to add:**

1. **Contrast ratio test**: For every chart type with data labels on colored backgrounds
   (pie, heatmap, funnel), render at 400×300 with each of the 6 themes, extract text
   overlay colors vs background colors, verify WCAG AA contrast ratio ≥ 4.5:1.

2. **Font scaling consistency test**: Render the same chart at 40×30, 400×300, and
   2000×1200. Verify that:
   - All text overlay `font_size` values scale proportionally
   - No `font_size` is below 7.0 or above 48.0 (clamp range)
   - Title font > label font > tick font (hierarchy preserved)

3. **Label non-overlap test**: Render bar chart with 20 long category labels at 400×300.
   Verify that `auto_skip_labels` reduces the count and no two adjacent labels occupy
   the same vertical pixel band.

4. **Heatmap subtitle test**: Render heatmap with `.subtitle("Sub")`. Verify the subtitle
   appears in `text_overlays` (regression test for Session 1D).

5. **Proportional offset test**: Render gauge and radar at 100×75 and 2000×1200. Verify
   that label offsets scale (not fixed 16px / 12px at both sizes).

### Session 4 Verification Checklist
- [ ] `cargo test -p scry-chart --lib` passes
- [ ] `cargo test -p scry-chart --test render_tests` passes
- [ ] `cargo test -p scry-chart --test quality_audit` passes
- [ ] `cargo clippy -p scry-chart -- -D warnings` clean
- [ ] `cargo check --workspace` clean

---

## Code Quality Rules

All sessions must follow these rules:

1. **No new hardcoded pixel sizes** — use `scaled_font_size()` or proportional calculations
2. **No new hardcoded text colors** — use `contrast_text_color()` or theme tokens
3. **All layout through RenderContext** — no chart type should create raw `PixelCanvas`
4. **Snapshot updates** — run `cargo insta test -p scry-chart --review` after layout changes
5. **Clippy clean** — zero warnings after every session
6. **Doc comments** — every public function and notable internal function gets documentation

## Known Issues & Gotchas

- **Snapshot churn**: Sessions 1 and 2 will change many snapshots. Review carefully with
  `cargo insta review` to ensure changes are improvements, not regressions.
- **Heatmap bypass is structural**: Session 1D requires understanding the `render_chart()`
  match arms in `layout/mod.rs`. Study how scatter/bar/line flow through `RenderContext`
  before modifying heatmap.
- **`contrast_text_color` depends on Color's channel range**: scry-engine `Color` uses
  `f32` channels in `[0.0, 1.0]`. Verify this before implementing the WCAG formula.
- **Widget text is character-cell**: Sessions 3D and 3F are documentation-only for the
  terminal path. The real font-size handling only applies to SVG/PNG export.
