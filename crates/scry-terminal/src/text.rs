// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text rendering — cosmic-text shaping + glyphon atlas integration.
//!
//! Converts terminal grid lines into shaped, GPU-rendered text using
//! `cosmic-text` for text shaping/layout and `glyphon` for wgpu atlas
//! rendering.
//!
//! Each cell's foreground color, bold, italic, dim, hidden, and inverse
//! attributes are mapped to per-span `Attrs` via `set_rich_text`.

use cosmic_text::{Attrs, Buffer, Color as CsColor, Family, FontSystem, Metrics, Shaping, Style, Weight};
use glyphon::{SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport};

use crate::config::{ColorConfig, TerminalConfig};
use crate::grid::{CellColor, CellFlags, TerminalGrid};

/// The text rendering engine.
///
/// Manages font loading, text shaping, glyph caching, and GPU rendering.
pub struct TextEngine {
    /// Font system (handles font discovery and caching).
    font_system: FontSystem,
    /// Glyph cache for rasterization.
    swash_cache: SwashCache,
    /// GPU texture atlas for glyphs.
    atlas: TextAtlas,
    /// GPU text renderer.
    renderer: TextRenderer,
    /// Viewport for rendering.
    viewport: Viewport,
    /// Per-line text buffers for shaping.
    line_buffers: Vec<Buffer>,
    /// Cell dimensions derived from font metrics.
    cell_width: f32,
    cell_height: f32,
    /// Font size.
    font_size: f32,
    /// Line height.
    line_height: f32,
    /// Per-line dirty tracking (mirrors grid dirty flags to avoid reshaping clean lines).
    line_dirty: Vec<bool>,
}

impl TextEngine {
    /// Create a new text engine.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: &TerminalConfig,
    ) -> Self {
        let mut font_system = FontSystem::new();

        // Load custom font if specified
        if let Some(font_path) = &config.font.path {
            if let Ok(data) = std::fs::read(font_path) {
                font_system.db_mut().load_font_data(data);
            }
        }

        let swash_cache = SwashCache::new();

        let cache = glyphon::Cache::new(device);
        let mut atlas = TextAtlas::new(device, queue, &cache, surface_format);
        let renderer = TextRenderer::new(
            &mut atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );
        let viewport = Viewport::new(device, &cache);

        let font_size = config.font.size;
        let line_height = (font_size * 1.2).ceil();

        // Measure monospace cell dimensions using advance width
        let metrics = Metrics::new(font_size, line_height);
        let cell_width = measure_cell_width(&mut font_system, metrics, font_size);

        Self {
            font_system,
            swash_cache,
            atlas,
            renderer,
            viewport,
            line_buffers: Vec::new(),
            cell_width,
            cell_height: line_height,
            font_size,
            line_height,
            line_dirty: Vec::new(),
        }
    }

    /// Cell width in pixels.
    pub fn cell_width(&self) -> f32 {
        self.cell_width
    }

    /// Cell height in pixels.
    pub fn cell_height(&self) -> f32 {
        self.cell_height
    }

    /// Prepare text rendering for a frame.
    ///
    /// Shapes dirty lines and updates the glyph atlas. Only lines flagged
    /// as dirty are re-shaped, dramatically reducing per-frame work.
    /// Current font size in pixels.
    pub fn font_size(&self) -> f32 {
        self.font_size
    }

    /// Change the font size, re-measure cell dimensions, and mark all lines dirty.
    pub fn set_font_size(&mut self, new_size: f32) {
        self.font_size = new_size;
        self.line_height = (new_size * 1.2).ceil();

        // Re-measure cell dimensions
        let metrics = Metrics::new(self.font_size, self.line_height);
        self.cell_width = measure_cell_width(&mut self.font_system, metrics, self.font_size);
        self.cell_height = self.line_height;

        // Recreate line buffers with new metrics
        let count = self.line_buffers.len();
        self.line_buffers.clear();
        for _ in 0..count {
            self.line_buffers
                .push(Buffer::new(&mut self.font_system, metrics));
        }

        // Mark all dirty
        for d in &mut self.line_dirty {
            *d = true;
        }
    }

    /// Prepare text rendering for a frame.
    ///
    /// Shapes dirty lines and updates the glyph atlas. Only lines flagged
    /// as dirty are re-shaped, dramatically reducing per-frame work.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &TerminalGrid,
        colors: &ColorConfig,
        screen_width: u32,
        screen_height: u32,
        padding: f32,
    ) -> Result<(), glyphon::PrepareError> {
        let rows = grid.rows() as usize;
        let cols = grid.cols() as usize;

        // Update viewport
        self.viewport.update(
            queue,
            glyphon::Resolution {
                width: screen_width,
                height: screen_height,
            },
        );

        // Ensure we have enough line buffers and dirty flags
        while self.line_buffers.len() < rows {
            let metrics = Metrics::new(self.font_size, self.line_height);
            let buf = Buffer::new(&mut self.font_system, metrics);
            self.line_buffers.push(buf);
            self.line_dirty.push(true); // New lines are considered dirty
        }
        // Trim if too many (e.g. after resize down)
        self.line_buffers.truncate(rows);
        self.line_dirty.truncate(rows);

        // Shape only dirty lines using per-cell rich text attributes
        for row in 0..rows {
            if !grid.is_dirty(row as u16) && !self.line_dirty.get(row).copied().unwrap_or(true) {
                continue; // Skip clean lines
            }

            // Build per-cell spans: each cell becomes a (text, Attrs) pair.
            // We coalesce runs of cells with identical attributes to minimize spans.
            let mut spans: Vec<(String, Attrs<'_>)> = Vec::new();
            let mut current_text = String::new();
            let mut current_attrs: Option<Attrs<'_>> = None;

            for col in 0..cols {
                let cell = grid.viewport_cell(col as u16, row as u16);

                // Skip continuation cells (second column of wide char)
                if cell.width == 0 {
                    continue;
                }

                let attrs = self.cell_to_attrs(cell.fg, cell.bg, cell.flags, colors);

                match &current_attrs {
                    Some(prev) if attrs_equal(prev, &attrs) => {
                        // Same attributes — extend the current span
                        cell.write_grapheme(&mut current_text);
                    }
                    _ => {
                        // Different attributes — flush previous span and start new one
                        if let Some(prev_attrs) = current_attrs.take() {
                            if !current_text.is_empty() {
                                spans.push((current_text, prev_attrs));
                                current_text = String::new();
                            }
                        }
                        cell.write_grapheme(&mut current_text);
                        current_attrs = Some(attrs);
                    }
                }
            }
            // Flush the last span
            if let Some(prev_attrs) = current_attrs {
                if !current_text.is_empty() {
                    spans.push((current_text, prev_attrs));
                }
            }

            // If the line is entirely empty, set plain text to avoid issues
            if spans.is_empty() {
                let buf = &mut self.line_buffers[row];
                buf.set_text(
                    &mut self.font_system,
                    "",
                    Attrs::new().family(Family::Monospace),
                    Shaping::Advanced,
                );
            } else {
                // Convert to (&str, Attrs) references for set_rich_text
                let span_refs: Vec<(&str, Attrs<'_>)> =
                    spans.iter().map(|(s, a)| (s.as_str(), *a)).collect();

                let default_attrs = Attrs::new().family(Family::Monospace);
                let buf = &mut self.line_buffers[row];
                buf.set_rich_text(
                    &mut self.font_system,
                    span_refs,
                    default_attrs,
                    Shaping::Advanced,
                );
            }

            let buf = &mut self.line_buffers[row];
            buf.set_size(
                &mut self.font_system,
                Some(screen_width as f32),
                Some(self.line_height),
            );
            buf.shape_until_scroll(&mut self.font_system, false);

            // Mark this line as clean in our tracking
            if let Some(d) = self.line_dirty.get_mut(row) {
                *d = false;
            }
        }

        // Phase 2: Build TextAreas (only needs &self.line_buffers)
        let (r, g, b) = colors.fg_rgb();
        let default_color = glyphon::Color::rgb(r, g, b);
        let cell_height = self.cell_height;

        let text_areas: Vec<TextArea> = self
            .line_buffers
            .iter()
            .enumerate()
            .map(|(row, buf)| {
                let y = padding + row as f32 * cell_height;
                TextArea {
                    buffer: buf,
                    left: padding,
                    top: y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: padding as i32,
                        top: y as i32,
                        right: screen_width as i32,
                        bottom: (y + cell_height) as i32,
                    },
                    default_color,
                    custom_glyphs: &[],
                }
            })
            .collect();

        self.renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        )
    }

    /// Render text into the given render pass.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) -> Result<(), glyphon::RenderError> {
        self.renderer.render(&self.atlas, &self.viewport, pass)
    }

    /// Trim the atlas to free unused space.
    pub fn trim(&mut self) {
        self.atlas.trim();
    }

    /// Force all lines dirty (e.g. after resize or full redraw).
    pub fn mark_all_dirty(&mut self) {
        for d in &mut self.line_dirty {
            *d = true;
        }
    }

    // ── Attribute mapping ──────────────────────────────────────────

    /// Map a cell's colors and flags to cosmic-text `Attrs`.
    fn cell_to_attrs<'a>(
        &self,
        fg: CellColor,
        bg: CellColor,
        flags: CellFlags,
        colors: &ColorConfig,
    ) -> Attrs<'a> {
        // Determine effective foreground color (handle inverse)
        let (r, g, b) = if flags.contains(CellFlags::INVERSE) {
            // Inverse: use background color as text color
            bg.resolve(false, colors)
        } else {
            fg.resolve(true, colors)
        };

        // Apply dim: reduce alpha
        let alpha = if flags.contains(CellFlags::HIDDEN) {
            0
        } else if flags.contains(CellFlags::DIM) {
            128
        } else {
            255
        };

        let color = CsColor::rgba(r, g, b, alpha);

        let weight = if flags.contains(CellFlags::BOLD) {
            Weight::BOLD
        } else {
            Weight::NORMAL
        };

        let style = if flags.contains(CellFlags::ITALIC) {
            Style::Italic
        } else {
            Style::Normal
        };

        Attrs::new()
            .family(Family::Monospace)
            .color(color)
            .weight(weight)
            .style(style)
    }
}

/// Compare two `Attrs` for equality of the fields we care about
/// (color, weight, style). This avoids reshaping when attributes match.
fn attrs_equal(a: &Attrs<'_>, b: &Attrs<'_>) -> bool {
    a.color_opt == b.color_opt && a.weight == b.weight && a.style == b.style && a.family == b.family
}

/// Measure monospace cell width by shaping two characters and computing the
/// advance (position difference).  Falls back to single-glyph `w` if only
/// one glyph is produced.
fn measure_cell_width(font_system: &mut FontSystem, metrics: Metrics, font_size: f32) -> f32 {
    let mono = Attrs::new().family(Family::Monospace);
    let mut buf = Buffer::new(font_system, metrics);
    buf.set_text(font_system, "MM", mono, Shaping::Advanced);
    buf.shape_until_scroll(font_system, false);

    if let Some(run) = buf.layout_runs().next() {
        let glyphs: Vec<_> = run.glyphs.iter().collect();
        if glyphs.len() >= 2 {
            // Advance = difference between the x positions of consecutive glyphs
            let advance = glyphs[1].x - glyphs[0].x;
            if advance > 0.0 {
                return advance;
            }
        }
        // Fallback: single glyph hitbox width
        if let Some(g) = glyphs.first() {
            return g.w;
        }
    }
    font_size * 0.6
}
