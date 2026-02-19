// SPDX-License-Identifier: MIT OR Apache-2.0
//! 3D chart module — interactive 3D scatter plots and scene rendering.
//!
//! # Architecture
//!
//! Scene and camera logic is **independent of any renderer**. The rendering
//! backend is abstracted via the [`Rasterizer3D`] trait:
//!
//! ```text
//! Scene3D + Camera3D + PerspectiveProjection
//!                 ↓ (projected points, segments, labels)
//!           Rasterizer3D trait
//!                 ↓
//!     ┌───────────────────────────┐
//!     │  SkiaRasterizer3D (v1)    │ ← tiny-skia + fontdue
//!     │  WgpuRasterizer3D (v2)    │ ← Sprint 9
//!     └───────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use scry_chart::chart3d::Chart3D;
//!
//! let png = Chart3D::scatter(
//!     &[1.0, 2.0, 3.0],
//!     &[4.0, 5.0, 6.0],
//!     &[7.0, 8.0, 9.0],
//! )
//! .title("My 3D Scatter")
//! .x_label("Feature 1")
//! .y_label("Feature 2")
//! .z_label("Feature 3")
//! .render_to_png(800, 600)?;
//! ```

pub mod camera;
pub mod projection;
pub mod scene;
#[cfg(feature = "sdf")]
pub mod sdf_surface;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

use camera::{Camera3D, Vec3};
use projection::{
    depth_sort, mat4_mul, project_batch_precomputed, PerspectiveProjection, ProjectedPoint,
};
use scene::{AxisConfig3D, PointCloud3D, Scene3D};
use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Default color palette
// ---------------------------------------------------------------------------

/// Default category color palette for 3D charts.
///
/// Six high-contrast, semi-transparent colors suitable for dark backgrounds.
/// Used by [`Chart3D::color_by_labels`], [`Chart3D::color_by_class`], and
/// the default rendering pipeline.
#[must_use]
pub fn default_palette() -> [Color; 6] {
    [
        Color::from_rgba8(99, 190, 255, 230),  // blue
        Color::from_rgba8(255, 107, 107, 230), // red
        Color::from_rgba8(80, 220, 140, 230),  // green
        Color::from_rgba8(255, 200, 60, 230),  // yellow
        Color::from_rgba8(200, 120, 255, 230), // purple
        Color::from_rgba8(255, 160, 80, 230),  // orange
    ]
}

// ---------------------------------------------------------------------------
// Rasterizer3D trait — the architecture boundary
// ---------------------------------------------------------------------------

/// Abstract 3D rendering backend.
///
/// Scene and camera logic calls this trait to draw projected geometry.
/// The trait has no dependency on tiny-skia, wgpu, or any specific renderer.
///
/// # Implementors
///
/// - [`SkiaRasterizer3D`] — v1 backend using `tiny_skia::Pixmap` (ships now)
/// - `WgpuRasterizer3D` — GPU backend (Sprint 9, swaps in behind this trait)
pub trait Rasterizer3D {
    /// Draw filled circles at the projected point positions.
    fn draw_points(&mut self, points: &[ProjectedPoint], colors: &[Color], sizes: &[f32]);

    /// Draw line segments between projected point pairs.
    fn draw_line_segments(
        &mut self,
        segments: &[(ProjectedPoint, ProjectedPoint)],
        color: Color,
        width: f32,
    );

    /// Draw a text label at the given screen position.
    fn draw_text(&mut self, x: f32, y: f32, text: &str, color: Color, font_size: f32);

    /// Finalize rendering and return the RGBA pixel data.
    ///
    /// The returned `Vec<u8>` has length `width * height * 4` in RGBA order.
    fn finish(self) -> Vec<u8>;

    /// Get the canvas width.
    fn width(&self) -> u32;

    /// Get the canvas height.
    fn height(&self) -> u32;
}

// ---------------------------------------------------------------------------
// SkiaRasterizer3D — v1 backend
// ---------------------------------------------------------------------------

/// Tiny-skia based 3D rasterizer.
///
/// Uses `tiny_skia::Pixmap` for anti-aliased 2D rendering of projected 3D
/// geometry. Text rendering uses fontdue (same approach as `export.rs`).
pub struct SkiaRasterizer3D {
    pixmap: tiny_skia::Pixmap,
}

impl SkiaRasterizer3D {
    /// Create a new rasterizer with the given dimensions and background color.
    #[must_use]
    pub fn new(width: u32, height: u32, background: Color) -> Self {
        let mut pixmap = tiny_skia::Pixmap::new(width, height)
            .unwrap_or_else(|| tiny_skia::Pixmap::new(1, 1).unwrap());

        if let Some(bg) = background.to_tiny_skia() {
            pixmap.fill(bg);
        }

        Self { pixmap }
    }
}

impl Rasterizer3D for SkiaRasterizer3D {
    #[allow(clippy::cast_possible_wrap)] // pixel dimensions never exceed i32::MAX
    fn draw_points(&mut self, points: &[ProjectedPoint], colors: &[Color], sizes: &[f32]) {
        let pw = self.pixmap.width();
        let ph = self.pixmap.height();
        let data = self.pixmap.data_mut();
        let stride = pw as usize * 4;

        for pt in points {
            let color = colors
                .get(pt.original_index)
                .copied()
                .unwrap_or(Color::WHITE);
            let size = sizes.get(pt.original_index).copied().unwrap_or(3.0);
            let radius = size;

            let sr = (color.r * 255.0) as u32;
            let sg = (color.g * 255.0) as u32;
            let sb = (color.b * 255.0) as u32;
            let sa_base = color.a;

            // Depth-based size attenuation for subtle 3D effect
            let depth_factor = 1.0 - pt.depth.clamp(0.0, 1.0) * 0.3;
            let effective_radius = radius * depth_factor;

            // Bounding box in pixel coords
            let r_ceil = effective_radius.ceil() as i32 + 1;
            let cx = pt.screen_x;
            let cy = pt.screen_y;
            let x_min = ((cx - r_ceil as f32) as i32).max(0);
            let x_max = ((cx + r_ceil as f32) as i32).min(pw as i32 - 1);
            let y_min = ((cy - r_ceil as f32) as i32).max(0);
            let y_max = ((cy + r_ceil as f32) as i32).min(ph as i32 - 1);

            let r_sq = effective_radius * effective_radius;
            // Anti-aliasing zone: 1px feather
            let inner_r_sq = (effective_radius - 1.0).max(0.0);
            let inner_r_sq = inner_r_sq * inner_r_sq;

            // Border ring: last 0.8px of the radius
            let border_r_sq = (effective_radius - 0.8).max(0.0);
            let border_r_sq = border_r_sq * border_r_sq;
            let border_alpha_base = 0.3 + 0.4 * (1.0 - pt.depth.clamp(0.0, 1.0));

            for py in y_min..=y_max {
                let dy = py as f32 + 0.5 - cy;
                let dy_sq = dy * dy;
                let row_offset = py as usize * stride;

                for px in x_min..=x_max {
                    let dx = px as f32 + 0.5 - cx;
                    let dist_sq = dx * dx + dy_sq;

                    if dist_sq > r_sq {
                        continue;
                    }

                    // Coverage for anti-aliasing
                    let coverage = if dist_sq <= inner_r_sq {
                        1.0
                    } else {
                        // Smooth falloff in the feather zone
                        let dist = dist_sq.sqrt();
                        (effective_radius - dist).clamp(0.0, 1.0)
                    };

                    // Border: darken pixels in the outer ring
                    let (fr, fg, fb, fa) = if dist_sq >= border_r_sq {
                        // Blend between fill color and dark border
                        let border_mix = if effective_radius > 0.8 {
                            ((dist_sq.sqrt() - border_r_sq.sqrt()) / 0.8).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let br = ((sr as f32 * (1.0 - border_mix * 0.6)) as u32).min(255);
                        let bg_c = ((sg as f32 * (1.0 - border_mix * 0.6)) as u32).min(255);
                        let bb = ((sb as f32 * (1.0 - border_mix * 0.6)) as u32).min(255);
                        let ba =
                            sa_base * coverage * (1.0 - border_mix * (1.0 - border_alpha_base));
                        (br, bg_c, bb, ba)
                    } else {
                        (sr, sg, sb, sa_base * coverage)
                    };

                    let sa = (fa * 255.0) as u32;
                    if sa == 0 {
                        continue;
                    }
                    let inv = 255 - sa;

                    let idx = row_offset + px as usize * 4;
                    data[idx] = ((fr * sa + data[idx] as u32 * inv) / 255) as u8;
                    data[idx + 1] = ((fg * sa + data[idx + 1] as u32 * inv) / 255) as u8;
                    data[idx + 2] = ((fb * sa + data[idx + 2] as u32 * inv) / 255) as u8;
                    data[idx + 3] = (sa + data[idx + 3] as u32 * inv / 255).min(255) as u8;
                }
            }
        }
    }

    fn draw_line_segments(
        &mut self,
        segments: &[(ProjectedPoint, ProjectedPoint)],
        color: Color,
        width: f32,
    ) {
        let Some(c) = color.to_tiny_skia() else {
            return;
        };

        let paint = tiny_skia::Paint {
            anti_alias: true,
            shader: tiny_skia::Shader::SolidColor(c),
            ..Default::default()
        };

        let stroke = tiny_skia::Stroke {
            width,
            line_cap: tiny_skia::LineCap::Round,
            ..Default::default()
        };

        for (start, end) in segments {
            let mut pb = tiny_skia::PathBuilder::new();
            pb.move_to(start.screen_x, start.screen_y);
            pb.line_to(end.screen_x, end.screen_y);
            if let Some(path) = pb.finish() {
                self.pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
        }
    }

    fn draw_text(&mut self, x: f32, y: f32, text: &str, color: Color, font_size: f32) {
        stamp_text(&mut self.pixmap, x, y, text, color, font_size);
    }

    fn finish(self) -> Vec<u8> {
        self.pixmap.take()
    }

    fn width(&self) -> u32 {
        self.pixmap.width()
    }

    fn height(&self) -> u32 {
        self.pixmap.height()
    }
}

// ---------------------------------------------------------------------------
// Text rendering (fontdue) — reused from export.rs pattern
// ---------------------------------------------------------------------------

/// Embedded font data.
static FONT_DATA: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
static FONT_DATA_BOLD: &[u8] = include_bytes!("../fonts/Inter-Bold.ttf");

/// Thread-local font cache to avoid re-parsing font data every render.
pub(super) fn with_font(bold: bool, f: impl FnOnce(&fontdue::Font)) {
    use std::cell::RefCell;
    thread_local! {
        static REGULAR: RefCell<Option<fontdue::Font>> = const { RefCell::new(None) };
        static BOLD: RefCell<Option<fontdue::Font>> = const { RefCell::new(None) };
    }

    let cell = if bold { &BOLD } else { &REGULAR };
    let data = if bold { FONT_DATA_BOLD } else { FONT_DATA };

    cell.with(|opt| {
        let mut opt = opt.borrow_mut();
        if opt.is_none() {
            let font = fontdue::Font::from_bytes(data, fontdue::FontSettings::default())
                .unwrap_or_else(|e| panic!("Failed to parse font: {e}"));
            *opt = Some(font);
        }
        f(opt.as_ref().unwrap());
    });
}

/// Stamp a single text string onto a pixmap, centered at (x, y).
fn stamp_text(
    pixmap: &mut tiny_skia::Pixmap,
    x: f32,
    y: f32,
    text: &str,
    color: Color,
    font_size: f32,
) {
    with_font(false, |font| {
        // Pre-rasterize glyphs and measure width
        let mut glyphs: Vec<(fontdue::Metrics, Vec<u8>)> = Vec::with_capacity(text.len());
        let mut total_width = 0.0_f32;

        for ch in text.chars() {
            let (metrics, bitmap) = font.rasterize(ch, font_size);
            total_width += metrics.advance_width;
            glyphs.push((metrics, bitmap));
        }

        let line_metrics = font.horizontal_line_metrics(font_size);
        let ascent = line_metrics.map_or(font_size * 0.8, |m| m.ascent);

        // Center text at (x, y)
        let x_start = x - total_width / 2.0;
        let baseline_y = y + ascent * 0.5;

        let r = (color.r * 255.0) as u8;
        let g = (color.g * 255.0) as u8;
        let b = (color.b * 255.0) as u8;
        let text_alpha = color.a;

        let pw = pixmap.width();
        let ph = pixmap.height();

        let mut cursor_x = x_start;

        for (metrics, bitmap) in &glyphs {
            let gx_f = cursor_x + metrics.xmin as f32;
            let gy_f = baseline_y - metrics.height as f32 - metrics.ymin as f32;

            let data = pixmap.data_mut();

            for row in 0..metrics.height {
                for col in 0..metrics.width {
                    let coverage = bitmap[row * metrics.width + col];
                    if coverage == 0 {
                        continue;
                    }

                    let px = (gx_f + col as f32) as i32;
                    let py = (gy_f + row as f32) as i32;

                    if px < 0 || py < 0 || (px as u32) >= pw || (py as u32) >= ph {
                        continue;
                    }

                    let idx = ((py as u32) * pw + px as u32) as usize * 4;
                    let sa = ((coverage as f32 / 255.0) * text_alpha * 255.0) as u32;
                    let inv = 255 - sa;

                    data[idx] = ((r as u32 * sa + data[idx] as u32 * inv) / 255) as u8;
                    data[idx + 1] = ((g as u32 * sa + data[idx + 1] as u32 * inv) / 255) as u8;
                    data[idx + 2] = ((b as u32 * sa + data[idx + 2] as u32 * inv) / 255) as u8;
                    data[idx + 3] = (sa + data[idx + 3] as u32 * inv / 255).min(255) as u8;
                }
            }

            cursor_x += metrics.advance_width;
        }
    });
}

// ---------------------------------------------------------------------------
// Chart3D builder
// ---------------------------------------------------------------------------

/// A 3D chart builder for creating scatter plots and other 3D visualizations.
///
/// Uses the builder pattern for fluent configuration. The rendering pipeline:
/// 1. Build `Scene3D` from user data
/// 2. Configure `Camera3D` (default or user-supplied)
/// 3. Project all geometry via `PerspectiveProjection`
/// 4. Depth sort for painter's algorithm
/// 5. Rasterize via `SkiaRasterizer3D`
///
/// # Example
///
/// ```ignore
/// use scry_chart::chart3d::Chart3D;
///
/// let png = Chart3D::scatter(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0])
///     .title("Demo")
///     .render_to_png(800, 600)?;
/// std::fs::write("chart3d.png", png)?;
/// ```
#[derive(Clone, Debug)]
pub struct Chart3D {
    /// X coordinates.
    x: Vec<f64>,
    /// Y coordinates.
    y: Vec<f64>,
    /// Z coordinates.
    z: Vec<f64>,
    /// Per-point colors (optional — defaults to theme palette).
    colors: Option<Vec<Color>>,
    /// Per-point sizes (optional — defaults to 4.0).
    sizes: Option<Vec<f32>>,
    /// Chart title.
    title: Option<String>,
    /// X-axis label.
    x_label: String,
    /// Y-axis label.
    y_label: String,
    /// Z-axis label.
    z_label: String,
    /// Background color.
    background: Color,
    /// User-supplied camera (if None, a default is computed).
    camera: Option<Camera3D>,
    /// Whether to show the XZ grid plane.
    show_grid: bool,
    /// Default point size.
    point_size: f32,
}

impl Chart3D {
    /// Create a 3D scatter chart from x, y, z coordinate arrays.
    ///
    /// All three arrays must have the same length. Non-finite values are
    /// filtered during rendering.
    #[must_use]
    pub fn scatter(x: &[f64], y: &[f64], z: &[f64]) -> Self {
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            z: z.to_vec(),
            colors: None,
            sizes: None,
            title: None,
            x_label: "X".into(),
            y_label: "Y".into(),
            z_label: "Z".into(),
            background: Color::from_rgba8(15, 15, 25, 255),
            camera: None,
            show_grid: true,
            point_size: 6.0,
        }
    }

    /// Set the chart title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the X-axis label.
    #[must_use]
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = label.into();
        self
    }

    /// Set the Y-axis label.
    #[must_use]
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = label.into();
        self
    }

    /// Set the Z-axis label.
    #[must_use]
    pub fn z_label(mut self, label: impl Into<String>) -> Self {
        self.z_label = label.into();
        self
    }

    /// Set per-point colors.
    ///
    /// Length must match the data arrays. Points without a corresponding
    /// color use a default palette.
    #[must_use]
    pub fn colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = Some(colors);
        self
    }

    /// Assign per-point colors based on categorical string labels.
    ///
    /// Each unique label value is mapped to a color from the default palette.
    /// Points sharing the same label receive the same color. The palette
    /// cycles for more than 6 unique labels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use scry_chart::chart3d::Chart3D;
    ///
    /// let chart = Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
    ///     .color_by_labels(&["setosa", "versicolor", "setosa"]);
    /// ```
    #[must_use]
    pub fn color_by_labels(self, labels: &[impl AsRef<str>]) -> Self {
        let palette = default_palette();

        // Build unique label → index map (preserving insertion order)
        let mut label_map: Vec<String> = Vec::new();
        let mut indices: Vec<usize> = Vec::with_capacity(labels.len());
        for label in labels {
            let s = label.as_ref();
            let idx = label_map.iter().position(|l| l == s).unwrap_or_else(|| {
                label_map.push(s.to_string());
                label_map.len() - 1
            });
            indices.push(idx);
        }

        let colors: Vec<Color> = indices
            .iter()
            .map(|&idx| palette[idx % palette.len()])
            .collect();
        self.colors(colors)
    }

    /// Assign per-point colors based on integer class labels.
    ///
    /// Each class index is mapped to a color from the default palette.
    /// The palette cycles for more than 6 classes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use scry_chart::chart3d::Chart3D;
    ///
    /// let chart = Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
    ///     .color_by_class(&[0, 1, 0]);
    /// ```
    #[must_use]
    pub fn color_by_class(self, classes: &[usize]) -> Self {
        let palette = default_palette();
        let colors: Vec<Color> = classes
            .iter()
            .map(|&cls| palette[cls % palette.len()])
            .collect();
        self.colors(colors)
    }

    /// Set per-point sizes (screen-space radii in pixels).
    #[must_use]
    pub fn sizes(mut self, sizes: Vec<f32>) -> Self {
        self.sizes = Some(sizes);
        self
    }

    /// Set the background color.
    #[must_use]
    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set a custom camera.
    #[must_use]
    pub fn camera(mut self, camera: Camera3D) -> Self {
        self.camera = Some(camera);
        self
    }

    /// Toggle the XZ grid plane.
    #[must_use]
    pub fn grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set the default point size.
    #[must_use]
    pub fn point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }

    /// Render to raw RGBA pixel data.
    ///
    /// Returns a `Vec<u8>` of length `width * height * 4`.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or all non-finite.
    pub fn render(&self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let rasterizer = SkiaRasterizer3D::new(width, height, self.background);
        self.render_with(rasterizer)
    }

    /// Render using the GPU-accelerated wgpu backend.
    ///
    /// This provides the same output as [`render()`](Self::render) but uses
    /// GPU instanced rendering for significantly higher throughput at large
    /// point counts (50K+).
    ///
    /// Requires the `gpu` feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails (no compatible adapter)
    /// or if the data is empty/non-finite.
    #[cfg(feature = "gpu")]
    pub fn render_gpu(&self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let rasterizer = wgpu_backend::WgpuRasterizer3D::new(width, height, self.background)?;
        self.render_with(rasterizer)
    }

    /// Render using a pre-initialized GPU device (cached device + pipelines).
    ///
    /// This is the **fast path** for multi-frame GPU rendering. Use
    /// [`GpuDevice::global()`](scry_engine::gpu::GpuDevice::global) to get the
    /// shared device reference and pass it to each frame.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use scry_chart::chart3d::Chart3D;
    /// use scry_engine::gpu::GpuDevice;
    ///
    /// let gpu = GpuDevice::global().unwrap();
    /// for _ in 0..60 {
    ///     let rgba = chart.render_gpu_with_device(gpu, 1920, 1080)?;
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or all non-finite.
    #[cfg(feature = "gpu")]
    pub fn render_gpu_with_device(
        &self,
        gpu: &'static scry_engine::gpu::GpuDevice,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let rasterizer =
            wgpu_backend::WgpuRasterizer3D::with_device(gpu, width, height, self.background);
        self.render_with(rasterizer)
    }

    /// Render using a custom [`Rasterizer3D`] backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or all non-finite.
    pub fn render_with<R: Rasterizer3D>(&self, mut rasterizer: R) -> Result<Vec<u8>, String> {
        let width = rasterizer.width();
        let height = rasterizer.height();

        // --- 1. Validate and normalize data ---
        let n = self.x.len().min(self.y.len()).min(self.z.len());
        if n == 0 {
            return Err("Chart3D: no data points".into());
        }

        // Filter to finite points and normalize to [0, 1] range
        let mut finite_points: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let (xv, yv, zv) = (self.x[i], self.y[i], self.z[i]);
            if xv.is_finite() && yv.is_finite() && zv.is_finite() {
                finite_points.push((i, xv, yv, zv));
            }
        }

        if finite_points.is_empty() {
            return Err("Chart3D: all data points are non-finite".into());
        }

        // Compute data extent
        let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut z_min, mut z_max) = (f64::INFINITY, f64::NEG_INFINITY);

        for &(_, x, y, z) in &finite_points {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }

        // Add padding to avoid points on edges
        let pad = 0.05;
        let x_range = (x_max - x_min).max(1e-6);
        let y_range = (y_max - y_min).max(1e-6);
        let z_range = (z_max - z_min).max(1e-6);
        x_min -= x_range * pad;
        x_max += x_range * pad;
        y_min -= y_range * pad;
        y_max += y_range * pad;
        z_min -= z_range * pad;
        z_max += z_range * pad;

        let x_span = x_max - x_min;
        let y_span = y_max - y_min;
        let z_span = z_max - z_min;

        // Normalize to scene space [0, 1]
        let scene_points: Vec<Vec3> = finite_points
            .iter()
            .map(|&(_, x, y, z)| {
                Vec3::new(
                    ((x - x_min) / x_span) as f32,
                    ((y - y_min) / y_span) as f32,
                    ((z - z_min) / z_span) as f32,
                )
            })
            .collect();

        // --- 2. Build Scene3D ---
        let mut scene = Scene3D::new(self.background);

        // Default color palette (category colors for unlabeled data)
        let palette = default_palette();

        let point_colors: Vec<Color> = self.colors.as_ref().map_or_else(
            || vec![palette[0]; scene_points.len()],
            |user_colors| {
                finite_points
                    .iter()
                    .map(|&(orig_idx, _, _, _)| {
                        user_colors.get(orig_idx).copied().unwrap_or(palette[0])
                    })
                    .collect()
            },
        );

        let point_sizes: Vec<f32> = self.sizes.as_ref().map_or_else(
            || vec![self.point_size; scene_points.len()],
            |user_sizes| {
                finite_points
                    .iter()
                    .map(|&(orig_idx, _, _, _)| {
                        user_sizes.get(orig_idx).copied().unwrap_or(self.point_size)
                    })
                    .collect()
            },
        );

        scene.add_point_cloud(PointCloud3D {
            points: scene_points,
            colors: point_colors,
            sizes: point_sizes,
            label: None,
        });

        // Build axes with original data ranges for real-value tick labels
        let axis_config = AxisConfig3D {
            x_label: self.x_label.clone(),
            y_label: self.y_label.clone(),
            z_label: self.z_label.clone(),
            show_grid: self.show_grid,
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 1.0),
            tick_count: 5,
            data_min: Some(Vec3::new(x_min as f32, y_min as f32, z_min as f32)),
            data_max: Some(Vec3::new(x_max as f32, y_max as f32, z_max as f32)),
            ..Default::default()
        };
        scene.build_axes(&axis_config);

        // --- 3. Camera ---
        let center = Vec3::new(0.5, 0.5, 0.5);
        let cam = self.camera.clone().unwrap_or_else(|| {
            // Slightly offset azimuth for better depth perception
            Camera3D::orbiting(center, 3.0, 0.6, 0.35)
        });

        // --- 4. Projection ---
        let aspect = width as f32 / height as f32;
        let proj = PerspectiveProjection::new(cam.fov_y, aspect, cam.near, cam.far);
        let view = cam.view_matrix();

        // Pre-compute the VP matrix once for all geometry
        let proj_mat = proj.projection_matrix();
        let vp = mat4_mul(&proj_mat, &view);

        // --- 5. Project all geometry ---

        // Project line segments using precomputed VP
        let mut projected_segments: Vec<(ProjectedPoint, ProjectedPoint, Color, f32)> = Vec::new();
        for seg in &scene.line_segments {
            let a_clip =
                projection::mat4_mul_vec4(&vp, [seg.start.x, seg.start.y, seg.start.z, 1.0]);
            let b_clip = projection::mat4_mul_vec4(&vp, [seg.end.x, seg.end.y, seg.end.z, 1.0]);

            if a_clip[3] <= 0.0 || b_clip[3] <= 0.0 {
                continue;
            }

            let w_f = width as f32;
            let h_f = height as f32;
            let a_inv = 1.0 / a_clip[3];
            let b_inv = 1.0 / b_clip[3];

            let a = ProjectedPoint {
                screen_x: (a_clip[0] * a_inv + 1.0) * 0.5 * w_f,
                screen_y: (1.0 - a_clip[1] * a_inv) * 0.5 * h_f,
                depth: a_clip[2] * a_inv,
                original_index: 0,
            };
            let b = ProjectedPoint {
                screen_x: (b_clip[0] * b_inv + 1.0) * 0.5 * w_f,
                screen_y: (1.0 - b_clip[1] * b_inv) * 0.5 * h_f,
                depth: b_clip[2] * b_inv,
                original_index: 0,
            };
            projected_segments.push((a, b, seg.color, seg.width));
        }

        // Sort segments by depth
        projected_segments.sort_unstable_by(|a, b| {
            let da = (a.0.depth + a.1.depth) * 0.5;
            let db = (b.0.depth + b.1.depth) * 0.5;
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Draw segments grouped by (color, width) for batching
        let mut seg_idx = 0;
        while seg_idx < projected_segments.len() {
            let (_, _, color, w) = projected_segments[seg_idx];
            let batch_start = seg_idx;
            while seg_idx < projected_segments.len()
                && projected_segments[seg_idx].2 == color
                && (projected_segments[seg_idx].3 - w).abs() < f32::EPSILON
            {
                seg_idx += 1;
            }
            let batch: Vec<(ProjectedPoint, ProjectedPoint)> = projected_segments
                [batch_start..seg_idx]
                .iter()
                .map(|(a, b, _, _)| (*a, *b))
                .collect();
            rasterizer.draw_line_segments(&batch, color, w);
        }

        // Project point clouds using precomputed VP
        for cloud in &scene.point_clouds {
            let mut projected_pts = project_batch_precomputed(&vp, &cloud.points, width, height);
            depth_sort(&mut projected_pts);

            rasterizer.draw_points(&projected_pts, &cloud.colors, &cloud.sizes);
        }

        // Project and draw labels
        for label in &scene.labels {
            if let Some(p) = proj.project(label.position, &view, width, height, 0) {
                rasterizer.draw_text(
                    p.screen_x,
                    p.screen_y,
                    &label.text,
                    label.color,
                    label.font_size,
                );
            }
        }

        // --- 6. Title ---
        if let Some(ref title) = self.title {
            let title_color = Color::from_rgba8(230, 230, 240, 255);
            rasterizer.draw_text(width as f32 / 2.0, 20.0, title, title_color, 18.0);
        }

        Ok(rasterizer.finish())
    }

    /// Render to a PNG byte buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering or PNG encoding fails.
    pub fn render_to_png(&self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let rgba = self.render(width, height)?;
        let pixmap =
            tiny_skia::Pixmap::from_vec(rgba, tiny_skia::IntSize::from_wh(width, height).unwrap())
                .ok_or("failed to create pixmap from RGBA data")?;
        pixmap
            .encode_png()
            .map_err(|e| format!("PNG encoding failed: {e}"))
    }

    /// Render and save to a PNG file.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering, PNG encoding, or file I/O fails.
    pub fn save_png(
        &self,
        width: u32,
        height: u32,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let data = self.render_to_png(width, height)?;
        std::fs::write(path.as_ref(), data)
            .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
    }

    /// Render to a [`PixelCanvas`] scene ready for terminal display.
    ///
    /// This bridges the 3D rendering pipeline into scry-engine's display
    /// pipeline. The resulting canvas can be passed to [`PixelCanvasWidget`]
    /// or rasterized and transmitted directly via a [`ProtocolBackend`].
    ///
    /// [`PixelCanvasWidget`]: scry_engine::prelude::PixelCanvasWidget
    /// [`ProtocolBackend`]: scry_engine::transport::ProtocolBackend
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or all non-finite.
    pub fn render_to_canvas(
        &self,
        width: u32,
        height: u32,
    ) -> Result<scry_engine::scene::PixelCanvas, String> {
        let rgba = self.render(width, height)?;
        let image = scry_engine::scene::ImageData::new(width, height, rgba);
        Ok(scry_engine::scene::PixelCanvas::new(width, height)
            .image(image, 0.0, 0.0)
            .done())
    }

    /// Render to a [`PixelCanvas`] using the GPU-accelerated wgpu backend.
    ///
    /// This is the GPU equivalent of [`render_to_canvas()`](Self::render_to_canvas).
    /// Uses a shared [`GpuDevice`](scry_engine::gpu::GpuDevice) to skip
    /// the expensive ~100ms device initialization on each frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or all non-finite.
    #[cfg(feature = "gpu")]
    pub fn render_gpu_to_canvas_with_device(
        &self,
        gpu: &'static scry_engine::gpu::GpuDevice,
        width: u32,
        height: u32,
    ) -> Result<scry_engine::scene::PixelCanvas, String> {
        let rgba = self.render_gpu_with_device(gpu, width, height)?;
        let image = scry_engine::scene::ImageData::new(width, height, rgba);
        Ok(scry_engine::scene::PixelCanvas::new(width, height)
            .image(image, 0.0, 0.0)
            .done())
    }

    /// Render a [`Surface3D`] as a lit 3D surface using the SDF ray marcher.
    ///
    /// This produces a Phong-shaded, shadow-casting surface plot — much richer
    /// than the wireframe scatter plot from [`render()`](Self::render). The
    /// camera is either the user-supplied one or a default orbiting view.
    ///
    /// Requires the `sdf` feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if SDF rendering fails.
    #[cfg(feature = "sdf")]
    pub fn render_surface_lit(
        &self,
        surface: &scene::Surface3D,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let center = Vec3::new(0.5, 0.5, 0.5);
        let cam = self.camera.clone().unwrap_or_else(|| {
            Camera3D::orbiting(center, 3.0, 0.6, 0.35)
        });

        let sdf_scene = sdf_surface::surface_to_sdf_scene(surface, &cam);
        let pixmap = scry_engine::sdf::SdfRenderer::render_to_pixmap(&sdf_scene, width, height, 0.0)
            .map_err(|e| format!("SDF render failed: {e}"))?;

        Ok(pixmap.take())
    }

    /// Render and display interactively in the terminal.
    ///
    /// This is the **inline** rendering mode — it streams the chart directly
    /// to stdout via the detected graphics protocol (Kitty/Sixel/halfblock).
    /// No ratatui dependency is needed (but does require `crossterm` via the
    /// `widget` feature).
    ///
    /// Keyboard controls:
    /// - **Arrow keys** / **WASD** — rotate the camera
    /// - **+** / **-** — zoom in/out
    /// - **Q** / **Esc** — quit
    ///
    /// The terminal enters scoped raw mode for key capture. On exit, raw mode
    /// is disabled and the image is cleaned up.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering or terminal interaction fails.
    #[cfg(feature = "widget")]
    pub fn show(&self) -> Result<(), String> {
        use crossterm::event::{self, Event, KeyCode, KeyEventKind};
        use crossterm::terminal::{self};
        use crossterm::ExecutableCommand;
        use scry_engine::prelude::{Picker, ProtocolKind};
        use scry_engine::rasterize::Rasterizer;
        use scry_engine::transport::backend::TerminalPosition;
        use scry_engine::transport::{self, ProtocolBackend};
        use std::io::Write;

        let picker = Picker::detect();
        let font = picker.font_size();

        // Get terminal dimensions in pixels
        let (cols, rows) =
            terminal::size().map_err(|e| format!("failed to get terminal size: {e}"))?;
        // Leave 2 rows for the status bar
        let display_rows = rows.saturating_sub(2);
        let pixel_w = u32::from(cols) * u32::from(font.width);
        let pixel_h = u32::from(display_rows) * u32::from(font.height);

        if pixel_w == 0 || pixel_h == 0 {
            return Err("terminal too small for rendering".into());
        }

        // Create protocol backend
        let mut backend: Box<dyn ProtocolBackend> = match picker.protocol() {
            ProtocolKind::Kitty => {
                Box::new(transport::kitty::KittyBackend::new(picker.font_size()))
            }
            _ => Box::new(transport::halfblock::HalfblockBackend::new()),
        };

        let position = TerminalPosition::new(0, 0, cols, display_rows);

        // Initial camera angles
        let mut angle_y: f64 = 0.45;
        let mut angle_x: f64 = 0.35;
        let mut distance: f64 = 2.5;

        // Enter alternate screen + raw mode so terminal text is hidden
        let mut stdout = std::io::stdout();
        stdout
            .execute(crossterm::terminal::EnterAlternateScreen)
            .map_err(|e| format!("failed to enter alternate screen: {e}"))?;
        terminal::enable_raw_mode().map_err(|e| format!("failed to enable raw mode: {e}"))?;
        // Hide cursor for a cleaner look
        let _ = stdout.execute(crossterm::cursor::Hide);

        // Render initial frame
        let cam = Camera3D::orbiting(
            Vec3::new(0.5, 0.5, 0.5),
            distance as f32,
            angle_y as f32,
            angle_x as f32,
        );
        let chart = self.clone().camera(cam);
        let canvas = chart.render_to_canvas(pixel_w, pixel_h)?;
        let pixmap =
            Rasterizer::rasterize(&canvas).map_err(|e| format!("rasterize failed: {e}"))?;
        let mut handle = backend
            .transmit(&pixmap, position, -1)
            .map_err(|e| format!("transmit failed: {e}"))?;

        // Status bar
        let status = " ←→↑↓/WASD rotate | +/- zoom | Q quit";
        let _ = stdout.execute(crossterm::cursor::MoveTo(0, display_rows));
        let _ = write!(stdout, "\x1b[1;36m{status}\x1b[0m");
        let _ = stdout.flush();

        loop {
            if event::poll(std::time::Duration::from_millis(50))
                .map_err(|e| format!("poll failed: {e}"))?
            {
                if let Event::Key(key) = event::read().map_err(|e| format!("read failed: {e}"))? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    let mut changed = true;
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Left | KeyCode::Char('a') => angle_y -= 0.12,
                        KeyCode::Right | KeyCode::Char('d') => angle_y += 0.12,
                        KeyCode::Up | KeyCode::Char('w') => angle_x -= 0.12,
                        KeyCode::Down | KeyCode::Char('s') => angle_x += 0.12,
                        KeyCode::Char('+' | '=') => {
                            distance = (distance - 0.2).max(0.5);
                        }
                        KeyCode::Char('-') => {
                            distance = (distance + 0.2).min(10.0);
                        }
                        _ => changed = false,
                    }

                    if changed {
                        let cam = Camera3D::orbiting(
                            Vec3::new(0.5, 0.5, 0.5),
                            distance as f32,
                            angle_y as f32,
                            angle_x as f32,
                        );
                        let chart = self.clone().camera(cam);
                        let canvas = chart.render_to_canvas(pixel_w, pixel_h)?;
                        let pixmap = Rasterizer::rasterize(&canvas)
                            .map_err(|e| format!("rasterize failed: {e}"))?;
                        handle = backend
                            .replace(&handle, &pixmap, position, -1)
                            .map_err(|e| format!("replace failed: {e}"))?;
                        let _ = stdout.flush();
                    }
                }
            }
        }

        // Cleanup: restore terminal state
        let _ = backend.remove(&handle);
        let _ = stdout.execute(crossterm::cursor::Show);
        let _ = terminal::disable_raw_mode();
        let _ = stdout.execute(crossterm::terminal::LeaveAlternateScreen);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart3d_scatter_builder() {
        let chart = Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
            .title("Test")
            .x_label("A")
            .y_label("B")
            .z_label("C");

        assert_eq!(chart.title.as_deref(), Some("Test"));
        assert_eq!(chart.x_label, "A");
        assert_eq!(chart.x.len(), 3);
    }

    #[test]
    fn chart3d_render_produces_rgba() {
        let rgba =
            Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]).render(200, 150);

        assert!(rgba.is_ok(), "render should succeed: {:?}", rgba.err());
        let data = rgba.unwrap();
        assert_eq!(data.len(), 200 * 150 * 4, "RGBA should be width*height*4");
    }

    #[test]
    fn chart3d_render_to_png() {
        let png = Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
            .title("PNG Test")
            .render_to_png(200, 150);

        assert!(png.is_ok(), "PNG render should succeed: {:?}", png.err());
        let data = png.unwrap();
        // PNG magic bytes
        assert_eq!(&data[..4], &[0x89, 0x50, 0x4E, 0x47], "should be valid PNG");
    }

    #[test]
    fn chart3d_empty_data_errors() {
        let result = Chart3D::scatter(&[], &[], &[]).render(200, 150);
        assert!(result.is_err(), "empty data should error");
    }

    #[test]
    fn chart3d_all_nan_errors() {
        let result = Chart3D::scatter(
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
        )
        .render(200, 150);
        assert!(result.is_err(), "all NaN should error");
    }

    #[test]
    fn chart3d_with_custom_colors() {
        let colors = vec![Color::RED, Color::BLUE, Color::GREEN];
        let result = Chart3D::scatter(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0])
            .colors(colors)
            .render(200, 150);
        assert!(result.is_ok());
    }

    #[test]
    fn chart3d_with_custom_camera() {
        let cam = Camera3D::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(0.5, 0.5, 0.5), Vec3::Y);
        let result = Chart3D::scatter(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0])
            .camera(cam)
            .render(200, 150);
        assert!(result.is_ok());
    }

    #[test]
    fn chart3d_no_grid() {
        let result = Chart3D::scatter(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0])
            .grid(false)
            .render(200, 150);
        assert!(result.is_ok());
    }

    #[test]
    fn skia_rasterizer_basic() {
        let mut rast = SkiaRasterizer3D::new(100, 100, Color::BLACK);
        rast.draw_points(
            &[ProjectedPoint {
                screen_x: 50.0,
                screen_y: 50.0,
                depth: 0.5,
                original_index: 0,
            }],
            &[Color::RED],
            &[5.0],
        );
        let data = rast.finish();
        assert_eq!(data.len(), 100 * 100 * 4);
        // At least one non-black pixel should exist
        let has_color = data.chunks(4).any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
        assert!(has_color, "rasterizer should produce visible output");
    }

    #[test]
    fn chart3d_color_by_labels() {
        let labels = ["setosa", "versicolor", "setosa", "virginica"];
        let chart = Chart3D::scatter(
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        )
        .color_by_labels(&labels);

        let colors = chart.colors.as_ref().expect("colors should be set");
        assert_eq!(colors.len(), 4);
        // Same label → same color
        assert_eq!(colors[0], colors[2], "setosa should have same color");
        // Different labels → different colors
        assert_ne!(colors[0], colors[1], "different species should differ");

        // Should render successfully
        let result = chart.render(200, 150);
        assert!(result.is_ok());
    }

    #[test]
    fn chart3d_color_by_class() {
        let classes = [0, 1, 2, 0, 1];
        let chart = Chart3D::scatter(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[6.0, 7.0, 8.0, 9.0, 10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0],
        )
        .color_by_class(&classes);

        let colors = chart.colors.as_ref().expect("colors should be set");
        assert_eq!(colors.len(), 5);
        // Same class → same color
        assert_eq!(colors[0], colors[3], "class 0 should match");
        assert_eq!(colors[1], colors[4], "class 1 should match");
        // Different classes → different colors
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);

        let result = chart.render(200, 150);
        assert!(result.is_ok());
    }

    #[test]
    fn chart3d_default_palette_has_six_colors() {
        let p = default_palette();
        assert_eq!(p.len(), 6);
        // All colors should be distinct
        for i in 0..p.len() {
            for j in (i + 1)..p.len() {
                assert_ne!(p[i], p[j], "palette colors {i} and {j} should differ");
            }
        }
    }
}
