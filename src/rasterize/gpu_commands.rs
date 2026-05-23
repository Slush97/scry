// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU command processing and batch accumulation for the 2D rasterizer.
//!
//! Contains the deferred draw batch structs (`ShapeBatch`, `LineBatch`,
//! `MeshBatch`, `GradientDraw`, `ImageOverlay`), command dispatch
//! (`process_command`, `process_commands`), CPU fallback, and style
//! extraction helpers.
//!
//! The standalone [`collect_gpu_batches`] function provides the same GPU
//! batch collection without requiring a `WgpuRasterizer` — used by the
//! compositor to render directly into its render pass.

use super::skia::Rasterizer;
use super::tessellate;
use super::wgpu_context::{
    GpuGradientStop, GradientUniforms, LineVertex, MeshVertex, ShapeInstance,
};
use crate::scene::command::DrawCommand;
use crate::scene::style::{FillStyle, GradientKind};
use tiny_skia::Pixmap;

// ---------------------------------------------------------------------------
// Deferred draw batches
// ---------------------------------------------------------------------------

pub(super) struct ShapeBatch {
    pub(super) instances: Vec<ShapeInstance>,
}

pub(super) struct LineBatch {
    pub(super) vertices: Vec<LineVertex>,
}

pub(super) struct MeshBatch {
    pub(super) vertices: Vec<MeshVertex>,
}

pub(super) struct GradientDraw {
    pub(super) uniforms: GradientUniforms,
}

/// CPU-rasterized overlay to blit onto the GPU output.
pub(super) struct ImageOverlay {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) rgba: Vec<u8>,
    pub(super) width: u32,
    pub(super) height: u32,
}

// ---------------------------------------------------------------------------
// Standalone GPU batch collection (no WgpuRasterizer dependency)
// ---------------------------------------------------------------------------

/// Collected GPU batches ready for direct render pass submission.
///
/// Contains flattened vertex/instance data for all GPU-compatible commands.
/// Commands that require CPU fallback (images, text, SDF) are silently
/// skipped — the compositor's existing overlay blit path handles those.
pub struct GpuBatches {
    /// Shape instances (circles, rects, ellipses).
    pub shapes: Vec<ShapeInstance>,
    /// Line vertices (6 per segment).
    pub lines: Vec<LineVertex>,
    /// Tessellated mesh vertices (paths, arcs, polygons).
    pub meshes: Vec<MeshVertex>,
    /// Gradient draw uniforms (one per gradient command).
    pub gradients: Vec<GradientUniforms>,
}

/// Collect GPU-compatible draw batches from a canvas scene.
///
/// This is the standalone entry point for direct render pass integration.
/// It walks the canvas display list and sorts commands into GPU batches
/// without creating a `WgpuRasterizer` or any GPU resources.
///
/// Commands that would need CPU fallback (images, text, complex groups)
/// are silently skipped.
#[allow(clippy::cast_precision_loss)]
pub fn collect_gpu_batches(
    canvas: &crate::scene::PixelCanvas,
    viewport_width: u32,
    viewport_height: u32,
) -> GpuBatches {
    let mut batches = GpuBatches {
        shapes: Vec::new(),
        lines: Vec::new(),
        meshes: Vec::new(),
        gradients: Vec::new(),
    };
    let mut mesh_groups: Vec<Vec<MeshVertex>> = Vec::new();

    for cmd in canvas.commands() {
        collect_command(
            cmd,
            viewport_width,
            viewport_height,
            &mut batches.shapes,
            &mut batches.lines,
            &mut mesh_groups,
            &mut batches.gradients,
        );
    }

    if !mesh_groups.is_empty() {
        batches.meshes = mesh_groups.into_iter().flatten().collect();
    }

    batches
}

/// Process a single command for standalone batch collection.
///
/// GPU-incompatible commands (images, text, SDF, complex groups) are
/// silently skipped.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn collect_command(
    cmd: &DrawCommand,
    viewport_width: u32,
    viewport_height: u32,
    shapes: &mut Vec<ShapeInstance>,
    lines: &mut Vec<LineVertex>,
    meshes: &mut Vec<Vec<MeshVertex>>,
    gradients: &mut Vec<GradientUniforms>,
) {
    match cmd {
        DrawCommand::Clear { .. } => {}

        DrawCommand::Circle {
            cx,
            cy,
            radius,
            style,
        } => {
            let (fill_color, stroke_color, stroke_width) = extract_style(style);
            shapes.push(ShapeInstance {
                pos: [*cx, *cy],
                size: [*radius, *radius, 0.0, 0.0],
                fill_color,
                stroke_color,
                stroke_width,
                shape_type: 0,
            });
        }

        DrawCommand::Rectangle {
            rect,
            corner_radius,
            style,
        } => {
            let (fill_color, stroke_color, stroke_width) = extract_style(style);
            shapes.push(ShapeInstance {
                pos: [rect.x, rect.y],
                size: [rect.width, rect.height, *corner_radius, 0.0],
                fill_color,
                stroke_color,
                stroke_width,
                shape_type: 1,
            });
        }

        DrawCommand::Ellipse {
            cx,
            cy,
            rx,
            ry,
            rotation,
            style,
        } => {
            let (fill_color, stroke_color, stroke_width) = extract_style(style);
            shapes.push(ShapeInstance {
                pos: [*cx, *cy],
                size: [*rx, *ry, *rotation, 0.0],
                fill_color,
                stroke_color,
                stroke_width,
                shape_type: 2,
            });
        }

        DrawCommand::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
            ..
        } => {
            let color = [
                stroke.color.r,
                stroke.color.g,
                stroke.color.b,
                stroke.color.a,
            ];
            emit_line_segment(lines, *x1, *y1, *x2, *y2, stroke.width, color);
        }

        DrawCommand::Polyline {
            points,
            style,
            closed,
        } => {
            if points.len() < 2 {
                return;
            }
            if let Some(stroke) = &style.stroke {
                let color = [
                    stroke.color.r,
                    stroke.color.g,
                    stroke.color.b,
                    stroke.color.a,
                ];
                for window in points.windows(2) {
                    emit_line_segment(
                        lines,
                        window[0].0,
                        window[0].1,
                        window[1].0,
                        window[1].1,
                        stroke.width,
                        color,
                    );
                }
                if *closed && points.len() > 2 {
                    let first = points[0];
                    let last = points[points.len() - 1];
                    emit_line_segment(lines, last.0, last.1, first.0, first.1, stroke.width, color);
                }
            }
            if let Some(color) = solid_fill_color(style) {
                let verts = tessellate::tessellate_polygon(points, color);
                if !verts.is_empty() {
                    meshes.push(verts);
                }
            }
        }

        DrawCommand::Gradient { rect, gradient, .. } => {
            let mut stops = [GpuGradientStop {
                color: [0.0; 4],
                position: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            }; 8];
            let num_stops = gradient.stops.len().min(8);
            for (i, s) in gradient.stops.iter().take(8).enumerate() {
                stops[i] = GpuGradientStop {
                    color: [s.color.r, s.color.g, s.color.b, s.color.a],
                    position: s.position,
                    _pad1: 0.0,
                    _pad2: 0.0,
                    _pad3: 0.0,
                };
            }
            let (grad_start, grad_end, grad_type) = match &gradient.kind {
                GradientKind::Linear { start, end } => ([start.x, start.y], [end.x, end.y], 0.0),
                GradientKind::Radial { center, radius } => {
                    ([center.x, center.y], [*radius, 0.0], 1.0)
                }
            };
            gradients.push(GradientUniforms {
                viewport: [viewport_width as f32, viewport_height as f32],
                rect_pos: [rect.x, rect.y],
                rect_size: [rect.width, rect.height],
                grad_start,
                grad_end,
                grad_type,
                num_stops: num_stops as f32,
                _pad: [0.0, 0.0],
                _pre_stops_pad: [0.0, 0.0],
                stops,
            });
        }

        DrawCommand::Path { path, style } => {
            if let Some(color) = solid_fill_color(style) {
                let verts = tessellate::tessellate_path(path.path(), color);
                if !verts.is_empty() {
                    meshes.push(verts);
                }
            }
            // Gradient fills / strokes are skipped in direct path
        }

        DrawCommand::Arc {
            cx,
            cy,
            radius,
            start_angle,
            sweep_angle,
            style,
        } => {
            if let Some(color) = solid_fill_color(style) {
                let verts = tessellate::tessellate_arc(
                    *cx,
                    *cy,
                    *radius,
                    *start_angle,
                    *sweep_angle,
                    color,
                );
                if !verts.is_empty() {
                    meshes.push(verts);
                }
            }
        }

        DrawCommand::Group {
            commands,
            opacity,
            blend_mode,
            clip,
            transform,
        } => {
            let needs_compositing = *opacity < 1.0
                || clip.is_some()
                || *blend_mode != crate::scene::style::BlendMode::SrcOver;
            if !needs_compositing && *transform == crate::scene::style::Transform::IDENTITY {
                for child in commands {
                    collect_command(
                        child,
                        viewport_width,
                        viewport_height,
                        shapes,
                        lines,
                        meshes,
                        gradients,
                    );
                }
            }
            // Complex groups are silently skipped in direct path
        }

        // Image, Text, SDF — skip in direct render (no CPU fallback available)
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Command processing (WgpuRasterizer path — used by rasterize-to-pixmap)
// ---------------------------------------------------------------------------

/// Walk canvas commands and batch them for GPU submission.
pub(super) fn process_commands(
    rast: &mut super::wgpu::WgpuRasterizer<'_>,
    canvas: &crate::scene::PixelCanvas,
) {
    let mut shapes = Vec::new();
    let mut lines = Vec::new();
    let mut meshes: Vec<Vec<MeshVertex>> = Vec::new();

    for cmd in canvas.commands() {
        process_command(rast, cmd, &mut shapes, &mut lines, &mut meshes);
    }

    if !shapes.is_empty() {
        rast.shape_batches.push(ShapeBatch { instances: shapes });
    }
    if !lines.is_empty() {
        rast.line_batches.push(LineBatch { vertices: lines });
    }
    if !meshes.is_empty() {
        let vertices = meshes.into_iter().flatten().collect();
        rast.mesh_batches.push(MeshBatch { vertices });
    }
}

/// Process a single draw command, accumulating into shape/line batches
/// or creating gradient draws / CPU fallback overlays.
#[allow(clippy::too_many_lines)]
pub(super) fn process_command(
    rast: &mut super::wgpu::WgpuRasterizer<'_>,
    cmd: &DrawCommand,
    shapes: &mut Vec<ShapeInstance>,
    lines: &mut Vec<LineVertex>,
    meshes: &mut Vec<Vec<MeshVertex>>,
) {
    match cmd {
        DrawCommand::Clear { .. } => {
            // Handled by render pass clear color
        }

        DrawCommand::Circle {
            cx,
            cy,
            radius,
            style,
        } => {
            let (fill_color, stroke_color, stroke_width) = extract_style(style);
            shapes.push(ShapeInstance {
                pos: [*cx, *cy],
                size: [*radius, *radius, 0.0, 0.0],
                fill_color,
                stroke_color,
                stroke_width,
                shape_type: 0, // circle
            });
        }

        DrawCommand::Rectangle {
            rect,
            corner_radius,
            style,
        } => {
            let (fill_color, stroke_color, stroke_width) = extract_style(style);
            shapes.push(ShapeInstance {
                pos: [rect.x, rect.y],
                size: [rect.width, rect.height, *corner_radius, 0.0],
                fill_color,
                stroke_color,
                stroke_width,
                shape_type: 1, // rect
            });
        }

        DrawCommand::Ellipse {
            cx,
            cy,
            rx,
            ry,
            rotation,
            style,
        } => {
            let (fill_color, stroke_color, stroke_width) = extract_style(style);
            shapes.push(ShapeInstance {
                pos: [*cx, *cy],
                size: [*rx, *ry, *rotation, 0.0],
                fill_color,
                stroke_color,
                stroke_width,
                shape_type: 2, // ellipse
            });
        }

        DrawCommand::Line {
            x1,
            y1,
            x2,
            y2,
            stroke,
            ..
        } => {
            let color = [
                stroke.color.r,
                stroke.color.g,
                stroke.color.b,
                stroke.color.a,
            ];
            emit_line_segment(lines, *x1, *y1, *x2, *y2, stroke.width, color);
        }

        DrawCommand::Polyline {
            points,
            style,
            closed,
        } => {
            if points.len() < 2 {
                return;
            }

            // Stroke the polyline segments
            if let Some(stroke) = &style.stroke {
                let color = [
                    stroke.color.r,
                    stroke.color.g,
                    stroke.color.b,
                    stroke.color.a,
                ];
                let width = stroke.width;

                for window in points.windows(2) {
                    emit_line_segment(
                        lines,
                        window[0].0,
                        window[0].1,
                        window[1].0,
                        window[1].1,
                        width,
                        color,
                    );
                }
                if *closed && points.len() > 2 {
                    let first = points[0];
                    let last = points[points.len() - 1];
                    emit_line_segment(lines, last.0, last.1, first.0, first.1, width, color);
                }
            }

            // Fill the polygon via GPU tessellation if solid, else CPU fallback
            if let Some(color) = solid_fill_color(style) {
                let verts = tessellate::tessellate_polygon(points, color);
                if !verts.is_empty() {
                    meshes.push(verts);
                }
            } else if style.fill.is_some() {
                cpu_fallback_command(rast, cmd);
            }
        }

        DrawCommand::Gradient { rect, gradient, .. } => {
            let mut stops = [GpuGradientStop {
                color: [0.0; 4],
                position: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            }; 8];

            let num_stops = gradient.stops.len().min(8);
            for (i, s) in gradient.stops.iter().take(8).enumerate() {
                stops[i] = GpuGradientStop {
                    color: [s.color.r, s.color.g, s.color.b, s.color.a],
                    position: s.position,
                    _pad1: 0.0,
                    _pad2: 0.0,
                    _pad3: 0.0,
                };
            }

            let (grad_start, grad_end, grad_type) = match &gradient.kind {
                GradientKind::Linear { start, end } => ([start.x, start.y], [end.x, end.y], 0.0),
                GradientKind::Radial { center, radius } => {
                    ([center.x, center.y], [*radius, 0.0], 1.0)
                }
            };

            #[allow(clippy::cast_precision_loss)]
            rast.gradient_draws.push(GradientDraw {
                uniforms: GradientUniforms {
                    viewport: [rast.width as f32, rast.height as f32],
                    rect_pos: [rect.x, rect.y],
                    rect_size: [rect.width, rect.height],
                    grad_start,
                    grad_end,
                    grad_type,
                    num_stops: num_stops as f32,
                    _pad: [0.0, 0.0],
                    _pre_stops_pad: [0.0, 0.0],
                    stops,
                },
            });
        }

        DrawCommand::Image { image, x, y, .. } => {
            // Images are always CPU-side data → overlay
            rast.image_overlays.push(ImageOverlay {
                x: *x as i32,
                y: *y as i32,
                rgba: image.data().to_vec(),
                width: image.width(),
                height: image.height(),
            });
        }

        DrawCommand::Path { path, style } => {
            if let Some(color) = solid_fill_color(style) {
                let verts = tessellate::tessellate_path(path.path(), color);
                if !verts.is_empty() {
                    meshes.push(verts);
                }
            }
            // Gradient fills / strokes still fall back to CPU
            if has_non_solid_fill(style) || style.stroke.is_some() {
                cpu_fallback_command(rast, cmd);
            }
        }

        DrawCommand::Arc {
            cx,
            cy,
            radius,
            start_angle,
            sweep_angle,
            style,
        } => {
            if let Some(color) = solid_fill_color(style) {
                let verts = tessellate::tessellate_arc(
                    *cx,
                    *cy,
                    *radius,
                    *start_angle,
                    *sweep_angle,
                    color,
                );
                if !verts.is_empty() {
                    meshes.push(verts);
                }
            }
            if has_non_solid_fill(style) || style.stroke.is_some() {
                cpu_fallback_command(rast, cmd);
            }
        }

        #[cfg(feature = "text")]
        DrawCommand::Text { .. } => {
            cpu_fallback_command(rast, cmd);
        }

        #[cfg(feature = "sdf")]
        DrawCommand::Sdf3D { .. } => {
            cpu_fallback_command(rast, cmd);
        }

        DrawCommand::Group {
            commands,
            opacity,
            blend_mode,
            clip,
            transform,
        } => {
            let needs_compositing = *opacity < 1.0
                || clip.is_some()
                || *blend_mode != crate::scene::style::BlendMode::SrcOver;

            if needs_compositing || *transform != crate::scene::style::Transform::IDENTITY {
                // Complex group: fall back to CPU
                cpu_fallback_command(rast, cmd);
            } else {
                // Simple group: recurse
                for child in commands {
                    process_command(rast, child, shapes, lines, meshes);
                }
            }
        }
    }
}

/// Rasterize a command via CPU (tiny-skia) and add as image overlay.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn cpu_fallback_command(rast: &mut super::wgpu::WgpuRasterizer<'_>, cmd: &DrawCommand) {
    // Record the fallback for diagnostics
    rast.gpu_fallbacks.push(super::backend::GpuFallbackWarning {
        command_type: cmd.type_name(),
        reason: "command not supported by GPU pipeline",
    });

    // Estimate bounds to create a tight temp pixmap
    let (min_x, min_y, max_x, max_y) = Rasterizer::estimate_command_bounds(cmd);
    if min_x >= max_x || min_y >= max_y {
        return;
    }

    let margin = 4.0;
    let x0 = (min_x - margin).max(0.0).floor();
    let y0 = (min_y - margin).max(0.0).floor();
    let x1 = (max_x + margin).min(rast.width as f32).ceil();
    let y1 = (max_y + margin).min(rast.height as f32).ceil();
    let w = (x1 - x0) as u32;
    let h = (y1 - y0) as u32;
    if w == 0 || h == 0 {
        return;
    }

    let Some(mut pixmap) = Pixmap::new(w, h) else {
        return;
    };

    // Offset transform so command renders at (0,0) in the temp pixmap
    let offset = tiny_skia::Transform::from_translate(-x0, -y0);
    let mut pool = Vec::new();
    let mut grad_cache = std::collections::HashMap::new();
    Rasterizer::render_command(
        &mut pixmap,
        cmd,
        offset,
        &mut pool,
        &mut grad_cache,
        &mut None,
        0,
    );

    rast.image_overlays.push(ImageOverlay {
        x: x0 as i32,
        y: y0 as i32,
        rgba: pixmap.data().to_vec(),
        width: w,
        height: h,
    });
}

// ---------------------------------------------------------------------------
// Style helpers
// ---------------------------------------------------------------------------

/// Extract fill/stroke color and stroke width from a `ShapeStyle`.
pub(super) fn extract_style(style: &crate::scene::style::ShapeStyle) -> ([f32; 4], [f32; 4], f32) {
    let fill_color = match &style.fill {
        Some(FillStyle::Solid(c)) => [c.r, c.g, c.b, c.a],
        Some(FillStyle::LinearGradient(_) | FillStyle::RadialGradient(_)) => {
            // Gradient fills on shapes: use transparent fill, handle separately
            // (would need a more complex shader — for now, CPU fallback)
            [0.0, 0.0, 0.0, 0.0]
        }
        None => [0.0, 0.0, 0.0, 0.0],
    };

    let (stroke_color, stroke_width) = match &style.stroke {
        Some(s) => ([s.color.r, s.color.g, s.color.b, s.color.a], s.width),
        None => ([0.0, 0.0, 0.0, 0.0], 0.0),
    };

    (fill_color, stroke_color, stroke_width)
}

/// Extract the solid fill color from a shape style, applying opacity.
pub(super) fn solid_fill_color(style: &crate::scene::style::ShapeStyle) -> Option<[f32; 4]> {
    match &style.fill {
        Some(FillStyle::Solid(c)) => Some([c.r, c.g, c.b, c.a * style.opacity]),
        _ => None,
    }
}

/// Returns `true` if the style has a non-solid (gradient) fill.
pub(super) fn has_non_solid_fill(style: &crate::scene::style::ShapeStyle) -> bool {
    matches!(
        &style.fill,
        Some(FillStyle::LinearGradient(_) | FillStyle::RadialGradient(_))
    )
}

/// Emit 6 vertices (2 triangles) for one line segment.
pub(super) fn emit_line_segment(
    vertices: &mut Vec<LineVertex>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    width: f32,
    color: [f32; 4],
) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = dx.hypot(dy);
    if len < 1e-6 {
        return;
    }

    let half_width = width * 0.5;
    let nx = -dy / len;
    let ny = dx / len;
    let normal = [nx, ny];

    let s = [x1, y1];
    let e = [x2, y2];

    // Triangle 1: p0, p1, p2
    vertices.push(LineVertex {
        position: s,
        normal,
        color,
        line_width: half_width,
        edge_dist: 1.0,
    });
    vertices.push(LineVertex {
        position: s,
        normal,
        color,
        line_width: half_width,
        edge_dist: -1.0,
    });
    vertices.push(LineVertex {
        position: e,
        normal,
        color,
        line_width: half_width,
        edge_dist: 1.0,
    });
    // Triangle 2: p1, p3, p2
    vertices.push(LineVertex {
        position: s,
        normal,
        color,
        line_width: half_width,
        edge_dist: -1.0,
    });
    vertices.push(LineVertex {
        position: e,
        normal,
        color,
        line_width: half_width,
        edge_dist: -1.0,
    });
    vertices.push(LineVertex {
        position: e,
        normal,
        color,
        line_width: half_width,
        edge_dist: 1.0,
    });
}
