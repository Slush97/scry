// SPDX-License-Identifier: MIT OR Apache-2.0
//! CPU tessellation of paths, arcs, and polygons into triangle meshes for GPU rendering.
//!
//! Converts arbitrary geometry into `MeshVertex` triangles via:
//! - Bézier curve flattening (quadratic and cubic)
//! - Path → contour conversion
//! - Ear-clipping triangulation

use super::wgpu_context::MeshVertex;

/// Flatness tolerance in pixels — max deviation from true curve to line approximation.
const FLATNESS: f32 = 0.25;

/// Maximum recursion depth for Bézier flattening.
const MAX_DEPTH: u32 = 16;

// ---------------------------------------------------------------------------
// 1A. Bézier Flattening
// ---------------------------------------------------------------------------

/// Flatten a quadratic Bézier (p0, p1, p2) into `out` points.
fn flatten_quad(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], out: &mut Vec<[f32; 2]>) {
    flatten_quad_recursive(p0, p1, p2, out, 0);
}

fn flatten_quad_recursive(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    out: &mut Vec<[f32; 2]>,
    depth: u32,
) {
    if depth >= MAX_DEPTH {
        out.push(p2);
        return;
    }

    // Midpoint deviation: distance from control point to chord midpoint
    let mid_x = (p0[0] + p2[0]) * 0.5;
    let mid_y = (p0[1] + p2[1]) * 0.5;
    let dx = p1[0] - mid_x;
    let dy = p1[1] - mid_y;

    if dx * dx + dy * dy <= FLATNESS * FLATNESS {
        out.push(p2);
        return;
    }

    // De Casteljau subdivision at t=0.5
    let q0 = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
    let q1 = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5];
    let r0 = [(q0[0] + q1[0]) * 0.5, (q0[1] + q1[1]) * 0.5];

    flatten_quad_recursive(p0, q0, r0, out, depth + 1);
    flatten_quad_recursive(r0, q1, p2, out, depth + 1);
}

/// Flatten a cubic Bézier (p0, p1, p2, p3) into `out` points.
fn flatten_cubic(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], out: &mut Vec<[f32; 2]>) {
    flatten_cubic_recursive(p0, p1, p2, p3, out, 0);
}

fn flatten_cubic_recursive(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    out: &mut Vec<[f32; 2]>,
    depth: u32,
) {
    if depth >= MAX_DEPTH {
        out.push(p3);
        return;
    }

    // Check flatness: max deviation of control points from chord
    let dx = p3[0] - p0[0];
    let dy = p3[1] - p0[1];
    let len_sq = dx * dx + dy * dy;

    let d1 = if len_sq > 1e-12 {
        let t = ((p1[0] - p0[0]) * dx + (p1[1] - p0[1]) * dy) / len_sq;
        let proj_x = p0[0] + t * dx;
        let proj_y = p0[1] + t * dy;
        let ex = p1[0] - proj_x;
        let ey = p1[1] - proj_y;
        ex * ex + ey * ey
    } else {
        let ex = p1[0] - p0[0];
        let ey = p1[1] - p0[1];
        ex * ex + ey * ey
    };

    let d2 = if len_sq > 1e-12 {
        let t = ((p2[0] - p0[0]) * dx + (p2[1] - p0[1]) * dy) / len_sq;
        let proj_x = p0[0] + t * dx;
        let proj_y = p0[1] + t * dy;
        let ex = p2[0] - proj_x;
        let ey = p2[1] - proj_y;
        ex * ex + ey * ey
    } else {
        let ex = p2[0] - p0[0];
        let ey = p2[1] - p0[1];
        ex * ex + ey * ey
    };

    let tol_sq = FLATNESS * FLATNESS;
    if d1 <= tol_sq && d2 <= tol_sq {
        out.push(p3);
        return;
    }

    // De Casteljau subdivision at t=0.5
    let ab = mid(p0, p1);
    let bc = mid(p1, p2);
    let cd = mid(p2, p3);
    let abc = mid(ab, bc);
    let bcd = mid(bc, cd);
    let abcd = mid(abc, bcd);

    flatten_cubic_recursive(p0, ab, abc, abcd, out, depth + 1);
    flatten_cubic_recursive(abcd, bcd, cd, p3, out, depth + 1);
}

fn mid(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

// ---------------------------------------------------------------------------
// 1B. Path → Polygon Conversion
// ---------------------------------------------------------------------------

/// Convert a `tiny_skia::Path` into a list of contours (closed polygon point lists).
pub(super) fn path_to_contours(path: &tiny_skia::Path) -> Vec<Vec<[f32; 2]>> {
    let mut contours = Vec::new();
    let mut current: Vec<[f32; 2]> = Vec::new();

    for segment in path.segments() {
        match segment {
            tiny_skia::PathSegment::MoveTo(pt) => {
                if current.len() >= 3 {
                    contours.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
                current.push([pt.x, pt.y]);
            }
            tiny_skia::PathSegment::LineTo(pt) => {
                current.push([pt.x, pt.y]);
            }
            tiny_skia::PathSegment::QuadTo(p1, p2) => {
                let last = current.last().copied().unwrap_or([0.0, 0.0]);
                flatten_quad(last, [p1.x, p1.y], [p2.x, p2.y], &mut current);
            }
            tiny_skia::PathSegment::CubicTo(p1, p2, p3) => {
                let last = current.last().copied().unwrap_or([0.0, 0.0]);
                flatten_cubic(last, [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y], &mut current);
            }
            tiny_skia::PathSegment::Close => {
                if current.len() >= 3 {
                    contours.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }

    // Flush any remaining open contour
    if current.len() >= 3 {
        contours.push(current);
    }

    contours
}

// ---------------------------------------------------------------------------
// 1C. Arc → Polygon
// ---------------------------------------------------------------------------

/// Approximate a circular arc as a polyline (contour).
pub(super) fn arc_to_contour(
    cx: f32,
    cy: f32,
    radius: f32,
    start_angle: f32,
    sweep_angle: f32,
) -> Vec<[f32; 2]> {
    if radius <= 0.0 || sweep_angle.abs() < 1e-6 {
        return Vec::new();
    }

    // Adaptive angular step based on flatness tolerance
    let step = (FLATNESS / radius).acos().clamp(0.01, 0.3) * 2.0;
    let n = ((sweep_angle.abs() / step).ceil() as usize).max(2);
    let dt = sweep_angle / n as f32;

    let mut points = Vec::with_capacity(n + 2);
    // Include center for pie-slice fill
    points.push([cx, cy]);
    for i in 0..=n {
        let angle = start_angle + dt * i as f32;
        points.push([cx + radius * angle.cos(), cy - radius * angle.sin()]);
    }

    points
}

// ---------------------------------------------------------------------------
// 1D. Ear-Clipping Triangulation
// ---------------------------------------------------------------------------

/// Triangulate a simple polygon into triangles (returned as flat `[f32; 2]` triplets).
pub(super) fn ear_clip(contour: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = contour.len();
    if n < 3 {
        return Vec::new();
    }

    // Ensure CCW winding
    let mut poly: Vec<[f32; 2]> = contour.to_vec();
    if signed_area(&poly) < 0.0 {
        poly.reverse();
    }

    let mut indices: Vec<usize> = (0..poly.len()).collect();
    let mut result = Vec::with_capacity((n - 2) * 3);

    let mut safety = poly.len() * poly.len();
    while indices.len() > 2 {
        safety -= 1;
        if safety == 0 {
            break; // prevent infinite loop on degenerate polygons
        }

        let len = indices.len();
        let mut found = false;

        for i in 0..len {
            let prev = indices[(i + len - 1) % len];
            let curr = indices[i];
            let next = indices[(i + 1) % len];

            let a = poly[prev];
            let b = poly[curr];
            let c = poly[next];

            // Must be convex (left turn for CCW)
            if cross(a, b, c) <= 0.0 {
                continue;
            }

            // No other vertex inside this triangle
            let mut ear = true;
            for j in 0..len {
                let idx = indices[j];
                if idx == prev || idx == curr || idx == next {
                    continue;
                }
                if point_in_triangle(poly[idx], a, b, c) {
                    ear = false;
                    break;
                }
            }

            if ear {
                result.push(a);
                result.push(b);
                result.push(c);
                indices.remove(i);
                found = true;
                break;
            }
        }

        if !found {
            break; // no ear found — degenerate polygon
        }
    }

    result
}

/// Signed area of a polygon (positive = CCW, negative = CW).
fn signed_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut area = 0.0_f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1];
        area -= poly[j][0] * poly[i][1];
    }
    area * 0.5
}

/// Cross product of vectors (b-a) × (c-a). Positive = left turn (CCW).
fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Check if point p is inside triangle (a, b, c) using barycentric coordinates.
fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(b, c, p);
    let d3 = cross(c, a, p);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
}

// ---------------------------------------------------------------------------
// 1E. High-Level Entry Points
// ---------------------------------------------------------------------------

/// Tessellate a path into colored triangles for GPU upload.
pub(super) fn tessellate_path(path: &tiny_skia::Path, color: [f32; 4]) -> Vec<MeshVertex> {
    let contours = path_to_contours(path);
    let mut vertices = Vec::new();

    for contour in &contours {
        let tris = ear_clip(contour);
        for pt in &tris {
            vertices.push(MeshVertex {
                position: *pt,
                color,
            });
        }
    }

    vertices
}

/// Tessellate a filled polyline into colored triangles.
pub(super) fn tessellate_polygon(points: &[(f32, f32)], color: [f32; 4]) -> Vec<MeshVertex> {
    if points.len() < 3 {
        return Vec::new();
    }

    let contour: Vec<[f32; 2]> = points.iter().map(|p| (*p).into()).collect();
    let tris = ear_clip(&contour);

    tris.iter()
        .map(|pt| MeshVertex {
            position: *pt,
            color,
        })
        .collect()
}

/// Tessellate an arc into colored triangles (pie-slice fill).
pub(super) fn tessellate_arc(
    cx: f32,
    cy: f32,
    radius: f32,
    start_angle: f32,
    sweep_angle: f32,
    color: [f32; 4],
) -> Vec<MeshVertex> {
    let contour = arc_to_contour(cx, cy, radius, start_angle, sweep_angle);
    if contour.len() < 3 {
        return Vec::new();
    }

    let tris = ear_clip(&contour);

    tris.iter()
        .map(|pt| MeshVertex {
            position: *pt,
            color,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_quad_straight_line() {
        // Collinear control point → should produce ~1 point (the endpoint)
        let mut out = vec![[0.0, 0.0]];
        flatten_quad([0.0, 0.0], [50.0, 0.0], [100.0, 0.0], &mut out);
        assert!(
            out.len() <= 3,
            "straight quad should produce few points, got {}",
            out.len()
        );
        // Last point should be the endpoint
        let last = out.last().unwrap();
        assert!((last[0] - 100.0).abs() < 0.01);
        assert!((last[1]).abs() < 0.01);
    }

    #[test]
    fn flatten_cubic_s_curve() {
        let mut out = vec![[0.0, 0.0]];
        flatten_cubic(
            [0.0, 0.0],
            [0.0, 100.0],
            [100.0, -100.0],
            [100.0, 0.0],
            &mut out,
        );
        // S-curve should produce many points
        assert!(
            out.len() > 4,
            "S-curve should produce >4 points, got {}",
            out.len()
        );
    }

    #[test]
    fn ear_clip_triangle() {
        let tri = [[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]];
        let result = ear_clip(&tri);
        assert_eq!(result.len(), 3, "triangle should produce 3 vertices");
    }

    #[test]
    fn ear_clip_square() {
        let square = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let result = ear_clip(&square);
        assert_eq!(
            result.len(),
            6,
            "square should produce 6 vertices (2 triangles), got {}",
            result.len()
        );
    }

    #[test]
    fn ear_clip_concave_l_shape() {
        // L-shaped polygon (concave)
        let l_shape = [
            [0.0, 0.0],
            [60.0, 0.0],
            [60.0, 40.0],
            [20.0, 40.0],
            [20.0, 80.0],
            [0.0, 80.0],
        ];
        let result = ear_clip(&l_shape);
        // 6 vertices → 4 triangles → 12 output vertices
        assert_eq!(
            result.len(),
            12,
            "L-shape (6 verts) should produce 12 vertices (4 triangles), got {}",
            result.len()
        );
    }

    #[test]
    fn path_to_contours_rect() {
        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(10.0, 10.0);
        pb.line_to(90.0, 10.0);
        pb.line_to(90.0, 90.0);
        pb.line_to(10.0, 90.0);
        pb.close();
        let path = pb.finish().unwrap();

        let contours = path_to_contours(&path);
        assert_eq!(contours.len(), 1, "rect path should produce 1 contour");
        assert_eq!(
            contours[0].len(),
            4,
            "rect contour should have 4 points, got {}",
            contours[0].len()
        );
    }

    #[test]
    fn arc_contour_produces_points() {
        let contour = arc_to_contour(50.0, 50.0, 30.0, 0.0, std::f32::consts::PI);
        assert!(
            contour.len() >= 4,
            "arc should produce multiple points, got {}",
            contour.len()
        );
        // First point should be center (pie-slice)
        assert!((contour[0][0] - 50.0).abs() < 0.01);
        assert!((contour[0][1] - 50.0).abs() < 0.01);
    }

    #[test]
    fn tessellate_polygon_empty() {
        let result = tessellate_polygon(&[(0.0, 0.0), (1.0, 1.0)], [1.0; 4]);
        assert!(result.is_empty(), "2 points should produce no triangles");
    }

    #[test]
    fn tessellate_polygon_triangle() {
        let result = tessellate_polygon(
            &[(0.0, 0.0), (100.0, 0.0), (50.0, 100.0)],
            [1.0, 0.0, 0.0, 1.0],
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].color, [1.0, 0.0, 0.0, 1.0]);
    }
}
