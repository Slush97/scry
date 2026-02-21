// SDF ray marching compute shader.
//
// Full-screen dispatch: one thread per pixel. Each thread generates a camera
// ray, sphere-traces through the scene SDF, then shades the hit point with
// Phong lighting, soft shadows, ambient occlusion, and iterative reflections.

// ═══════════════════════════════════════════════════════════════════
// Uniform / storage types
// ═══════════════════════════════════════════════════════════════════

struct Uniforms {
    // Camera
    eye: vec3<f32>,
    _pad0: f32,
    cam_right: vec3<f32>,
    _pad1: f32,
    cam_up: vec3<f32>,
    _pad2: f32,
    cam_forward: vec3<f32>,
    fov_scale: f32,

    // Viewport
    width: u32,
    height: u32,
    aspect: f32,
    time: f32,

    // Scene
    sky_color: vec4<f32>,
    ambient: f32,
    max_bounces: u32,
    num_objects: u32,
    num_lights: u32,
    has_water: u32,
    god_rays: u32,
    god_ray_density: f32,
    god_ray_samples: u32,
};

// Object shape types (discriminant)
const SHAPE_SPHERE: u32 = 0u;
const SHAPE_BOX: u32 = 1u;
const SHAPE_PLANE: u32 = 2u;
const SHAPE_TORUS: u32 = 3u;
const SHAPE_CYLINDER: u32 = 4u;
const SHAPE_SMOOTH_BLEND: u32 = 5u;
const SHAPE_CAPSULE: u32 = 6u;
const SHAPE_ROUNDED_BOX: u32 = 7u;
const SHAPE_CONE: u32 = 8u;
const SHAPE_TEXT3D: u32 = 9u;
const SHAPE_SUBTRACT: u32 = 10u;
const SHAPE_MANDELBULB: u32 = 11u;
const SHAPE_MENGER: u32 = 12u;
const SHAPE_GYROID: u32 = 13u;
const SHAPE_MORPH: u32 = 14u;

// Material types (discriminant)
const MAT_SOLID: u32 = 0u;
const MAT_WATER: u32 = 1u;
const MAT_FIRE: u32 = 2u;
const MAT_CHECKER: u32 = 3u;
const MAT_GLASS: u32 = 4u;
const MAT_RAINBOW: u32 = 5u;
const MAT_SUBSURFACE: u32 = 6u;

struct GpuObject {
    position: vec3<f32>,
    shape_type: u32,
    shape_params: vec4<f32>,
    blend_a_params: vec4<f32>,
    blend_b_params: vec4<f32>,
    blend_b_offset: vec3<f32>,
    material_type: u32,
    material_params: vec4<f32>,
    material_color: vec4<f32>,
    bounding_radius: f32,
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    orientation: vec4<f32>,  // quaternion (x, y, z, w) — pre-conjugated for inverse rotation
};

struct GpuLight {
    position: vec3<f32>,
    intensity: f32,
    color: vec4<f32>,
};

// ═══════════════════════════════════════════════════════════════════
// Bindings
// ═══════════════════════════════════════════════════════════════════

struct GpuGlyphMeta {
    x_offset: f32,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    grid_width: u32,
    grid_height: u32,
    grid_offset: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> objects: array<GpuObject>;
@group(0) @binding(2) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read> glyph_meta: array<GpuGlyphMeta>;
@group(0) @binding(5) var<storage, read> glyph_grids: array<f32>;

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const MAX_DIST: f32 = 50.0;
const SURF_DIST: f32 = 0.002;
const NORMAL_EPS: f32 = 0.005;
const OMEGA: f32 = 1.6;
const RELAX_DIST: f32 = 0.02;
const SHADOW_K: f32 = 16.0;

// ═══════════════════════════════════════════════════════════════════
// Permutation table for noise (same as CPU)
// ═══════════════════════════════════════════════════════════════════

const PERM: array<u32, 256> = array<u32, 256>(
    151u, 160u, 137u, 91u, 90u, 15u, 131u, 13u, 201u, 95u, 96u, 53u, 194u, 233u, 7u, 225u,
    140u, 36u, 103u, 30u, 69u, 142u, 8u, 99u, 37u, 240u, 21u, 10u, 23u, 190u, 6u, 148u,
    247u, 120u, 234u, 75u, 0u, 26u, 197u, 62u, 94u, 252u, 219u, 203u, 117u, 35u, 11u, 32u,
    57u, 177u, 33u, 88u, 237u, 149u, 56u, 87u, 174u, 20u, 125u, 136u, 171u, 168u, 68u, 175u,
    74u, 165u, 71u, 134u, 139u, 48u, 27u, 166u, 77u, 146u, 158u, 231u, 83u, 111u, 229u, 122u,
    60u, 211u, 133u, 230u, 220u, 105u, 92u, 41u, 55u, 46u, 245u, 40u, 244u, 102u, 143u, 54u,
    65u, 25u, 63u, 161u, 1u, 216u, 80u, 73u, 209u, 76u, 132u, 187u, 208u, 89u, 18u, 169u,
    200u, 196u, 135u, 130u, 116u, 188u, 159u, 86u, 164u, 100u, 109u, 198u, 173u, 186u, 3u, 64u,
    52u, 217u, 226u, 250u, 124u, 123u, 5u, 202u, 38u, 147u, 118u, 126u, 255u, 82u, 85u, 212u,
    207u, 206u, 59u, 227u, 47u, 16u, 58u, 17u, 182u, 189u, 28u, 42u, 223u, 183u, 170u, 213u,
    119u, 248u, 152u, 2u, 44u, 154u, 163u, 70u, 221u, 153u, 101u, 155u, 167u, 43u, 172u, 9u,
    129u, 22u, 39u, 253u, 19u, 98u, 108u, 110u, 79u, 113u, 224u, 232u, 178u, 185u, 112u, 104u,
    218u, 246u, 97u, 228u, 251u, 34u, 242u, 193u, 238u, 210u, 144u, 12u, 191u, 179u, 162u, 241u,
    81u, 51u, 145u, 235u, 249u, 14u, 239u, 107u, 49u, 192u, 214u, 31u, 181u, 199u, 106u, 157u,
    184u, 84u, 204u, 176u, 115u, 121u, 50u, 45u, 127u, 4u, 150u, 254u, 138u, 236u, 205u, 93u,
    222u, 114u, 67u, 29u, 24u, 72u, 243u, 141u, 128u, 195u, 78u, 66u, 215u, 61u, 156u, 180u
);

fn perm(i: i32) -> u32 {
    return PERM[u32(i) & 255u];
}

fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let xi = u32(x) & 255u;
    let a = (PERM[xi] + u32(y)) & 255u;
    let b = (PERM[a] + u32(z)) & 255u;
    return PERM[b];
}

// Quintic smoothstep (6t^5 - 15t^4 + 10t^3)
fn smooth_noise(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 12 gradient vectors for 3D Perlin noise
fn grad3(hash_val: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash_val % 12u;
    switch h {
        case 0u:  { return  x + y; }
        case 1u:  { return -x + y; }
        case 2u:  { return  x - y; }
        case 3u:  { return -x - y; }
        case 4u:  { return  x + z; }
        case 5u:  { return -x + z; }
        case 6u:  { return  x - z; }
        case 7u:  { return -x - z; }
        case 8u:  { return  y + z; }
        case 9u:  { return -y + z; }
        case 10u: { return  y - z; }
        case 11u: { return -y - z; }
        default:  { return  x + y; }
    }
}

// 3D gradient noise (Perlin-style), output in [0, 1]
fn noise3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    let zi = i32(floor(z));
    let xf = x - floor(x);
    let yf = y - floor(y);
    let zf = z - floor(z);
    let uu = smooth_noise(xf);
    let vv = smooth_noise(yf);
    let ww = smooth_noise(zf);

    let c000 = grad3(hash3(xi,     yi,     zi),     xf,       yf,       zf);
    let c100 = grad3(hash3(xi + 1, yi,     zi),     xf - 1.0, yf,       zf);
    let c010 = grad3(hash3(xi,     yi + 1, zi),     xf,       yf - 1.0, zf);
    let c110 = grad3(hash3(xi + 1, yi + 1, zi),     xf - 1.0, yf - 1.0, zf);
    let c001 = grad3(hash3(xi,     yi,     zi + 1), xf,       yf,       zf - 1.0);
    let c101 = grad3(hash3(xi + 1, yi,     zi + 1), xf - 1.0, yf,       zf - 1.0);
    let c011 = grad3(hash3(xi,     yi + 1, zi + 1), xf,       yf - 1.0, zf - 1.0);
    let c111 = grad3(hash3(xi + 1, yi + 1, zi + 1), xf - 1.0, yf - 1.0, zf - 1.0);

    let x00 = c000 + uu * (c100 - c000);
    let x10 = c010 + uu * (c110 - c010);
    let x01 = c001 + uu * (c101 - c001);
    let x11 = c011 + uu * (c111 - c011);
    let y0 = x00 + vv * (x10 - x00);
    let y1 = x01 + vv * (x11 - x01);
    let result = y0 + ww * (y1 - y0);
    // Remap from ~[-0.7, 0.7] to [0, 1]
    return result * 0.5 + 0.5;
}

fn fbm2d(x: f32, y: f32, octaves: u32) -> f32 {
    var value = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < octaves; i++) {
        value += amp * noise3d(x * freq, y * freq, 0.0);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

fn fbm3d(p: vec3<f32>, octaves: u32) -> f32 {
    var value = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < octaves; i++) {
        value += amp * noise3d(p.x * freq, p.y * freq, p.z * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// ═══════════════════════════════════════════════════════════════════
// SDF primitives
// ═══════════════════════════════════════════════════════════════════

fn sd_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sd_plane(p: vec3<f32>) -> f32 {
    return p.y;
}

fn sd_box(p: vec3<f32>, half_ext: vec3<f32>) -> f32 {
    let q = abs(p) - half_ext;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_torus(p: vec3<f32>, major: f32, minor: f32) -> f32 {
    let q_x = length(vec2<f32>(p.x, p.z)) - major;
    return length(vec2<f32>(q_x, p.y)) - minor;
}

fn sd_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let d_x = length(vec2<f32>(p.x, p.z)) - radius;
    let d_y = abs(p.y) - half_height;
    return length(max(vec2<f32>(d_x, d_y), vec2<f32>(0.0))) + min(max(d_x, d_y), 0.0);
}

fn sd_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let py = p.y - clamp(p.y, -half_height, half_height);
    let q = vec3<f32>(p.x, py, p.z);
    return length(q) - radius;
}

fn sd_rounded_box(p: vec3<f32>, half_ext: vec3<f32>, radius: f32) -> f32 {
    return sd_box(p, half_ext) - radius;
}

fn sd_cone(p: vec3<f32>, radius: f32, height: f32) -> f32 {
    let q_len = length(vec2<f32>(p.x, p.z));
    let q = vec2<f32>(q_len, p.y);
    let tip = vec2<f32>(0.0, height);
    let base = vec2<f32>(radius, 0.0);
    let cb = base - tip;
    let cb_len_sq = dot(cb, cb);
    let qp = q - tip;
    let t = clamp(dot(qp, cb) / cb_len_sq, 0.0, 1.0);
    let closest = tip + cb * t;
    let diff = q - closest;
    let dist_to_edge = length(diff);
    let cross_val = cb.x * qp.y - cb.y * qp.x;
    if cross_val <= 0.0 && q.y >= 0.0 && q.y <= height {
        return -dist_to_edge;
    }
    return dist_to_edge;
}

fn sd_mandelbulb(pos: vec3<f32>, power: f32, max_iter: u32) -> f32 {
    var w = pos;
    var m = dot(w, w);
    var dz = 1.0;

    for (var i = 0u; i < max_iter; i++) {
        let m_sqrt = sqrt(m);
        dz = power * pow(m_sqrt, power - 1.0) * dz + 1.0;

        let r = m_sqrt;
        let theta = atan2(w.y, w.x);
        let phi = asin(clamp(w.z / r, -1.0, 1.0));

        let rp = pow(r, power);
        let tp = theta * power;
        let pp = phi * power;

        w = vec3<f32>(
            rp * cos(pp) * cos(tp),
            rp * cos(pp) * sin(tp),
            rp * sin(pp),
        ) + pos;

        m = dot(w, w);
        if m > 256.0 {
            break;
        }
    }

    let r = sqrt(m);
    return 0.25 * r * log(r) / dz;
}

fn sd_menger_sponge(p: vec3<f32>, iterations: u32) -> f32 {
    var d = sd_box(p, vec3<f32>(1.0, 1.0, 1.0));
    var s = 1.0;

    for (var i = 0u; i < iterations; i++) {
        let a = vec3<f32>(
            ((p.x * s) % 2.0 + 3.0) % 2.0 - 1.0,
            ((p.y * s) % 2.0 + 3.0) % 2.0 - 1.0,
            ((p.z * s) % 2.0 + 3.0) % 2.0 - 1.0,
        );
        s *= 3.0;

        let r = vec3<f32>(
            abs(1.0 - 3.0 * abs(a.x)),
            abs(1.0 - 3.0 * abs(a.y)),
            abs(1.0 - 3.0 * abs(a.z)),
        );

        let da = max(r.y, r.z) / s;
        let db = max(r.x, r.z) / s;
        let dc = max(r.x, r.y) / s;
        let c = min(min(da, db), dc);

        d = max(d, c);
    }

    return d;
}

fn sd_gyroid(p: vec3<f32>, scale: f32, thickness: f32, bound: f32) -> f32 {
    let sp = p * scale;
    let val = sin(sp.x) * cos(sp.y)
            + sin(sp.y) * cos(sp.z)
            + sin(sp.z) * cos(sp.x);
    let gyroid_d = (abs(val) - thickness) / scale;

    if bound > 0.0 {
        let sphere_d = length(p) - bound;
        return max(gyroid_d, sphere_d);
    }
    return gyroid_d;
}

// ═══════════════════════════════════════════════════════════════════
// Text3D glyph sampling
// ═══════════════════════════════════════════════════════════════════

fn sample_glyph_sdf(glyph: GpuGlyphMeta, x: f32, y: f32) -> f32 {
    let bw = glyph.max_x - glyph.min_x;
    let bh = glyph.max_y - glyph.min_y;
    if bw < 1e-10 || bh < 1e-10 {
        return 1e6;
    }

    // Map world coords to grid coords
    let gx = (x - glyph.min_x) / bw * (f32(glyph.grid_width) - 1.0);
    let gy = (y - glyph.min_y) / bh * (f32(glyph.grid_height) - 1.0);

    // Outside bounds: approximate distance to bounding box
    if gx < -1.0 || gy < -1.0 || gx > f32(glyph.grid_width) || gy > f32(glyph.grid_height) {
        var dx = 0.0;
        if x < glyph.min_x { dx = glyph.min_x - x; }
        else if x > glyph.max_x { dx = x - glyph.max_x; }
        var dy = 0.0;
        if y < glyph.min_y { dy = glyph.min_y - y; }
        else if y > glyph.max_y { dy = y - glyph.max_y; }
        return sqrt(dx * dx + dy * dy) + 0.01;
    }

    // Catmull-Rom bicubic interpolation (C1 continuous — smooth surface AND gradient)
    let cgx = clamp(gx, 0.0, f32(glyph.grid_width - 1u));
    let cgy = clamp(gy, 0.0, f32(glyph.grid_height - 1u));
    let ix = i32(cgx);
    let iy = i32(cgy);
    let fx = cgx - f32(ix);
    let fy = cgy - f32(iy);

    // Catmull-Rom basis weights
    let fx2 = fx * fx;
    let fx3 = fx2 * fx;
    let wx0 = -0.5 * fx3 + fx2 - 0.5 * fx;
    let wx1 =  1.5 * fx3 - 2.5 * fx2 + 1.0;
    let wx2 = -1.5 * fx3 + 2.0 * fx2 + 0.5 * fx;
    let wx3 =  0.5 * fx3 - 0.5 * fx2;

    let fy2 = fy * fy;
    let fy3 = fy2 * fy;
    let wy0 = -0.5 * fy3 + fy2 - 0.5 * fy;
    let wy1 =  1.5 * fy3 - 2.5 * fy2 + 1.0;
    let wy2 = -1.5 * fy3 + 2.0 * fy2 + 0.5 * fy;
    let wy3 =  0.5 * fy3 - 0.5 * fy2;

    let base = glyph.grid_offset;
    let w = i32(glyph.grid_width);
    let h = i32(glyph.grid_height);

    // 4×4 separable convolution with clamped boundary access
    var grid_dist = 0.0;
    let wy = array<f32, 4>(wy0, wy1, wy2, wy3);
    let wx = array<f32, 4>(wx0, wx1, wx2, wx3);
    for (var jj = 0; jj < 4; jj++) {
        let sy = clamp(iy + jj - 1, 0, h - 1);
        var row_val = 0.0;
        for (var ii = 0; ii < 4; ii++) {
            let sx = clamp(ix + ii - 1, 0, w - 1);
            row_val += glyph_grids[u32(i32(base) + sy * w + sx)] * wx[ii];
        }
        grid_dist += row_val * wy[jj];
    }

    // Convert from grid-cell units to world units
    let pixels_per_world = f32(glyph.grid_width) / bw;
    return grid_dist / pixels_per_world;
}

// Gradient of the Catmull-Rom interpolated SDF via central finite differences.
// Returns vec3(distance, ∂d/∂x, ∂d/∂y) in world units.
// Uses a half-grid-cell step — since Catmull-Rom is C1 continuous, this gives
// smooth gradients equivalent to analytical derivatives without the code bloat.
fn sample_glyph_sdf_gradient(glyph: GpuGlyphMeta, x: f32, y: f32) -> vec3<f32> {
    let bw = glyph.max_x - glyph.min_x;
    let bh = glyph.max_y - glyph.min_y;
    if bw < 1e-10 || bh < 1e-10 {
        return vec3<f32>(1e6, 0.0, 0.0);
    }
    let d = sample_glyph_sdf(glyph, x, y);
    // Half-grid-cell step in world units — small enough for accuracy,
    // large enough to span the C1 interpolation smoothly.
    let hx = 0.5 * bw / f32(glyph.grid_width);
    let hy = 0.5 * bh / f32(glyph.grid_height);
    let dx = sample_glyph_sdf(glyph, x + hx, y)
           - sample_glyph_sdf(glyph, x - hx, y);
    let dy = sample_glyph_sdf(glyph, x, y + hy)
           - sample_glyph_sdf(glyph, x, y - hy);
    return vec3<f32>(d, dx / (2.0 * hx), dy / (2.0 * hy));
}



// Analytical normal for extruded Text3D shapes.
// Uses exact Catmull-Rom bicubic gradient derivatives for C1-smooth normals
// with zero grid artifacts, regardless of curve tightness.
fn estimate_text3d_normal(p: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let depth = obj.shape_params.x;
    let total_width = obj.shape_params.y;
    let ascent = obj.shape_params.z;
    let descent = obj.shape_params.w;

    let glyph_start = u32(obj.blend_a_params.x);
    let glyph_count = u32(obj.blend_a_params.y);

    let center_x = total_width * 0.5;
    let center_y = (ascent - descent) * 0.5;
    let sample_x = p.x + center_x;
    let sample_y = p.y + center_y;

    // Find the closest glyph and get its analytical gradient in one pass.
    var d2d = 1e6;
    var best_grad = vec2<f32>(0.0, 1.0);
    for (var i = 0u; i < glyph_count; i++) {
        let glyph = glyph_meta[glyph_start + i];
        let gx = sample_x - glyph.x_offset;
        let result = sample_glyph_sdf_gradient(glyph, gx, sample_y);
        if result.x < d2d {
            d2d = result.x;
            best_grad = result.yz;  // (∂d/∂x, ∂d/∂y)
        }
    }

    let dz = abs(p.z) - depth * 0.5;
    let sign_z = select(-1.0, 1.0, p.z >= 0.0);

    // Normalize the analytical 2D gradient (fallback to +Y if degenerate)
    let grad_len = length(best_grad);
    let grad_dir = select(vec2<f32>(0.0, 1.0), best_grad / grad_len, grad_len > 1e-6);

    // Smooth blend between face normal (2D gradient) and cap normal (±Z).
    // ±0.06 gives a wide enough transition to hide any residual artifacts
    // at the front-face-to-side-face crease.
    let blend = smoothstep(-0.06, 0.06, d2d - dz);
    let face_n = vec3<f32>(grad_dir.x, grad_dir.y, 0.0);
    let side_n = vec3<f32>(0.0, 0.0, sign_z);

    return normalize(mix(side_n, face_n, blend));
}

fn sd_text3d(p: vec3<f32>, obj: GpuObject) -> f32 {
    let depth = obj.shape_params.x;
    let total_width = obj.shape_params.y;
    let ascent = obj.shape_params.z;
    let descent = obj.shape_params.w;

    let glyph_start = u32(obj.blend_a_params.x);
    let glyph_count = u32(obj.blend_a_params.y);

    // Center the text
    let center_x = total_width * 0.5;
    let center_y = (ascent - descent) * 0.5;
    let sample_x = p.x + center_x;
    let sample_y = p.y + center_y;

    // Find minimum 2D distance across all glyphs
    var d2d = 1e6;
    for (var i = 0u; i < glyph_count; i++) {
        let glyph = glyph_meta[glyph_start + i];
        let gx = sample_x - glyph.x_offset;
        let d = sample_glyph_sdf(glyph, gx, sample_y);
        d2d = min(d2d, d);
    }

    // IQ extrusion formula
    let dz = abs(p.z) - depth * 0.5;
    let w = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return length(w) + min(max(d2d, dz), 0.0);
}

// Rotate vector by unit quaternion (q = xyz, w)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn smooth_min(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return d2 + (d1 - d2) * h - k * h * (1.0 - h);
}

// Evaluate a single sub-shape SDF given type + params
fn eval_sub_shape(shape_type: u32, params: vec4<f32>, p: vec3<f32>) -> f32 {
    switch shape_type {
        case 0u: { return sd_sphere(p, params.x); }                    // SHAPE_SPHERE
        case 1u: { return sd_box(p, vec3<f32>(params.x, params.y, params.z)); } // SHAPE_BOX
        case 2u: { return sd_plane(p); }                               // SHAPE_PLANE
        case 3u: { return sd_torus(p, params.x, params.y); }          // SHAPE_TORUS
        case 4u: { return sd_cylinder(p, params.x, params.y); }       // SHAPE_CYLINDER
        case 6u: { return sd_capsule(p, params.x, params.y); }        // SHAPE_CAPSULE
        case 7u: { return sd_rounded_box(p, vec3<f32>(params.x, params.y, params.z), params.w); } // SHAPE_ROUNDED_BOX
        case 8u: { return sd_cone(p, params.x, params.y); }           // SHAPE_CONE
        case 11u: { return sd_mandelbulb(p, params.x, u32(params.y)); } // SHAPE_MANDELBULB
        case 12u: { return sd_menger_sponge(p, u32(params.x)); }      // SHAPE_MENGER
        case 13u: { return sd_gyroid(p, params.x, params.y, params.z); } // SHAPE_GYROID
        default: { return sd_sphere(p, 1.0); }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Water displacement
// ═══════════════════════════════════════════════════════════════════

fn water_displacement(x: f32, z: f32, time: f32, amplitude: f32, frequency: f32) -> f32 {
    let w1 = sin(x * frequency + time * 2.0) * amplitude;
    let w2 = sin(z * frequency * 0.7 + time * 1.5) * amplitude * 0.6;
    let w3 = sin((x + z) * frequency * 1.3 + time * 2.5) * amplitude * 0.3;
    let n = fbm2d(x * 0.5, z * 0.5 + time * 0.3, 2u) * amplitude * 0.4;
    return w1 + w2 + w3 + n;
}

// ═══════════════════════════════════════════════════════════════════
// Scene SDF evaluation
// ═══════════════════════════════════════════════════════════════════

struct SdfResult {
    dist: f32,
    idx: u32,
};

fn shape_sdf(obj: GpuObject, local: vec3<f32>) -> f32 {
    if obj.shape_type == SHAPE_TEXT3D {
        return sd_text3d(local, obj);
    }
    if obj.shape_type == SHAPE_SMOOTH_BLEND {
        let sub_a_type = u32(obj.shape_params.y);
        let sub_b_type = u32(obj.shape_params.z);
        let k = obj.shape_params.x;
        let da = eval_sub_shape(sub_a_type, obj.blend_a_params, local);
        let db = eval_sub_shape(sub_b_type, obj.blend_b_params, local - obj.blend_b_offset);
        return smooth_min(da, db, k);
    }
    if obj.shape_type == SHAPE_SUBTRACT {
        let sub_a_type = u32(obj.shape_params.y);
        let sub_b_type = u32(obj.shape_params.z);
        let da = eval_sub_shape(sub_a_type, obj.blend_a_params, local);
        let db = eval_sub_shape(sub_b_type, obj.blend_b_params, local - obj.blend_b_offset);
        return max(da, -db);
    }
    if obj.shape_type == SHAPE_MORPH {
        let t = obj.shape_params.x;
        let sub_a_type = u32(obj.shape_params.y);
        let sub_b_type = u32(obj.shape_params.z);
        let da = eval_sub_shape(sub_a_type, obj.blend_a_params, local);
        let db = eval_sub_shape(sub_b_type, obj.blend_b_params, local);
        return mix(da, db, t);
    }
    return eval_sub_shape(obj.shape_type, obj.shape_params, local);
}

fn object_sdf(obj: GpuObject, point: vec3<f32>) -> f32 {
    var local = point - obj.position;

    // Apply inverse rotation via pre-conjugated quaternion
    // Identity quaternion (0,0,0,1) is a no-op: cross with zero = zero
    local = quat_rotate(obj.orientation, local);

    let base_dist = shape_sdf(obj, local);

    // Water displacement
    if obj.material_type == MAT_WATER && obj.shape_type == SHAPE_PLANE {
        let amplitude = obj.material_params.y;
        let frequency = obj.material_params.z;
        let max_disp = amplitude * 2.3;
        if abs(base_dist) > max_disp {
            return base_dist;
        }
        let disp = water_displacement(point.x, point.z, u.time, amplitude, frequency);
        return base_dist - disp;
    }

    return base_dist;
}

fn scene_sdf(point: vec3<f32>) -> SdfResult {
    var min_dist = MAX_DIST;
    var closest = 0u;
    for (var i = 0u; i < u.num_objects; i++) {
        let obj = objects[i];
        // Bounding sphere culling
        let center_dist = length(point - obj.position);
        if center_dist - obj.bounding_radius > min_dist {
            continue;
        }
        let d = object_sdf(obj, point);
        if d < min_dist {
            min_dist = d;
            closest = i;
        }
    }
    return SdfResult(min_dist, closest);
}

// ═══════════════════════════════════════════════════════════════════
// Normal estimation
// ═══════════════════════════════════════════════════════════════════

fn water_normal(x: f32, z: f32, amplitude: f32, frequency: f32) -> vec3<f32> {
    let e = 0.002;
    let dx = water_displacement(x + e, z, u.time, amplitude, frequency)
           - water_displacement(x - e, z, u.time, amplitude, frequency);
    let dz = water_displacement(x, z + e, u.time, amplitude, frequency)
           - water_displacement(x, z - e, u.time, amplitude, frequency);
    return normalize(vec3<f32>(-dx / (2.0 * e), 1.0, -dz / (2.0 * e)));
}

fn estimate_normal(point: vec3<f32>, obj_idx: u32) -> vec3<f32> {
    let obj = objects[obj_idx];

    // Water: displacement gradient
    if obj.material_type == MAT_WATER && obj.shape_type == SHAPE_PLANE {
        return water_normal(point.x, point.z, obj.material_params.y, obj.material_params.z);
    }

    // Text3D: use analytical normals from bilinear-interpolated SDF gradient.
    // This completely avoids finite-difference artifacts on the discrete grid.
    if obj.shape_type == SHAPE_TEXT3D {
        // Transform hit point to object-local space
        var local = point - obj.position;
        local = quat_rotate(obj.orientation, local);
        let local_n = estimate_text3d_normal(local, obj);
        // Rotate the local normal back to world space (inverse of conjugated quat = original quat)
        let inv_q = vec4<f32>(-obj.orientation.xyz, obj.orientation.w);
        return normalize(quat_rotate(inv_q, local_n));
    }

    // Tetrahedron technique (4 scene_sdf evals)
    let e = NORMAL_EPS;
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);
    let n = k0 * scene_sdf(point + k0 * e).dist
          + k1 * scene_sdf(point + k1 * e).dist
          + k2 * scene_sdf(point + k2 * e).dist
          + k3 * scene_sdf(point + k3 * e).dist;
    return normalize(n);
}

// ═══════════════════════════════════════════════════════════════════
// Ray marching
// ═══════════════════════════════════════════════════════════════════

struct HitResult {
    hit: bool,
    point: vec3<f32>,
    obj_idx: u32,
    dist: f32,
};

// Enhanced sphere tracing with over-relaxation (ω=1.6) and per-bounce step budgets
fn ray_march(origin: vec3<f32>, dir: vec3<f32>, bounce: u32) -> HitResult {
    // Per-bounce step budget: primary=48, bounce1=32, bounce2+=24
    var steps: u32;
    var shadow_steps_unused: u32; // shadow steps handled separately
    if bounce == 0u {
        steps = 128u;
    } else if bounce == 1u {
        steps = 64u;
    } else {
        steps = 48u;
    }
    let always_relax = u.has_water == 0u;
    var omega = select(1.0, OMEGA, always_relax);
    var t = SURF_DIST;
    var prev_d = 0.0;
    var prev_step = 0.0;
    for (var i = 0u; i < steps; i++) {
        let p = origin + dir * t;
        let res = scene_sdf(p);
        if res.dist < SURF_DIST {
            // Bisection refinement: binary search between prev and current t
            // for sub-grid-cell precision (8 iterations ≈ 1/256 step accuracy)
            var lo = t - prev_step;
            var hi = t;
            var mid_idx = res.idx;
            for (var b = 0u; b < 8u; b++) {
                let mid = (lo + hi) * 0.5;
                let mp = origin + dir * mid;
                let mr = scene_sdf(mp);
                mid_idx = mr.idx;
                if mr.dist < SURF_DIST {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let final_t = (lo + hi) * 0.5;
            let final_p = origin + dir * final_t;
            return HitResult(true, final_p, mid_idx, final_t);
        }
        if !always_relax {
            omega = select(1.0, OMEGA, res.dist > RELAX_DIST);
        }
        // Rewind-on-overshoot fallback
        if omega > 1.0 && prev_step > 0.0 && res.dist + prev_d < prev_step {
            t -= prev_step - prev_d;
            omega = 1.0;
            prev_d = 0.0;
            prev_step = 0.0;
            continue;
        }
        let step_size = res.dist * omega;
        prev_d = res.dist;
        prev_step = step_size;
        t += step_size;
        if t > MAX_DIST {
            break;
        }
    }
    return HitResult(false, vec3<f32>(0.0), 0u, t);
}

// Soft shadow using IQ's penumbra technique
// Returns 1.0 (fully lit) to 0.0 (fully shadowed)
fn soft_shadow(origin: vec3<f32>, dir: vec3<f32>, max_t: f32, shadow_steps: u32) -> f32 {
    var t = SURF_DIST * 16.0;
    var res = 1.0;
    for (var i = 0u; i < shadow_steps; i++) {
        let p = origin + dir * t;
        let sdf = scene_sdf(p);
        if sdf.dist < SURF_DIST * 0.5 {
            return 0.0;
        }
        res = min(res, SHADOW_K * sdf.dist / t);
        t += clamp(sdf.dist, 0.01, 0.5);
        if t > max_t {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════
// Ambient occlusion
// ═══════════════════════════════════════════════════════════════════

fn ambient_occlusion(hit: vec3<f32>, normal: vec3<f32>, ao_scale: f32) -> f32 {
    var occ = 0.0;
    var weight = 1.0;
    // ao_scale > 1 widens the sampling hemisphere (reduces false occlusion on
    // thin features like text edges); values < 1 tighten it.
    let step_base = 0.02 * ao_scale;
    for (var i = 1; i <= 5; i++) {
        let dist = step_base * f32(i);
        let d = scene_sdf(hit + normal * dist).dist;
        occ += (dist - d) * weight;
        weight *= 0.75;
    }
    // Dampen the occlusion contribution proportionally to ao_scale so that
    // larger step radii don't over-darken flat areas.
    return max(1.0 - clamp(occ / ao_scale, 0.0, 1.0), 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// Shading
// ═══════════════════════════════════════════════════════════════════

fn fresnel(cos_theta: f32, ior: f32) -> f32 {
    let r0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    return r0 + (1.0 - r0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

fn fire_color_ramp(t: f32) -> vec3<f32> {
    let tc = clamp(t, 0.0, 1.0);
    let r = clamp(tc * 3.0, 0.0, 1.0);
    let g = clamp((tc - 0.33) * 3.0, 0.0, 1.0);
    let b = clamp((tc - 0.66) * 3.0, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

// HSL to linear RGB conversion
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    var hh = fract(h);
    if hh < 0.0 { hh += 1.0; }
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let h6 = hh * 6.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    var r1 = 0.0;
    var g1 = 0.0;
    var b1 = 0.0;
    if h6 < 1.0 {
        r1 = c; g1 = x;
    } else if h6 < 2.0 {
        r1 = x; g1 = c;
    } else if h6 < 3.0 {
        g1 = c; b1 = x;
    } else if h6 < 4.0 {
        g1 = x; b1 = c;
    } else if h6 < 5.0 {
        r1 = x; b1 = c;
    } else {
        r1 = c; b1 = x;
    }
    let m = l - c * 0.5;
    return vec3<f32>(
        clamp(r1 + m, 0.0, 1.0),
        clamp(g1 + m, 0.0, 1.0),
        clamp(b1 + m, 0.0, 1.0),
    );
}

fn phong_full(hit: vec3<f32>, normal: vec3<f32>, ray_dir: vec3<f32>,
              base_color: vec3<f32>, spec_power: f32, do_shadows: bool) -> vec3<f32> {
    return phong_full_ex(hit, normal, ray_dir, base_color, spec_power, do_shadows, SURF_DIST * 8.0, 1.0, false);
}

// Extended Phong with configurable shadow bias, AO scale, and shadow skip.
// `shadow_bias`    — distance along the normal to offset the shadow ray origin.
// `ao_scale`       — multiplier for AO sampling radius (>1 = softer).
// `skip_shadows`   — if true, skip shadow rays entirely (for Text3D self-shadow avoidance).
fn phong_full_ex(hit: vec3<f32>, normal: vec3<f32>, ray_dir: vec3<f32>,
                 base_color: vec3<f32>, spec_power: f32, do_shadows: bool,
                 shadow_bias: f32, ao_scale: f32, skip_shadows: bool) -> vec3<f32> {
    var ao = 1.0;
    if do_shadows {
        ao = ambient_occlusion(hit, normal, ao_scale);
    }
    var r = base_color.x * u.ambient * ao;
    var g = base_color.y * u.ambient * ao;
    var b = base_color.z * u.ambient * ao;

    for (var i = 0u; i < u.num_lights; i++) {
        let light = lights[i];
        let delta = light.position - hit;
        let light_dist = length(delta);
        let to_light = delta / light_dist;

        let n_dot_l = dot(normal, to_light);
        if n_dot_l <= 0.0 {
            continue;
        }

        var shadow = 1.0;
        if do_shadows && !skip_shadows {
            let shadow_origin = hit + normal * shadow_bias;
            shadow = soft_shadow(shadow_origin, to_light, light_dist, 32u);
            if shadow < 0.001 {
                continue;
            }
        }

        let intensity = light.intensity;
        let diff = n_dot_l * intensity * shadow;
        r += base_color.x * light.color.x * diff;
        g += base_color.y * light.color.y * diff;
        b += base_color.z * light.color.z * diff;

        let half_vec = normalize(to_light - ray_dir);
        let spec = pow(max(dot(normal, half_vec), 0.0), spec_power) * intensity * shadow;
        r += light.color.x * spec * 0.5;
        g += light.color.y * spec * 0.5;
        b += light.color.z * spec * 0.5;
    }

    return vec3<f32>(min(r, 1.0), min(g, 1.0), min(b, 1.0));
}

fn phong_specular_only(hit: vec3<f32>, normal: vec3<f32>, ray_dir: vec3<f32>,
                       spec_power: f32) -> vec3<f32> {
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    for (var i = 0u; i < u.num_lights; i++) {
        let light = lights[i];
        let to_light = normalize(light.position - hit);
        let half_vec = normalize(to_light - ray_dir);
        let spec = pow(max(dot(normal, half_vec), 0.0), spec_power) * light.intensity;
        r += light.color.x * spec;
        g += light.color.y * spec;
        b += light.color.z * spec;
    }
    return vec3<f32>(min(r, 1.0), min(g, 1.0), min(b, 1.0));
}

fn reflect_dir(d: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return d - n * (2.0 * dot(d, n));
}

fn refract_dir(d: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_i = -dot(d, n);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t > 1.0 {
        // Total internal reflection — return zero sentinel
        return vec3<f32>(0.0);
    }
    let cos_t = sqrt(1.0 - sin2_t);
    return d * eta + n * (eta * cos_i - cos_t);
}

// ═══════════════════════════════════════════════════════════════════
// Fire volume marching
// ═══════════════════════════════════════════════════════════════════

fn march_fire_volume(origin: vec3<f32>, dir: vec3<f32>, obj: GpuObject) -> vec4<f32> {
    let intensity = obj.material_params.x;
    let noise_scale = obj.material_params.y;
    let speed = obj.material_params.z;

    var bounding_radius: f32;
    if obj.shape_type == SHAPE_SPHERE {
        bounding_radius = obj.shape_params.x;
    } else if obj.shape_type == SHAPE_CYLINDER {
        bounding_radius = max(obj.shape_params.x, obj.shape_params.y);
    } else {
        bounding_radius = 2.0;
    }

    let oc = origin - obj.position;
    let b_val = dot(oc, dir);
    let c_val = dot(oc, oc) - bounding_radius * bounding_radius;
    let disc = b_val * b_val - c_val;
    if disc < 0.0 {
        return vec4<f32>(0.0);
    }

    let sqrt_disc = sqrt(disc);
    let t_near = max(-b_val - sqrt_disc, 0.0);
    let t_far = -b_val + sqrt_disc;
    if t_far < 0.0 {
        return vec4<f32>(0.0);
    }

    let step_size = bounding_radius * 0.1;
    var t = t_near;
    var accum = vec4<f32>(0.0);

    while t < t_far && accum.w < 0.95 {
        let p = origin + dir * t;
        let local = p - obj.position;

        let noise_p = vec3<f32>(
            local.x * noise_scale,
            local.y * noise_scale - u.time * speed,
            local.z * noise_scale,
        );
        let raw_density = fbm3d(noise_p, 3u);

        let dist_from_center = length(vec2<f32>(local.x, local.z)) / bounding_radius;
        let height_factor = max(1.0 - abs(local.y / bounding_radius), 0.0);
        let shape_factor = max(1.0 - dist_from_center, 0.0) * height_factor;
        let density = max(raw_density * shape_factor * intensity - 0.2, 0.0);

        if density > 0.001 {
            let temp = clamp(
                (local.y / bounding_radius + 0.5) * 0.7 + raw_density * 0.3,
                0.0, 1.0
            );
            let fc = fire_color_ramp(temp);
            let alpha = min(density * step_size * 8.0, 1.0);
            let contrib = alpha * (1.0 - accum.w);
            accum = vec4<f32>(
                accum.x + fc.x * contrib,
                accum.y + fc.y * contrib,
                accum.z + fc.z * contrib,
                accum.w + contrib,
            );
        }
        t += step_size;
    }
    return accum;
}

// ═══════════════════════════════════════════════════════════════════
// Main shading (iterative — no recursion in WGSL)
// ═══════════════════════════════════════════════════════════════════

fn shade_pixel(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    // Fire check on primary ray
    var fire_color = vec4<f32>(0.0);
    for (var i = 0u; i < u.num_objects; i++) {
        if objects[i].material_type == MAT_FIRE {
            fire_color = march_fire_volume(origin, dir, objects[i]);
            if fire_color.w > 0.01 {
                break;
            }
        }
    }

    // Iterative bounce loop
    var ray_origin = origin;
    var ray_dir = dir;
    var accumulated_color = vec3<f32>(0.0);
    var attenuation = 1.0;
    var alpha = 1.0;

    for (var bounce = 0u; bounce <= u.max_bounces; bounce++) {
        let hit = ray_march(ray_origin, ray_dir, bounce);
        if !hit.hit {
            accumulated_color += u.sky_color.xyz * attenuation;
            if bounce == 0u {
                alpha = u.sky_color.w;
            }
            break;
        }

        let obj = objects[hit.obj_idx];
        let normal = estimate_normal(hit.point, hit.obj_idx);
        let do_shadows = bounce == 0u;
        // Text3D shapes: skip self-shadowing entirely (exclude the text
        // object from shadow rays), use much wider AO sampling to avoid
        // false darkening at the front-face-to-side-face transition.
        let is_text = obj.shape_type == SHAPE_TEXT3D;
        let shadow_bias = select(SURF_DIST * 8.0, SURF_DIST * 64.0, is_text);
        let ao_scale    = select(1.0, 5.0, is_text);

        if obj.material_type == MAT_SOLID {
            let reflectivity = obj.material_params.x;
            let spec_power = obj.material_params.y;
            let base_color = obj.material_color.xyz;

            let shaded = phong_full_ex(hit.point, normal, ray_dir, base_color, spec_power, do_shadows, shadow_bias, ao_scale, is_text);

            if reflectivity > 0.01 && bounce < u.max_bounces {
                // Add (1 - reflectivity) contribution now, continue with reflection
                accumulated_color += shaded * (1.0 - reflectivity) * attenuation;
                attenuation *= reflectivity;
                ray_dir = reflect_dir(ray_dir, normal);
                ray_origin = hit.point + normal * (SURF_DIST * 2.0);
                continue;
            } else {
                accumulated_color += shaded * attenuation;
                break;
            }
        } else if obj.material_type == MAT_WATER {
            let ior = obj.material_params.x;
            let tint = obj.material_color.xyz;
            let cos_theta = max(dot(-ray_dir, normal), 0.0);
            let f = fresnel(cos_theta, ior);

            // For water, do a single reflection (simplified iterative approach)
            let refl_dir = reflect_dir(ray_dir, normal);
            let refl_origin = hit.point + normal * (SURF_DIST * 2.0);
            let refl_hit = ray_march(refl_origin, refl_dir, bounce + 1u);
            var refl_color: vec3<f32>;
            if refl_hit.hit {
                let rn = estimate_normal(refl_hit.point, refl_hit.obj_idx);
                let robj = objects[refl_hit.obj_idx];
                refl_color = phong_full(refl_hit.point, rn, refl_dir,
                                        robj.material_color.xyz,
                                        robj.material_params.y, false);
            } else {
                refl_color = u.sky_color.xyz;
            }

            // Refraction
            let eta = 1.0 / ior;
            let rd = refract_dir(ray_dir, normal, eta);
            var refr_color: vec3<f32>;
            if length(rd) > 0.001 {
                let refr_origin = hit.point - normal * (SURF_DIST * 2.0);
                let refr_hit = ray_march(refr_origin, rd, bounce + 1u);
                if refr_hit.hit {
                    let rrn = estimate_normal(refr_hit.point, refr_hit.obj_idx);
                    let rrobj = objects[refr_hit.obj_idx];
                    refr_color = phong_full(refr_hit.point, rrn, rd,
                                            rrobj.material_color.xyz,
                                            rrobj.material_params.y, false) * tint;
                } else {
                    refr_color = max(u.sky_color.xyz, vec3<f32>(u.ambient)) * tint;
                }
            } else {
                refr_color = refl_color;
            }

            var color = mix(refr_color, refl_color, f);
            let spec = phong_specular_only(hit.point, normal, ray_dir, 128.0);
            color = min(color + spec, vec3<f32>(1.0));
            accumulated_color += color * attenuation;
            break;
        } else if obj.material_type == MAT_CHECKER {
            // Checkerboard material: color_a in material_color, color_b in blend_a_params
            let reflectivity = obj.material_params.x;
            let spec_power = obj.material_params.y;
            let scale = obj.material_params.z;
            let color_a = obj.material_color.xyz;
            let color_b = obj.blend_a_params.xyz;

            // Determine checker pattern from world xz position
            let cx = i32(floor(hit.point.x / scale));
            let cz = i32(floor(hit.point.z / scale));
            var base_color: vec3<f32>;
            if ((cx + cz) & 1) == 0 {
                base_color = color_a;
            } else {
                base_color = color_b;
            }

            let shaded = phong_full_ex(hit.point, normal, ray_dir, base_color, spec_power, do_shadows, shadow_bias, ao_scale, is_text);

            if reflectivity > 0.01 && bounce < u.max_bounces {
                accumulated_color += shaded * (1.0 - reflectivity) * attenuation;
                attenuation *= reflectivity;
                ray_dir = reflect_dir(ray_dir, normal);
                ray_origin = hit.point + normal * (SURF_DIST * 2.0);
                continue;
            } else {
                accumulated_color += shaded * attenuation;
                break;
            }
        } else if obj.material_type == MAT_GLASS {
            // Glass: Fresnel reflection + refraction with optional chromatic dispersion
            let ior = obj.material_params.x;
            let opacity = obj.material_params.y;
            let dispersion = obj.material_params.z;
            let tint = obj.material_color.xyz;
            let cos_theta = max(dot(-ray_dir, normal), 0.0);
            let f = fresnel(cos_theta, ior);

            // Reflection
            let refl_dir_g = reflect_dir(ray_dir, normal);
            let refl_origin_g = hit.point + normal * (SURF_DIST * 2.0);
            let refl_hit_g = ray_march(refl_origin_g, refl_dir_g, bounce + 1u);
            var refl_color_g: vec3<f32>;
            if refl_hit_g.hit {
                let rn = estimate_normal(refl_hit_g.point, refl_hit_g.obj_idx);
                let robj = objects[refl_hit_g.obj_idx];
                // Shade the reflected hit — use its material color
                var rbase = robj.material_color.xyz;
                if robj.material_type == MAT_CHECKER {
                    let sc = robj.material_params.z;
                    let cx2 = i32(floor(refl_hit_g.point.x / sc));
                    let cz2 = i32(floor(refl_hit_g.point.z / sc));
                    if ((cx2 + cz2) & 1) != 0 {
                        rbase = robj.blend_a_params.xyz;
                    }
                } else if robj.material_type == MAT_RAINBOW {
                    let rlocal = refl_hit_g.point - robj.position;
                    let rangle = atan2(rlocal.z, rlocal.x);
                    let rhue = rangle / 6.283185 + 0.5 + robj.material_params.z / 6.283185;
                    rbase = hsl_to_rgb(rhue, robj.material_params.x, robj.material_params.y);
                }
                refl_color_g = phong_full(refl_hit_g.point, rn, refl_dir_g,
                                          rbase, robj.material_params.y, false);
            } else {
                refl_color_g = max(u.sky_color.xyz, vec3<f32>(u.ambient));
            }

            // Refraction (with optional chromatic dispersion)
            var refr_color_g: vec3<f32>;
            if dispersion > 0.001 {
                // Per-channel refraction for prismatic effect
                let refr_origin_g = hit.point - normal * (SURF_DIST * 2.0);
                let ior_r = 1.0 / (ior - dispersion);
                let ior_gg = 1.0 / ior;
                let ior_b = 1.0 / (ior + dispersion);

                let rd_r = refract_dir(ray_dir, normal, ior_r);
                let rd_gg = refract_dir(ray_dir, normal, ior_gg);
                let rd_b = refract_dir(ray_dir, normal, ior_b);

                var cr = 0.0; var cg = 0.0; var cb = 0.0;

                // Red channel
                if length(rd_r) > 0.001 {
                    let rh = ray_march(refr_origin_g, rd_r, bounce + 1u);
                    if rh.hit {
                        let rn2 = estimate_normal(rh.point, rh.obj_idx);
                        let ro2 = objects[rh.obj_idx];
                        var rb = ro2.material_color.xyz;
                        if ro2.material_type == MAT_RAINBOW {
                            let rl = rh.point - ro2.position;
                            let ra = atan2(rl.z, rl.x);
                            let rhu = ra / 6.283185 + 0.5 + ro2.material_params.z / 6.283185;
                            rb = hsl_to_rgb(rhu, ro2.material_params.x, ro2.material_params.y);
                        }
                        let sc = phong_full(rh.point, rn2, rd_r, rb, ro2.material_params.y, false);
                        cr = sc.x * tint.x;
                    } else { cr = max(u.sky_color.x, u.ambient) * tint.x; }
                } else { cr = refl_color_g.x; }

                // Green channel
                if length(rd_gg) > 0.001 {
                    let rh = ray_march(refr_origin_g, rd_gg, bounce + 1u);
                    if rh.hit {
                        let rn2 = estimate_normal(rh.point, rh.obj_idx);
                        let ro2 = objects[rh.obj_idx];
                        var rb = ro2.material_color.xyz;
                        if ro2.material_type == MAT_RAINBOW {
                            let rl = rh.point - ro2.position;
                            let ra = atan2(rl.z, rl.x);
                            let rhu = ra / 6.283185 + 0.5 + ro2.material_params.z / 6.283185;
                            rb = hsl_to_rgb(rhu, ro2.material_params.x, ro2.material_params.y);
                        }
                        let sc = phong_full(rh.point, rn2, rd_gg, rb, ro2.material_params.y, false);
                        cg = sc.y * tint.y;
                    } else { cg = max(u.sky_color.y, u.ambient) * tint.y; }
                } else { cg = refl_color_g.y; }

                // Blue channel
                if length(rd_b) > 0.001 {
                    let rh = ray_march(refr_origin_g, rd_b, bounce + 1u);
                    if rh.hit {
                        let rn2 = estimate_normal(rh.point, rh.obj_idx);
                        let ro2 = objects[rh.obj_idx];
                        var rb = ro2.material_color.xyz;
                        if ro2.material_type == MAT_RAINBOW {
                            let rl = rh.point - ro2.position;
                            let ra = atan2(rl.z, rl.x);
                            let rhu = ra / 6.283185 + 0.5 + ro2.material_params.z / 6.283185;
                            rb = hsl_to_rgb(rhu, ro2.material_params.x, ro2.material_params.y);
                        }
                        let sc = phong_full(rh.point, rn2, rd_b, rb, ro2.material_params.y, false);
                        cb = sc.z * tint.z;
                    } else { cb = max(u.sky_color.z, u.ambient) * tint.z; }
                } else { cb = refl_color_g.z; }

                refr_color_g = vec3<f32>(cr, cg, cb);
            } else {
                // Single IOR refraction
                let eta = 1.0 / ior;
                let rd = refract_dir(ray_dir, normal, eta);
                if length(rd) > 0.001 {
                    let refr_origin_g = hit.point - normal * (SURF_DIST * 2.0);
                    let refr_hit_g = ray_march(refr_origin_g, rd, bounce + 1u);
                    if refr_hit_g.hit {
                        let rrn = estimate_normal(refr_hit_g.point, refr_hit_g.obj_idx);
                        let rrobj = objects[refr_hit_g.obj_idx];
                        refr_color_g = phong_full(refr_hit_g.point, rrn, rd,
                                                  rrobj.material_color.xyz,
                                                  rrobj.material_params.y, false) * tint;
                    } else {
                        refr_color_g = max(u.sky_color.xyz, vec3<f32>(u.ambient)) * tint;
                    }
                } else {
                    refr_color_g = refl_color_g;
                }
            }

            // Blend via Fresnel
            var glass_color = mix(refr_color_g, refl_color_g, f);
            // Opacity blend
            if opacity > 0.001 {
                glass_color = mix(glass_color, tint, opacity);
            }
            // Specular highlights
            let spec_g = phong_specular_only(hit.point, normal, ray_dir, 128.0);
            glass_color = min(glass_color + spec_g, vec3<f32>(1.0));
            accumulated_color += glass_color * attenuation;
            break;
        } else if obj.material_type == MAT_RAINBOW {
            // Rainbow: HSL mapping from local object space
            let local_r = hit.point - obj.position;
            let hue_offset = obj.material_params.z;
            var hue: f32;
            if obj.shape_type == SHAPE_TEXT3D {
                // Text3D: use x-position for smooth left-to-right gradient
                let text_width = obj.shape_params.y;
                let half_w = text_width * 0.5;
                hue = clamp((local_r.x + half_w) / max(text_width, 0.001), 0.0, 1.0) + hue_offset / 6.283185;
            } else {
                // Other shapes: angular mapping
                let angle = atan2(local_r.z, local_r.x);
                hue = angle / 6.283185 + 0.5 + hue_offset / 6.283185;
            }
            let base_color = hsl_to_rgb(hue, obj.material_params.x, obj.material_params.y);
            let spec_power = obj.material_params.w;

            let shaded = phong_full_ex(hit.point, normal, ray_dir, base_color, spec_power, do_shadows, shadow_bias, ao_scale, is_text);
            accumulated_color += shaded * attenuation;
            break;
        } else if obj.material_type == MAT_SUBSURFACE {
            // Subsurface scattering: Phong front-lighting + SDF thickness back-illumination
            let sss_thickness = obj.material_params.x;
            let spec_power = obj.material_params.y;
            let surface_color = obj.material_color.xyz;
            let scatter_color = obj.blend_a_params.xyz;

            let front_shaded = phong_full_ex(hit.point, normal, ray_dir, surface_color, spec_power, do_shadows, shadow_bias, ao_scale, is_text);

            // SSS: for each light, estimate thickness via SDF and add back-illumination
            var sss_accum = vec3<f32>(0.0);
            for (var li = 0u; li < u.num_lights; li++) {
                let light = lights[li];
                let to_light = normalize(light.position - hit.point);
                let sample_pt = hit.point + to_light * sss_thickness;
                let thickness_d = scene_sdf(sample_pt).dist;
                let sss_factor = exp(-abs(thickness_d) * 3.0);
                let n_dot_l_inv = max(dot(-normal, to_light), 0.0);
                let contribution = sss_factor * n_dot_l_inv * light.intensity;
                sss_accum += scatter_color * light.color.xyz * contribution;
            }

            accumulated_color += min(front_shaded + sss_accum, vec3<f32>(1.0)) * attenuation;
            break;
        } else {
            // Fire surface hit — faint glow
            let glow = fire_color_ramp(0.3) * 0.5;
            accumulated_color += glow * attenuation;
            break;
        }
    }

    // Apply fire compositing (primary ray only)
    if fire_color.w > 0.01 {
        let inv_a = 1.0 - fire_color.w;
        accumulated_color = fire_color.xyz + accumulated_color * inv_a;
    }

    // Volumetric god rays (primary ray only)
    if u.god_rays != 0u {
        var god_ray_accum = vec3<f32>(0.0);
        let max_march_t = 20.0; // don't march beyond reasonable distance
        let gr_step = max_march_t / f32(u.god_ray_samples);
        for (var gi = 0u; gi < u.god_ray_samples; gi++) {
            let gt = gr_step * (f32(gi) + 0.5);
            let gp = origin + dir * gt;
            let gd = scene_sdf(gp).dist;
            if gd < 0.0 {
                continue; // inside geometry
            }
            for (var li = 0u; li < u.num_lights; li++) {
                let light = lights[li];
                let to_light = light.position - gp;
                let light_dist = length(to_light);
                let light_dir = to_light / light_dist;
                let shadow = soft_shadow(gp, light_dir, light_dist, 8u);
                if shadow > 0.01 {
                    let contribution = u.god_ray_density * gr_step * shadow * light.intensity * 0.02;
                    god_ray_accum += light.color.xyz * contribution;
                }
            }
        }
        accumulated_color = min(accumulated_color + god_ray_accum, vec3<f32>(1.0));
    }

    return vec4<f32>(accumulated_color, alpha);
}

// ═══════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= u.width || y >= u.height {
        return;
    }

    // Generate camera ray
    let ndc_x = (2.0 * (f32(x) + 0.5) / f32(u.width) - 1.0) * u.aspect * u.fov_scale;
    let ndc_y = (1.0 - 2.0 * (f32(y) + 0.5) / f32(u.height)) * u.fov_scale;
    let dir = normalize(u.cam_right * ndc_x + u.cam_up * ndc_y + u.cam_forward);

    let color = shade_pixel(u.eye, dir);

    // Gamma correction (linear → sRGB)
    let gamma = 1.0 / 2.2;
    let corrected = vec3<f32>(
        pow(clamp(color.x, 0.0, 1.0), gamma),
        pow(clamp(color.y, 0.0, 1.0), gamma),
        pow(clamp(color.z, 0.0, 1.0), gamma),
    );

    // Pack RGBA8 into u32
    let r = u32(corrected.x * 255.0);
    let g = u32(corrected.y * 255.0);
    let b = u32(corrected.z * 255.0);
    let a = u32(clamp(color.w, 0.0, 1.0) * 255.0);
    let pixel = r | (g << 8u) | (b << 16u) | (a << 24u);

    output[y * u.width + x] = pixel;
}
