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
    _pad3a: u32,
    _pad3b: u32,
    _pad3c: u32,
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

// Material types (discriminant)
const MAT_SOLID: u32 = 0u;
const MAT_WATER: u32 = 1u;
const MAT_FIRE: u32 = 2u;
const MAT_CHECKER: u32 = 3u;
const MAT_GLASS: u32 = 4u;
const MAT_RAINBOW: u32 = 5u;

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
    rotation_cos_y: f32,  // cos of Y-axis rotation (1.0 = no rotation)
    rotation_sin_y: f32,  // sin of Y-axis rotation (0.0 = no rotation)
    _pad2: f32,
};

struct GpuLight {
    position: vec3<f32>,
    intensity: f32,
    color: vec4<f32>,
};

// ═══════════════════════════════════════════════════════════════════
// Bindings
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> objects: array<GpuObject>;
@group(0) @binding(2) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const MAX_DIST: f32 = 50.0;
const SURF_DIST: f32 = 0.002;
const NORMAL_EPS: f32 = 0.002;
const OMEGA: f32 = 1.6;
const RELAX_DIST: f32 = 0.1;
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
    if obj.shape_type == SHAPE_SMOOTH_BLEND {
        let sub_a_type = u32(obj.shape_params.y);
        let sub_b_type = u32(obj.shape_params.z);
        let k = obj.shape_params.x;
        let da = eval_sub_shape(sub_a_type, obj.blend_a_params, local);
        let db = eval_sub_shape(sub_b_type, obj.blend_b_params, local - obj.blend_b_offset);
        return smooth_min(da, db, k);
    }
    return eval_sub_shape(obj.shape_type, obj.shape_params, local);
}

fn object_sdf(obj: GpuObject, point: vec3<f32>) -> f32 {
    var local = point - obj.position;

    // Apply inverse Y-axis rotation if rotation is set (sin != 0)
    if abs(obj.rotation_sin_y) > 0.0001 || abs(obj.rotation_cos_y - 1.0) > 0.0001 {
        let rx = local.x * obj.rotation_cos_y - local.z * obj.rotation_sin_y;
        let rz = local.x * obj.rotation_sin_y + local.z * obj.rotation_cos_y;
        local = vec3<f32>(rx, local.y, rz);
    }

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
            return HitResult(true, p, res.idx, t);
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
    var t = SURF_DIST * 4.0;
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

fn ambient_occlusion(hit: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occ = 0.0;
    var scale = 1.0;
    for (var i = 1; i <= 5; i++) {
        let dist = 0.02 * f32(i);
        let d = scene_sdf(hit + normal * dist).dist;
        occ += (dist - d) * scale;
        scale *= 0.75;
    }
    return max(1.0 - clamp(occ, 0.0, 1.0), 0.0);
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
    var ao = 1.0;
    if do_shadows {
        ao = ambient_occlusion(hit, normal);
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
        if do_shadows {
            let shadow_origin = hit + normal * (SURF_DIST * 4.0);
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

fn shade_pixel(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
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

    for (var bounce = 0u; bounce <= u.max_bounces; bounce++) {
        let hit = ray_march(ray_origin, ray_dir, bounce);
        if !hit.hit {
            accumulated_color += u.sky_color.xyz * attenuation;
            break;
        }

        let obj = objects[hit.obj_idx];
        let normal = estimate_normal(hit.point, hit.obj_idx);
        let do_shadows = bounce == 0u;

        if obj.material_type == MAT_SOLID {
            let reflectivity = obj.material_params.x;
            let spec_power = obj.material_params.y;
            let base_color = obj.material_color.xyz;

            let shaded = phong_full(hit.point, normal, ray_dir, base_color, spec_power, do_shadows);

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
                    refr_color = u.sky_color.xyz * tint;
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

            let shaded = phong_full(hit.point, normal, ray_dir, base_color, spec_power, do_shadows);

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
                refl_color_g = u.sky_color.xyz;
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
                    } else { cr = u.sky_color.x * tint.x; }
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
                    } else { cg = u.sky_color.y * tint.y; }
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
                    } else { cb = u.sky_color.z * tint.z; }
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
                        refr_color_g = u.sky_color.xyz * tint;
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
            // Rainbow: angular HSL mapping from local object space
            let local_r = hit.point - obj.position;
            let angle = atan2(local_r.z, local_r.x);
            let hue_offset = obj.material_params.z;
            let hue = angle / 6.283185 + 0.5 + hue_offset / 6.283185;
            let base_color = hsl_to_rgb(hue, obj.material_params.x, obj.material_params.y);
            let spec_power = obj.material_params.w;

            let shaded = phong_full(hit.point, normal, ray_dir, base_color, spec_power, do_shadows);
            accumulated_color += shaded * attenuation;
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

    return accumulated_color;
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
    let pixel = r | (g << 8u) | (b << 16u) | (255u << 24u);

    output[y * u.width + x] = pixel;
}
