// Unified SDF shape shader for 2D rasterization.
//
// Renders circles, rectangles (optionally rounded), and ellipses using
// signed distance fields. Each instance defines a shape via a type
// discriminant, position, size, color, and corner radius.
//
// Shape types (encoded in shape_type field):
//   0 = Circle:    pos = center, size = (radius, radius, 0, 0)
//   1 = Rectangle: pos = top-left, size = (width, height, corner_radius, 0)
//   2 = Ellipse:   pos = center, size = (rx, ry, rotation_rad, 0)

struct InstanceInput {
    @location(0) pos: vec2<f32>,           // center or top-left (pixels)
    @location(1) size: vec4<f32>,          // shape-type-dependent params
    @location(2) fill_color: vec4<f32>,    // RGBA [0,1]
    @location(3) stroke_color: vec4<f32>,  // RGBA [0,1]
    @location(4) stroke_width: f32,        // stroke width in pixels
    @location(5) shape_type: u32,          // 0=circle, 1=rect, 2=ellipse
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,     // position relative to shape center
    @location(1) fill_color: vec4<f32>,
    @location(2) stroke_color: vec4<f32>,
    @location(3) half_size: vec2<f32>,     // half-width, half-height of bounding box
    @location(4) params: vec3<f32>,        // (corner_radius, stroke_width, rotation)
    @location(5) @interpolate(flat) shape_type: u32,
};

struct Uniforms {
    viewport: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    inst: InstanceInput,
) -> VertexOutput {
    // Unit quad: 6 vertices for 2 triangles
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let uv = positions[vertex_index];
    let shape_type = inst.shape_type;
    let stroke_width = inst.stroke_width;

    var center: vec2<f32>;
    var half_size: vec2<f32>;
    var corner_radius: f32 = 0.0;
    var rotation: f32 = 0.0;

    switch shape_type {
        case 0u: {
            // Circle: pos = center, size.x = radius
            center = inst.pos;
            let r = inst.size.x;
            half_size = vec2<f32>(r, r);
        }
        case 1u: {
            // Rectangle: pos = top-left, size = (w, h, corner_radius, 0)
            let w = inst.size.x;
            let h = inst.size.y;
            center = inst.pos + vec2<f32>(w * 0.5, h * 0.5);
            half_size = vec2<f32>(w * 0.5, h * 0.5);
            corner_radius = inst.size.z;
        }
        default: {
            // Ellipse: pos = center, size = (rx, ry, rotation, 0)
            center = inst.pos;
            half_size = vec2<f32>(inst.size.x, inst.size.y);
            rotation = inst.size.z;
        }
    }

    // Expand quad by half_size + stroke + 1px for AA
    let expand = half_size + vec2<f32>(stroke_width + 1.5, stroke_width + 1.5);
    let local_pos = uv * expand;

    // Apply rotation for ellipses
    var pixel_pos: vec2<f32>;
    if abs(rotation) > 0.001 {
        let cos_r = cos(rotation);
        let sin_r = sin(rotation);
        let rotated = vec2<f32>(
            local_pos.x * cos_r - local_pos.y * sin_r,
            local_pos.x * sin_r + local_pos.y * cos_r,
        );
        pixel_pos = center + rotated;
    } else {
        pixel_pos = center + local_pos;
    }

    // Pixel to NDC
    let ndc_x = pixel_pos.x / uniforms.viewport.x * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel_pos.y / uniforms.viewport.y * 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local_pos = local_pos;
    out.fill_color = inst.fill_color;
    out.stroke_color = inst.stroke_color;
    out.half_size = half_size;
    out.params = vec3<f32>(corner_radius, stroke_width, rotation);
    out.shape_type = shape_type;
    return out;
}

// Signed distance to a rounded rectangle centered at origin.
fn sd_round_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(radius, radius);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let shape_type = in.shape_type;
    let corner_radius = in.params.x;
    let stroke_width = in.params.y;
    let rotation = in.params.z;

    // Compute local position (undo rotation if applied)
    var lp = in.local_pos;
    if abs(rotation) > 0.001 {
        let cos_r = cos(-rotation);
        let sin_r = sin(-rotation);
        lp = vec2<f32>(
            in.local_pos.x * cos_r - in.local_pos.y * sin_r,
            in.local_pos.x * sin_r + in.local_pos.y * cos_r,
        );
    }

    var dist: f32;

    switch shape_type {
        case 0u: {
            // Circle SDF
            dist = length(lp) - in.half_size.x;
        }
        case 1u: {
            // Rounded rectangle SDF
            dist = sd_round_rect(lp, in.half_size, corner_radius);
        }
        default: {
            // Ellipse SDF (approximation via normalized-space circle)
            let normalized = lp / in.half_size;
            let norm_dist = length(normalized) - 1.0;
            dist = norm_dist * min(in.half_size.x, in.half_size.y);
        }
    }

    // Outside shape + stroke: discard
    if dist > stroke_width + 0.5 {
        discard;
    }

    // Determine color: fill interior, stroke the border
    var color: vec4<f32>;
    if stroke_width > 0.0 && dist > -stroke_width * 0.5 {
        // In the stroke band
        let stroke_alpha = clamp(stroke_width * 0.5 + 0.5 - abs(dist), 0.0, 1.0);
        color = vec4<f32>(in.stroke_color.rgb, in.stroke_color.a * stroke_alpha);
    } else if in.fill_color.a > 0.0 {
        // Interior fill
        let fill_alpha = clamp(0.5 - dist, 0.0, 1.0);
        color = vec4<f32>(in.fill_color.rgb, in.fill_color.a * fill_alpha);
    } else if stroke_width > 0.0 {
        // Stroke only (no fill)
        let stroke_alpha = clamp(stroke_width + 0.5 - dist, 0.0, 1.0)
                         * clamp(dist + 0.5, 0.0, 1.0);
        color = vec4<f32>(in.stroke_color.rgb, in.stroke_color.a * stroke_alpha);
    } else {
        discard;
    }

    if color.a <= 0.0 {
        discard;
    }

    return color;
}
