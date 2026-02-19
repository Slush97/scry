// Point shader — instanced circle rendering for 3D scatter plots.
//
// Each instance is a single point defined by:
//   - screen position (x, y) in pixels
//   - radius in pixels
//   - depth (0..1) for attenuation
//   - RGBA color
//
// A unit quad (4 vertices) is instanced per point.
// The fragment shader computes a circle SDF with 1px anti-aliased edge.

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct InstanceInput {
    @location(0) pos_size: vec4<f32>,  // x, y, radius, depth
    @location(1) color: vec4<f32>,      // r, g, b, a
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,         // [-1, 1] within quad
    @location(1) color: vec4<f32>,
    @location(2) radius: f32,
    @location(3) depth_val: f32,
};

struct Uniforms {
    viewport: vec2<f32>,  // width, height in pixels
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    // Unit quad: 2 triangles from 6 vertices
    // 0: (-1,-1)  1: (1,-1)  2: (-1,1)  3: (1,-1)  4: (1,1)  5: (-1,1)
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let uv = positions[vert.vertex_index];

    let center_x = inst.pos_size.x;
    let center_y = inst.pos_size.y;
    let radius = inst.pos_size.z;
    let depth = inst.pos_size.w;

    // Depth-based size attenuation (matches CPU: 1.0 - depth * 0.3)
    let depth_factor = 1.0 - clamp(depth, 0.0, 1.0) * 0.3;
    let effective_radius = radius * depth_factor;

    // Expand quad by radius + 1px for AA
    let expand = effective_radius + 1.0;
    let pixel_pos = vec2<f32>(center_x, center_y) + uv * expand;

    // Convert pixel coords to NDC: x: [0,w] -> [-1,1], y: [0,h] -> [1,-1]
    let ndc_x = pixel_pos.x / uniforms.viewport.x * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel_pos.y / uniforms.viewport.y * 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = uv;
    out.color = inst.color;
    out.radius = effective_radius;
    out.depth_val = depth;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let expand = in.radius + 1.0;
    let dist = length(in.uv * expand);

    // Outside circle: discard
    if dist > in.radius + 0.5 {
        discard;
    }

    // Anti-aliased edge: 1px feather
    let alpha = clamp(in.radius + 0.5 - dist, 0.0, 1.0);

    // Border ring: darken outer 0.8px
    let border_start = max(in.radius - 0.8, 0.0);
    let border_alpha = 0.3 + 0.4 * (1.0 - clamp(in.depth_val, 0.0, 1.0));
    var color = in.color.rgb;
    if dist > border_start {
        let border_mix = clamp((dist - border_start) / 0.8, 0.0, 1.0);
        color = color * (1.0 - border_mix * 0.6);
    }

    return vec4<f32>(color, in.color.a * alpha);
}
