// Line shader — anti-aliased line segment rendering.
//
// Each line segment is defined by two endpoints (screen-space pixels)
// with a shared color and width. The vertex shader expands each segment
// into a screen-aligned quad with the specified width + 1px for AA.

struct VertexInput {
    @location(0) position: vec2<f32>,  // screen-space pixel position
    @location(1) normal: vec2<f32>,    // perpendicular offset direction
    @location(2) color: vec4<f32>,     // RGBA
    @location(3) line_width: f32,      // half-width in pixels
    @location(4) edge_dist: f32,       // signed distance from line center
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) edge_dist: f32,
    @location(2) half_width: f32,
};

struct Uniforms {
    viewport: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Expand position along normal by (half_width + 0.5) for AA
    let expand = in.line_width + 0.5;
    let pixel_pos = in.position + in.normal * expand * in.edge_dist;

    // Pixel to NDC
    let ndc_x = pixel_pos.x / uniforms.viewport.x * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel_pos.y / uniforms.viewport.y * 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = in.color;
    out.edge_dist = in.edge_dist * expand;
    out.half_width = in.line_width;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Anti-aliased edge
    let dist = abs(in.edge_dist);
    let alpha = clamp(in.half_width + 0.5 - dist, 0.0, 1.0);

    if alpha <= 0.0 {
        discard;
    }

    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
