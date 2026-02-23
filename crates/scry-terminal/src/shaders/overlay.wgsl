// Region-scoped overlay shader.
//
// Draws a textured quad within a specific screen-space region using a
// 3-vertex fullscreen triangle. The fragment shader clips to the region
// bounds and maps UVs so the texture fills exactly the region rectangle.
// Transparent regions show the terminal content beneath via alpha blending.

struct RegionUniforms {
    // Screen size in pixels.
    screen_size: vec2<f32>,
    // Region top-left in pixels.
    region_origin: vec2<f32>,
    // Region size in pixels.
    region_size: vec2<f32>,
    // Global alpha multiplier (0.0 = invisible, 1.0 = fully opaque).
    global_alpha: f32,
    _pad: f32,
};

@group(0) @binding(0)
var overlay_texture: texture_2d<f32>;
@group(0) @binding(1)
var overlay_sampler: sampler;
@group(0) @binding(2)
var<uniform> region: RegionUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate a fullscreen triangle from vertex index (0, 1, 2).
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Map clip coords to pixel coords for region mapping
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert UV (0..1 of screen) to pixel position
    let pixel_pos = in.uv * region.screen_size;

    // Check if pixel is within the overlay region
    let local = pixel_pos - region.region_origin;
    if (local.x < 0.0 || local.y < 0.0 || local.x >= region.region_size.x || local.y >= region.region_size.y) {
        discard;
    }

    // Map local position to texture UV
    let tex_uv = local / region.region_size;
    var color = textureSample(overlay_texture, overlay_sampler, tex_uv);
    color = vec4<f32>(color.rgb * region.global_alpha, color.a * region.global_alpha);
    return color;
}
