// Fullscreen-triangle overlay shader.
//
// Draws a textured quad over the entire viewport using a 3-vertex
// fullscreen triangle (no vertex buffer needed). The fragment shader
// samples an RGBA overlay texture with alpha blending so transparent
// regions show the terminal content beneath.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate a fullscreen triangle from vertex index (0, 1, 2).
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    // This covers the entire clip space with a single triangle.
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Map clip coords to UV: (-1,-1) → (0,1), (1,1) → (1,0)
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@group(0) @binding(0)
var overlay_texture: texture_2d<f32>;
@group(0) @binding(1)
var overlay_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(overlay_texture, overlay_sampler, in.uv);
}
