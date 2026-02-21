// Cell background shader — instanced rendering of colored rectangles.
// Each instance is a cell with a non-default background color.

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    // Instance data
    @location(0) cell_pos: vec2<f32>,   // top-left corner (pixels)
    @location(1) cell_size: vec2<f32>,  // width, height (pixels)
    @location(2) color: vec4<f32>,      // RGBA background color
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Uniforms {
    screen_size: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Generate quad vertices from vertex_index (0..5 for two triangles)
    var pos: vec2<f32>;
    switch input.vertex_index {
        case 0u: { pos = vec2(0.0, 0.0); }
        case 1u: { pos = vec2(1.0, 0.0); }
        case 2u: { pos = vec2(0.0, 1.0); }
        case 3u: { pos = vec2(1.0, 0.0); }
        case 4u: { pos = vec2(1.0, 1.0); }
        case 5u: { pos = vec2(0.0, 1.0); }
        default: { pos = vec2(0.0, 0.0); }
    }

    // Scale to cell size and position
    let pixel_pos = input.cell_pos + pos * input.cell_size;

    // Convert from pixel coordinates to NDC (-1..1)
    let ndc = vec2(
        pixel_pos.x / uniforms.screen_size.x * 2.0 - 1.0,
        1.0 - pixel_pos.y / uniforms.screen_size.y * 2.0,
    );

    var output: VertexOutput;
    output.position = vec4(ndc, 0.0, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
