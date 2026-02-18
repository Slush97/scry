// Colored triangle mesh shader for GPU tessellation.
//
// Renders tessellated geometry (paths, arcs, polygons) that has been
// converted to triangle meshes on the CPU via ear-clipping.

struct Uniforms {
    viewport: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let ndc = vec2<f32>(
        in.position.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - in.position.y / uniforms.viewport.y * 2.0,
    );
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
