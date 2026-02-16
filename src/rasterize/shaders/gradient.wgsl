// Gradient shader for 2D rasterization.
//
// Renders a rectangle filled with a linear or radial gradient.
// Supports up to 8 color stops passed via uniform buffer.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pixel_pos: vec2<f32>,   // position in pixel coords
};

struct GradientStop {
    color: vec4<f32>,    // RGBA
    position: f32,       // [0, 1]
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

struct GradientUniforms {
    viewport: vec2<f32>,
    rect_pos: vec2<f32>,     // top-left corner
    rect_size: vec2<f32>,    // width, height
    grad_start: vec2<f32>,   // linear: start point, radial: center
    grad_end: vec2<f32>,     // linear: end point, radial: (radius, 0)
    grad_type: f32,          // 0 = linear, 1 = radial
    num_stops: f32,
    _pad: vec2<f32>,
    stops: array<GradientStop, 8>,
};

@group(0) @binding(0)
var<uniform> grad: GradientUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-rect quad: 2 triangles from 6 vertices
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );

    let uv = positions[vertex_index];
    let pixel_pos = grad.rect_pos + uv * grad.rect_size;

    let ndc_x = pixel_pos.x / grad.viewport.x * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel_pos.y / grad.viewport.y * 2.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.pixel_pos = pixel_pos;
    return out;
}

fn sample_gradient(t: f32) -> vec4<f32> {
    let num = i32(grad.num_stops);

    // Clamp to first/last stop
    if num <= 0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if num == 1 || t <= grad.stops[0].position {
        return grad.stops[0].color;
    }

    // Find the two stops bracketing t
    for (var i = 1; i < num; i = i + 1) {
        if t <= grad.stops[i].position {
            let prev = grad.stops[i - 1];
            let curr = grad.stops[i];
            let range = curr.position - prev.position;
            if range < 0.0001 {
                return curr.color;
            }
            let local_t = (t - prev.position) / range;
            return mix(prev.color, curr.color, local_t);
        }
    }

    return grad.stops[num - 1].color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var t: f32;

    if grad.grad_type < 0.5 {
        // Linear gradient
        let dir = grad.grad_end - grad.grad_start;
        let len_sq = dot(dir, dir);
        if len_sq < 0.0001 {
            t = 0.0;
        } else {
            t = dot(in.pixel_pos - grad.grad_start, dir) / len_sq;
        }
    } else {
        // Radial gradient
        let dist = length(in.pixel_pos - grad.grad_start);
        let radius = grad.grad_end.x;
        if radius < 0.0001 {
            t = 0.0;
        } else {
            t = dist / radius;
        }
    }

    t = clamp(t, 0.0, 1.0);
    return sample_gradient(t);
}
