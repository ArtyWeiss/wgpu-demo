// Vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Lights {
    positions: array<vec3<f32>, 32>,
    colors: array<vec4<f32>, 32>,
    count: i32,
}
@group(1) @binding(0)
var<uniform> lights: Lights;

struct VertexInput {
    @builtin(instance_index) instance_id: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main( model: VertexInput ) -> VertexOutput {
    let scale = 0.15;
    var out: VertexOutput;
    let light_index = model.instance_id;
    out.clip_position = camera.view_proj * vec4<f32>(model.position * scale + lights.positions[light_index], 1.0);
    out.color = lights.colors[light_index].xyz * lights.colors[light_index].w;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}