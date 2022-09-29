// Vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Lights {
    positions: array<vec4<f32>, 32>,
    colors: array<vec4<f32>, 32>,
    count: i32,
}
@group(2) @binding(0)
var<uniform> lights: Lights;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
}
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
}

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0)@binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    var diffuse_color = vec3(0.0, 0.0, 0.0);
    var specular_color = vec3(0.0, 0.0, 0.0);
    var i: i32 = 0;
    loop {
        if (i >= lights.count) {break;}
        // Direct light
        let light_v = lights.positions[i].xyz - in.world_position;
        let light_d = length(light_v);
        let light_falloff = lights.colors[i].w / (light_d * light_d);
        let light_dir = normalize(light_v);
        let view_dir = normalize(camera.view_pos.xyz - in.world_position);
        let half_dir = normalize(view_dir + light_dir);

        let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0) * light_falloff;
        diffuse_color +=  diffuse_strength * lights.colors[i].xyz;

        let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0) * light_falloff;
        specular_color += diffuse_strength * specular_strength * lights.colors[i].xyz;
        i++;
    }
    // Ambient light
    let ambient_strength = 0.025;
    let ambient_color = vec3(1.0, 1.0, 1.0) * ambient_strength;

    let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    return vec4<f32>(result, object_color.a);
}