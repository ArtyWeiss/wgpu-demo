use std::iter;

use cgmath::prelude::*;
use instant;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use model::{DrawLight, DrawModel, Vertex};

use crate::character_system::Character;
use crate::GameState;

mod model;
mod texture;
mod resources;
mod camera;

const NUM_INSTANCES_PER_ROW: u32 = 4;
const MAX_LIGHTS_COUNT: usize = 32;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CharacterUniform {
    model: [[f32; 4]; 4],
}

impl CharacterUniform {
    fn new() -> Self {
        Self {
            model: cgmath::Matrix4::identity().into(),
        }
    }
    fn update_matrix(&mut self, position: cgmath::Vector3<f32>) {
        self.model = cgmath::Matrix4::from_translation(position).into();
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightsUniform {
    positions: [[f32; 4]; MAX_LIGHTS_COUNT],
    colors: [[f32; 4]; MAX_LIGHTS_COUNT],
    // x,y,z - color; w - power
    count: u32,
    _padding2: [u32; 3], // uniforms requiring 16 byte spacing
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model = cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // model mat4x4 ===============================================
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // normal mat3x3 ==============================================
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub(crate) struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    camera: camera::Camera,
    projection: camera::Projection,
    pub(crate) camera_controller: camera::CameraController,
    pub(crate) mouse_pressed: bool,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    depth_texture: texture::Texture,
    pub(crate) size: winit::dpi::PhysicalSize<u32>,

    #[allow(dead_code)]
    light_uniform: LightsUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,

    #[allow(dead_code)]
    character_uniform: CharacterUniform,
    character_buffer: wgpu::Buffer,
    character_bind_group: wgpu::BindGroup,
    character_render_pipeline: wgpu::RenderPipeline,
    character_model: model::Model,

    models: Vec<model::Model>,
    light_source_model: model::Model,
    instances_data: Vec<(u32, wgpu::Buffer)>,
    render_pipeline: wgpu::RenderPipeline,
}

impl State {
    pub(crate) async fn new(window: &Window, game_state: &GameState) -> Self {
        // Initialize surface =============================================================================
        let size = window.inner_size();
        log::warn!("WGPU setup");
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        log::warn!("device and queue");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();
        log::warn!("Surface");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Load assets and create instances =======================================================================
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let instances_data = Self::create_instances(&device);
        let models = Self::load_models(&device, &queue, &texture_bind_group_layout).await.unwrap();
        let light_source_model = resources::load_model(
            "sphere.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        ).await.unwrap();
        // Load character and create uniform ====================================================================
        let character_model = resources::load_model(
            "cylinder.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        ).await.unwrap();
        let (character_uniform, character_buffer, character_bind_group_layout, character_bind_group) = Self::create_character_data(&device);

        // Create camera and camera data ========================================================================
        let camera = camera::Camera::new((0.0, -10.0, 5.0), cgmath::Rad(0.0), cgmath::Rad(0.0));
        let projection = camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = camera::CameraController::new(4.0, 0.5);
        let (camera_uniform, camera_buffer, camera_bind_group_layout, camera_bind_group) = Self::create_camera_data(&device, &camera, &projection);

        // Create light data ====================================================================================
        let (light_uniform, light_buffer, light_bind_group_layout, light_bind_group) = Self::create_light_data(&device);

        // Create render pipelines ==============================================================================
        let render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            Self::create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &light_bind_group_layout
                ],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            Self::create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        let character_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Character Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &character_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Character Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("character.wgsl").into()),
            };
            Self::create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        Self {
            surface,
            device,
            queue,
            config,
            mouse_pressed: false,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            size,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            models,
            light_source_model,
            instances_data,
            render_pipeline,
            character_uniform,
            character_buffer,
            character_bind_group,
            character_render_pipeline,
            character_model,
        }
    }

    fn create_camera_data(device: &wgpu::Device, camera: &camera::Camera, projection: &camera::Projection) -> (CameraUniform, wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        (camera_uniform, camera_buffer, camera_bind_group_layout, camera_bind_group)
    }

    fn create_character_data(device: &wgpu::Device) -> (CharacterUniform, wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let mut character_uniform = CharacterUniform::new();
        let character_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Character VB"),
                contents: bytemuck::cast_slice(&[character_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        let character_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });
        let character_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &character_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: character_buffer.as_entire_binding(),
            }],
            label: None,
        });
        (character_uniform, character_buffer, character_bind_group_layout, character_bind_group)
    }

    fn create_light_data(device: &wgpu::Device) -> (LightsUniform, wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let light_uniform = LightsUniform {
            positions: [[3.75, 0.0, 0.8, 0.0]; MAX_LIGHTS_COUNT],
            // positions: [[0.0, 0.0, 0.0, 0.0]; MAX_LIGHTS_COUNT],
            colors: [[0.0, 0.0, 0.0, 0.0]; MAX_LIGHTS_COUNT],
            count: 3,
            _padding2: [0; 3],
        };
        let light_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Light VB"),
                contents: bytemuck::cast_slice(&[light_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        (light_uniform, light_buffer, light_bind_group_layout, light_bind_group)
    }

    async fn load_models(
        device: &wgpu::Device, queue: &wgpu::Queue, texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<Vec<model::Model>> {
        let floor_model = resources::load_model(
            "plane.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        ).await.unwrap();
        let tree_model = resources::load_model(
            "tree.obj",
            &device,
            &queue,
            &texture_bind_group_layout,
        ).await.unwrap();

        Ok(vec![floor_model, tree_model])// Must be in the same order as the instance buffers!
    }

    fn create_instances(device: &wgpu::Device) -> Vec<(u32, wgpu::Buffer)> {
        const SPACING: f32 = 3.5;
        let tree_instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|y| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACING * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let y = SPACING * (y as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                    let position = cgmath::Vector3 { x, y, z: 0.0 };
                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        let angle = 22.0 * x + 45.0 * y;
                        cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(angle))
                    };

                    Instance { position, rotation }
                })
            }).collect::<Vec<_>>();

        let instance_data = tree_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let tree_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let plane_instances = (0..1).flat_map(|_| {
            (0..1).map(|_| {
                let position = cgmath::Vector3::zero();
                let rotation = cgmath::Quaternion::zero();
                Instance { position, rotation }
            })
        }).collect::<Vec<_>>();
        let instance_data = plane_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let plane_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        vec!(
            (plane_instances.len() as u32, plane_instance_buffer),
            (tree_instances.len() as u32, tree_instance_buffer),
        )
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        shader: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(shader);

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: vertex_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }
    pub(crate) fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    virtual_keycode: Some(key),
                    state,
                    ..
                },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    pub(crate) fn update(&mut self, character: &Character, dt: instant::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        self.character_uniform.update_matrix(character.position.clone());
        self.queue.write_buffer(
            &self.character_buffer,
            0,
            bytemuck::cast_slice(&[self.character_uniform]),
        );

        let old_position: cgmath::Vector4<_> = self.light_uniform.positions[0].into();
        let new_position: cgmath::Vector3<_> = (cgmath::Quaternion::from_axis_angle((0.0, 0.0, 1.0).into(), cgmath::Deg(60.0 * dt.as_secs_f32())) * old_position.xyz()).into();
        self.light_uniform.positions[0] = [new_position.x, new_position.y, new_position.z, 0.0];
        self.light_uniform.colors[0] = [0.05, 1.0, 0.05, 8.0];

        let old_position: cgmath::Vector4<_> = self.light_uniform.positions[1].into();
        let new_position: cgmath::Vector3<_> = (cgmath::Quaternion::from_axis_angle((0.0, 0.0, 1.0).into(), cgmath::Deg(30.0 * dt.as_secs_f32())) * old_position.xyz()).into();
        self.light_uniform.positions[1] = [new_position.x, new_position.y, new_position.z, 0.0];
        self.light_uniform.colors[1] = [1.0, 0.05, 0.05, 8.0];

        let old_position: cgmath::Vector4<_> = self.light_uniform.positions[2].into();
        let new_position: cgmath::Vector3<_> = (cgmath::Quaternion::from_axis_angle((0.0, 0.0, 1.0).into(), cgmath::Deg(15.0 * dt.as_secs_f32())) * old_position.xyz()).into();
        self.light_uniform.positions[2] = [new_position.x, new_position.y, new_position.z, 0.0];
        self.light_uniform.colors[2] = [0.05, 0.05, 1.0, 8.0];
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light_uniform]));
    }

    pub(crate) fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            // Light sources draw ================================================================================
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_instanced(&self.light_source_model, 0..self.light_uniform.count as u32, &self.camera_bind_group, &self.light_bind_group);
            // Geometry draw =====================================================================================
            render_pass.set_pipeline(&self.render_pipeline);
            for i in 0..self.models.len() as usize {
                render_pass.set_vertex_buffer(1, self.instances_data[i].1.slice(..));
                render_pass.draw_model_instanced(
                    &self.models[i],
                    0..self.instances_data[i].0,
                    &self.camera_bind_group,
                    &self.light_bind_group,
                )
            }
            // Character draw =================================================================================
            render_pass.set_pipeline(&self.character_render_pipeline);
            render_pass.draw_character(&self.character_model, &self.camera_bind_group, &self.light_bind_group, &self.character_bind_group);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}