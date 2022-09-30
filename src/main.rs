mod render_system;
mod character_system;

use cgmath::Point3;
use crate::render_system::{CameraController, FollowCameraController};

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
};
use winit::dpi::PhysicalSize;

pub struct GameState {
    pub character: character_system::Character,
    pub character_controller: character_system::CharacterController,
}

impl GameState {
    fn new() -> Self {
        Self {
            character: character_system::Character::new(1.0 ,3.5),
            character_controller: character_system::CharacterController::new(),
        }
    }

    fn input(&mut self, camera_controller: &mut FollowCameraController, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    virtual_keycode: Some(key),
                    state,
                    ..
                },
                ..
            } => {
                // camera_controller.process_keyboard(*key, *state) |
                self.character_controller.process_keyboard(*key, *state)
            },
            WindowEvent::MouseWheel { delta, .. } => {
                // camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                camera_controller.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let title = "Junk Souls";
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut game_state = GameState::new();
    let mut state = render_system::State::new(&window).await;
    let mut last_render_time = instant::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta, },
                ..
            } => if state.camera_controller.mouse_pressed {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !game_state.input(&mut state.camera_controller, event) => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let now = instant::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                game_state.character_controller.update_character(&mut game_state.character, dt);
                state.update(&game_state, dt);
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            _ => {}
        }
    });
}
