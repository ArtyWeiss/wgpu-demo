mod render_system;
mod character_system;

use winit::event::{VirtualKeyCode, KeyboardInput, ElementState};
use winit::event::*;
use winit::event_loop::{ControlFlow, EventLoop};

use winit::dpi::PhysicalSize;

pub struct GameState {
    pub character: character_system::Character,
    pub character_controller: character_system::CharacterController,
}

impl GameState {
    fn new() -> Self {
        Self {
            character: character_system::Character::new(1.5 ,2.5, 10.0),
            character_controller: character_system::CharacterController::new(),
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    virtual_keycode: Some(key),
                    state,
                    ..
                },
                ..
            } => {
                self.character_controller.process_keyboard(*key, *state)
            },
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
        .with_decorations(true)
        .with_maximized(true)
        .with_min_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();
    window.set_cursor_visible(false);

    let mut game_state = GameState::new();
    let mut state = render_system::State::new(&window).await;
    let mut last_render_time = instant::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        state.platform.handle_event(&event);
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta, },
                ..
            } => if delta.0.abs() > 0.1 || delta.1.abs() > 0.1 {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() && !game_state.input(event) => {
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
                game_state.character_controller.update_character(&mut game_state.character, &state.camera, dt);
                state.update(&game_state, dt);

                state.platform.update_time(now.elapsed().as_secs_f64());

                match state.render(&window) {
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
