use instant::Duration;
use cgmath;
use winit::event::*;

pub struct Character {
    pub position: cgmath::Vector3<f32>,
    pub direction: cgmath::Vector3<f32>,
    pub speed: f32,
}

impl Character {
    pub fn new(speed: f32) -> Self {
        Self {
            position: cgmath::Vector3 { x: 0.0, y: 0.0, z: 0.0 },
            direction: cgmath::Vector3::unit_y(),
            speed,
        }
    }
}

pub struct CharacterController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
}

impl CharacterController {
    pub fn new() -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            _ => false
        }
    }

    pub fn update_character(&mut self, character: &mut Character, dt: Duration) {
        let forward = cgmath::Vector3::unit_y();
        let right = cgmath::Vector3::unit_x();
        let dt = dt.as_secs_f32();
        character.position += forward * (self.amount_forward - self.amount_backward) * character.speed * dt;
        character.position += right * (self.amount_right - self.amount_left) * character.speed * dt;
    }
}