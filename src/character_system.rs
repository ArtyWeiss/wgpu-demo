use instant::Duration;
use cgmath;
use cgmath::{Angle, InnerSpace, Quaternion, Rotation3, Vector3};
use winit::event::*;

use crate::render_system::Camera;

pub struct Character {
    pub position: cgmath::Point3<f32>,
    pub head_height: f32,
    pub direction: cgmath::Vector3<f32>,
    pub speed: f32,
    pub rotation_speed: f32,
}

impl Character {
    pub fn new(head_height: f32, speed: f32, rotation_speed: f32) -> Self {
        Self {
            position: cgmath::Point3 { x: 0.0, y: 0.0, z: 0.0 },
            head_height,
            direction: cgmath::Vector3::unit_y(),
            speed,
            rotation_speed,
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
            VirtualKeyCode::W => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D => {
                self.amount_right = amount;
                true
            }
            _ => false
        }
    }

    pub fn update_character(&mut self, character: &mut Character, camera: &Camera, dt: Duration) {
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = cgmath::Vector3::new(yaw_sin, yaw_cos, 0.0);
        let right = cgmath::Vector3::new(yaw_cos, -yaw_sin, 0.0);
        let dt = dt.as_secs_f32();
        let movement_forward = forward * (self.amount_forward - self.amount_backward);
        let movement_right = right * (self.amount_right - self.amount_left);
        if movement_right.magnitude() > 0.0 || movement_forward.magnitude() > 0.0 {
            let movement_vector = (movement_forward + movement_right).normalize();
            character.position += movement_vector * character.speed * dt;

            let cross = Vector3::unit_z().cross(character.direction);
            let dot = cross.dot(movement_vector);

            let mut angle = Vector3::angle(movement_vector, character.direction);
            if dot < 0.0 {
                angle *= -1.0;
            }
            let rotation = Quaternion::from_axis_angle(
                Vector3::unit_z(),
                cgmath::Deg::from(angle * character.rotation_speed * dt),
            );
            character.direction = rotation * character.direction;
        }
    }
}