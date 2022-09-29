use instant;
use cgmath;

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
    pub fn update(&mut self, input_direction: cgmath::Vector2<f32>, dt: instant::Duration) {
        let movement = input_direction * self.speed * dt.as_secs_f32();
        self.position += [movement.x, movement.y, 0.0].into();
    }
}