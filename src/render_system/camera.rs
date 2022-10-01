use cgmath::*;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use instant::Duration;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.00001;

#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    pub fn new<
        V: Into<Point3<f32>>,
        Y: Into<Rad<f32>>,
        P: Into<Rad<f32>>,
    >(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_to_rh(
            self.position,
            Vector3::new(
                self.yaw.0.sin(),
                self.yaw.0.cos(),
                self.pitch.0.sin(),
            ).normalize(),
            Vector3::unit_z(),
        )
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(
        width: u32,
        height: u32,
        fovy: F,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug)]
pub struct FollowCameraController {
    pub(crate) mouse_pressed: bool,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    sensitivity: f32,
    distance: f32,
    focus_radius: f32,
    focus_point: Point3<f32>,
}

impl FollowCameraController {
    pub fn new(
        sensitivity: f32,
        distance: f32,
        focus_radius: f32,
    ) -> Self {
        Self {
            mouse_pressed: false,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            sensitivity,
            distance,
            focus_radius,
            focus_point: Point3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration, target: Point3<f32>) {
        let dt = dt.as_secs_f32();
        // Follow focus point
        if self.focus_radius > 0.0 {
            let distance = (target - self.focus_point).magnitude();
            let mut t: f32 = 1.0;
            let focus_centering: f32 = 0.85;
            if distance > 0.01 {
                t = (1.0 - focus_centering).powf(dt);
            }
            if distance > self.focus_radius {
                t = t.min(self.focus_radius / distance);
            }
            self.focus_point = Point3::from_vec(target.to_vec().lerp(self.focus_point.to_vec(), t));
        } else {
            self.focus_point = target;
        }
        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
        // Set position
        let look_direction = Vector3::new(
            camera.yaw.0.sin(),
            camera.yaw.0.cos(),
            camera.pitch.0.sin(),
        ).normalize();
        camera.position = self.focus_point - look_direction * self.distance;
    }
}

fn yaw_and_pitch_from_vector(v: cgmath::Vector3<f32>) -> (Rad<f32>, Rad<f32>) {
    let y = Vector3::unit_y();
    let z = Vector3::unit_z();

    let v_xy = Vector3::new(v.x, v.y, 0.0).normalize();
    if v_xy == Vector3::zero() {
        if v.dot(z) > 0.0 {
            return (Rad(0.0), Rad(FRAC_PI_2));
        } else {
            return (Rad(0.0), Rad(-FRAC_PI_2));
        }
    }

    let mut yaw = v_xy.angle(y);
    if v.x < 0.0 {
        yaw *= -1.0;
    }

    let mut pitch = v_xy.angle(v);
    if v.z < 0.0 {
        pitch *= -1.0;
    }

    (yaw, pitch)
}

#[derive(Debug)]
pub struct CameraController {
    pub(crate) mouse_pressed: bool,
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    is_reset_pressed: bool,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            mouse_pressed: false,
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            is_reset_pressed: false,
            scroll: 0.0,
            speed,
            sensitivity,
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
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true
            }
            VirtualKeyCode::R => {
                self.is_reset_pressed = true;
                true
            }
            _ => false
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        if self.is_reset_pressed {
            camera.position = Point3::new(0.0, 0.0, 1.0);
            camera.yaw = Rad(0.0);
            camera.pitch = Rad(0.0);
            self.is_reset_pressed = false;
        }

        let dt = dt.as_secs_f32();

        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_sin, yaw_cos, 0.0).normalize();
        let right = Vector3::new(yaw_cos, -yaw_sin, 0.0).normalize();
        // Move right/left/forward/backward
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;
        // Move in/out (similar to zoom, but with camera movement)
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_sin, pitch_cos * yaw_cos, pitch_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;
        // Move up/down
        camera.position.z += (self.amount_up - self.amount_down) * self.speed * dt;
        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}