use egui::Vec2;
use egui_extras::RetainedImage;

const FRAME_COUNT_INTERVAL: f32 = 0.1;

pub struct InfoPanel {
    image: RetainedImage,
    pub(crate) frame_count: i32,
    pub(crate) accumulated_frame_time: f32,
    pub(crate) average_frame_time: f32,
}

impl InfoPanel {
    pub fn default() -> Self {
        Self {
            image: RetainedImage::from_image_bytes(
                "lab-icon.png",
                include_bytes!("lab-icon.png"),
            ).unwrap(),
            frame_count: 0,
            accumulated_frame_time: 0.0,
            average_frame_time: 0.0,
        }
    }
    pub fn update_frame_time(&mut self, dt: f32) {
        self.accumulated_frame_time += dt;
        self.frame_count += 1;

        if self.accumulated_frame_time >= FRAME_COUNT_INTERVAL {
            self.average_frame_time = self.accumulated_frame_time * 1000.0 / self.frame_count as f32;
            self.accumulated_frame_time = 0.0;
            self.frame_count = 0;
        }
    }
    pub fn draw(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("info_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.set_height(48.0);
                self.image.show_size(ui, Vec2::new(32.0, 32.0));
                ui.add_space(20.0);
                ui.vertical(|ui|{
                    ui.add_space(5.0);
                    ui.horizontal(|ui| {
                        ui.label("Frame time:");
                        ui.label(format!("{0:.2}", self.average_frame_time));
                        ui.label("ms");
                    });
                    ui.horizontal(|ui|{
                        ui.label("Frame rate:");
                        let frame_rate = 1000.0 / self.average_frame_time;
                        ui.label(frame_rate.round().to_string());
                    });
                });
                ui.add_space(20.0);
                ui.label("WASD - Перемещение");
                ui.add_space(10.0);
                ui.label("F - Полноэкранный режим");
                ui.add_space(10.0);
                ui.label("H - Видимость курсора");
                ui.add_space(10.0);
                ui.label("ESC - Закрыть приложение");
            });
        });
    }
}