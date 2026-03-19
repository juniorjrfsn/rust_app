// file: cnncheckin/src/utils.rs
// Módulo de funções utilitárias


use minifb::{Key, Window, WindowOptions};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use crate::config::CameraConfig;

pub struct WebcamCapture {
    window: Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
    fps_counter: FpsCounter,
}

impl WebcamCapture {
    pub fn new(config: &CameraConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let window = Window::new("CNN CheckIn", config.width, config.height, WindowOptions::default())?;
        let buffer = vec![0u32; config.width * config.height];
        Ok(Self {
            window,
            buffer,
            width: config.width,
            height: config.height,
            fps_counter: FpsCounter::new(),
        })
    }

    pub fn capture_frame(&mut self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Placeholder: simula captura de frame
        let frame = vec![0u8; self.width * self.height * 3];
        self.fps_counter.tick();
        self.window.update_with_buffer(&self.buffer, self.width, self.height)?;
        Ok(frame)
    }

    pub fn is_window_open(&self) -> bool { self.window.is_open() }
    pub fn is_key_down(&self, key: Key) -> bool { self.window.is_key_down(key) }
    pub fn is_key_pressed(&self, key: Key) -> bool { 
        self.window.get_keys_pressed(minifb::KeyRepeat::No)
            .iter().any(|&k| k == key)
    }
    pub fn update_title(&mut self, title: &str) { self.window.set_title(title); }

    pub async fn capture_dataset(&mut self, count: u32, person: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
        let mut person_name = person.unwrap_or_default();
        let mut photos_taken = 0;
        let width = self.width;
        let height = self.height;

        while self.is_window_open() && !self.is_key_down(Key::Escape) && photos_taken < count {
            let frame = self.capture_frame()?;
            if self.is_key_pressed(Key::Space) && !frame.is_empty() {
                if person_name.is_empty() {
                    println!("Digite o nome da pessoa: ");
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input)?;
                    person_name = input.trim().to_string();
                }
                let filename = format!("../../dados/fotos_treino/{}/photo_{}.ppm", person_name, photos_taken + 1);
                save_photo(&frame, width, height, &filename)?;
                photos_taken += 1;
                println!("📸 Capturadas {}/{} fotos para {}", photos_taken, count, person_name);
            }
            if self.is_key_pressed(Key::N) {
                person_name.clear();
                photos_taken = 0;
            }
            std::thread::sleep(Duration::from_millis(33));
        }
        Ok(())
    }
}

pub fn save_photo(rgb_data: &[u8], width: usize, height: usize, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(filename)?;
    writeln!(file, "P6\n{} {}\n255", width, height)?;
    file.write_all(rgb_data)?;
    Ok(())
}

struct FpsCounter {
    frame_count: u64,
    last_update: Instant,
    current_fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self { frame_count: 0, last_update: Instant::now(), current_fps: 0.0 }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);
        if elapsed >= Duration::from_secs(1) {
            self.current_fps = self.frame_count as f64 / elapsed.as_secs_f64();
            self.frame_count = 0;
            self.last_update = now;
        }
    }

    fn fps(&self) -> f64 { self.current_fps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fps_counter() {
        let mut counter = FpsCounter::new();
        for _ in 0..30 {
            counter.tick();
            std::thread::sleep(Duration::from_millis(33));
        }
        let fps = counter.fps();
        assert!(fps > 25.0 && fps < 35.0);
    }
}