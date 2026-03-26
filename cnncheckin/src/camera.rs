// projeto cnncheckin
// src/camera.rs — Captura de frames isolada do framework de janela

use std::path::Path;
use std::time::{Duration, Instant};

use crate::config::CameraConfig;
use crate::error::{AppError, Result};
use crate::image_processor::save_ppm;

pub struct Frame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Frame {
    pub fn new_blank(width: usize, height: usize) -> Self {
        Self {
            data: vec![64u8; width * height * 3], // cinza escuro
            width,
            height,
        }
    }
}

pub struct WebcamCapture {
    width: usize,
    height: usize,
    fps_counter: FpsCounter,
    current_frame: Frame,
}

impl WebcamCapture {
    pub fn new(config: &CameraConfig) -> Result<Self> {
        let current_frame = Frame::new_blank(config.width, config.height);

        Ok(Self {
            width: config.width,
            height: config.height,
            fps_counter: FpsCounter::new(),
            current_frame,
        })
    }

    pub fn capture_frame(&mut self) -> Result<&Frame> {
        self.current_frame = generate_synthetic_frame(self.width, self.height);
        self.fps_counter.tick();
        Ok(&self.current_frame)
    }

    pub fn current_frame_data(&self) -> &[u8] {
        &self.current_frame.data
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn fps(&self) -> f64 {
        self.fps_counter.fps()
    }

    pub fn save_current_frame(&self, path: &Path) -> Result<()> {
        save_ppm(&self.current_frame.data, self.width, self.height, path)
    }
}

fn generate_synthetic_frame(width: usize, height: usize) -> Frame {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_millis() as usize;

    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x + t) % 256) as u8;
            let g = ((y + t / 2) % 256) as u8;
            let b = ((x + y + t / 3) % 256) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    Frame { data, width, height }
}

pub fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

pub struct FpsCounter {
    frame_count: u64,
    last_update: Instant,
    current_fps: f64,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            last_update: Instant::now(),
            current_fps: 0.0,
        }
    }

    pub fn tick(&mut self) {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);
        if elapsed >= Duration::from_secs(1) {
            self.current_fps = self.frame_count as f64 / elapsed.as_secs_f64();
            self.frame_count = 0;
            self.last_update = now;
        }
    }

    pub fn fps(&self) -> f64 {
        self.current_fps
    }
}