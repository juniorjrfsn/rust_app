// file: cnncheckin/src/camera.rs
// Módulo de captura de webcam

// file: src/camera/mod.rs
// Módulo de captura de webcam refatorado

mod capture;
mod frame;

pub use capture::WebcamCapture;
pub use frame::Frame;

use crate::config::CameraConfig;
use crate::error::{AppError, Result};
use std::time::{Duration, Instant};

/// Trait para abstrair captura de câmera
pub trait CameraDevice: Send + Sync {
    /// Captura um frame da câmera
    fn capture_frame(&mut self) -> Result<Frame>;
    
    /// Retorna dimensões da câmera
    fn dimensions(&self) -> (usize, usize);
    
    /// Verifica se a câmera está ativa
    fn is_active(&self) -> bool;
    
    /// Para a captura
    fn stop(&mut self) -> Result<()>;
}

/// Gerenciador de câmera com métricas
pub struct CameraManager {
    device: Box<dyn CameraDevice>,
    fps_counter: FpsCounter,
    config: CameraConfig,
}

impl CameraManager {
    pub fn new(config: CameraConfig) -> Result<Self> {
        let device = WebcamCapture::new(&config)?;
        
        Ok(Self {
            device: Box::new(device),
            fps_counter: FpsCounter::new(),
            config,
        })
    }
    
    /// Captura um frame e atualiza métricas
    pub fn capture(&mut self) -> Result<Frame> {
        let frame = self.device.capture_frame()?;
        self.fps_counter.tick();
        Ok(frame)
    }
    
    /// Retorna FPS atual
    pub fn fps(&self) -> f64 {
        self.fps_counter.fps()
    }
    
    /// Dimensões da câmera
    pub fn dimensions(&self) -> (usize, usize) {
        self.device.dimensions()
    }
    
    /// Para a câmera
    pub fn stop(&mut self) -> Result<()> {
        self.device.stop()
    }
}

/// Contador de FPS
struct FpsCounter {
    frame_count: u64,
    last_update: Instant,
    current_fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frame_count: 0,
            last_update: Instant::now(),
            current_fps: 0.0,
        }
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
    
    fn fps(&self) -> f64 {
        self.current_fps
    }
    
    fn reset(&mut self) {
        self.frame_count = 0;
        self.last_update = Instant::now();
        self.current_fps = 0.0;
    }
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