// file: cnncheckin/src/camera.rs
// Módulo de captura de webcam

 // file: cnncheckin/src/camera.rs
// Módulo de captura de webcam

use std::error::Error;
use std::time::{Duration, Instant};
use minifb::{Key, Window, WindowOptions};
use std::io::{self, Write};

use crate::utils::{save_photo, yuyv_to_rgb_bytes, get_timestamp, ensure_directory_exists, FileNameGenerator, InputValidator};
use crate::config::Config;

pub struct WebcamCapture {
    window: Window,
    buffer: Vec<u32>,
    rgb_buffer: Vec<u8>,
    current_frame: Vec<u8>,
    width: usize,
    height: usize,
}

impl WebcamCapture {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let config = Config::load()?;
        let width = config.camera.width;
        let height = config.camera.height;
        
        let window = Window::new(
            "CNN CheckIn - Captura",
            width,
            height,
            WindowOptions::default(),
        )?;
        
        let buffer_size = width * height;
        let rgb_buffer_size = width * height * 3;
        
        println!("📹 Webcam inicializada: {}x{}", width, height);
        
        Ok(Self {
            window,
            buffer: vec![0; buffer_size],
            rgb_buffer: vec![0; rgb_buffer_size],
            current_frame: vec![0; rgb_buffer_size],
            width,
            height,
        })
    }
    
    pub fn capture_frame(&mut self) -> Result<(), Box<dyn Error>> {
        // Simular captura de frame - em implementação real usaria rscam ou nokhwa
        // Por agora, vamos gerar um frame de teste
        self.generate_test_frame();
        
        // Converter RGB para buffer de display
        self.rgb_to_display_buffer();
        
        // Atualizar janela
        self.window.update_with_buffer(&self.buffer, self.width, self.height)?;
        
        Ok(())
    }
    
    fn generate_test_frame(&mut self) {
        // Gerar padrão de teste colorido
        let timestamp = get_timestamp() as f32 * 0.001;
        
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * 3;
                
                let r = ((x as f32 / self.width as f32) * 255.0) as u8;
                let g = ((y as f32 / self.height as f32) * 255.0) as u8;
                let b = ((timestamp.sin() * 0.5 + 0.5) * 255.0) as u8;
                
                if idx + 2 < self.current_frame.len() {
                    self.current_frame[idx] = r;
                    self.current_frame[idx + 1] = g;
                    self.current_frame[idx + 2] = b;
                }
            }
        }
    }
    
    fn rgb_to_display_buffer(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let rgb_idx = (y * self.width + x) * 3;
                let buf_idx = y * self.width + x;
                
                if rgb_idx + 2 < self.current_frame.len() && buf_idx < self.buffer.len() {
                    let r = self.current_frame[rgb_idx] as u32;
                    let g = self.current_frame[rgb_idx + 1] as u32;
                    let b = self.current_frame[rgb_idx + 2] as u32;
                    
                    self.buffer[buf_idx] = (r << 16) | (g << 8) | b;
                }
            }
        }
    }
    
    pub fn is_window_open(&self) -> bool {
        self.window.is_open()
    }
    
    pub fn is_key_down(&self, key: Key) -> bool {
        self.window.is_key_down(key)
    }
    
    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.window.is_key_pressed(key, minifb::KeyRepeat::No)
    }
    
    pub fn update_title(&mut self, title: &str) {
        // Note: minifb doesn't support dynamic title updates
        // This would need a different approach in a real implementation
    }
    
    pub fn get_current_frame(&self) -> &[u8] {
        &self.current_frame
    }
    
    pub fn get_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

pub async fn capture_training_images(count: u32) -> Result<(), Box<dyn Error>> {
    let config = Config::load()?;
    
    // Criar diretórios necessários
    ensure_directory_exists(&config.paths.photos_dir)?;
    ensure_directory_exists(&config.paths.training_dir)?;
    
    let mut capture = WebcamCapture::new()?;
    
    println!("🎥 Modo Captura de Treinamento");
    println!("📋 Controles:");
    println!("  ESPAÇO ou S - Capturar foto");
    println!("  N - Nova pessoa");
    println!("  R - Reset contador FPS");
    println!("  ESC - Sair");
    
    let mut current_person = String::new();
    let mut photo_count = 0u32;
    let mut person_count = 1u32;
    let mut frame_count = 0u64;
    let mut last_fps_update = Instant::now();
    
    // Solicitar nome da primeira pessoa
    print!("👤 Digite o nome da primeira pessoa: ");
    io::stdout().flush()?;
    io::stdin().read_line(&mut current_person)?;
    current_person = current_person.trim().to_string();
    
    if let Err(e) = InputValidator::validate_person_name(&current_person) {
        println!("❌ Nome inválido: {}", e);
        return Ok(());
    }
    
    while capture.is_window_open() && !capture.is_key_down(Key::Escape) {
        let frame_start = Instant::now();
        
        // Capturar frame
        if let Err(e) = capture.capture_frame() {
            eprintln!("⚠️ Erro ao capturar frame: {}", e);
            continue;
        }
        
        // Processar teclas
        if capture.is_key_pressed(Key::Space) || capture.is_key_pressed(Key::S) {
            if current_person.is_empty() {
                println!("❌ Nenhuma pessoa definida. Use 'N' para definir uma nova pessoa.");
                continue;
            }
            
            // Capturar foto
            let frame = capture.get_current_frame();
            let (width, height) = capture.get_dimensions();
            
            // Criar nome do arquivo
            let filename = FileNameGenerator::generate_photo_name(&current_person, photo_count + 1);
            let person_dir = format!("{}/{:03}_{}", config.paths.training_dir, person_count, current_person);
            ensure_directory_exists(&person_dir)?;
            
            let filepath = format!("{}/{}", person_dir, filename);
            
            // Salvar foto
            match save_photo(frame, width, height, &filepath) {
                Ok(_) => {
                    photo_count += 1;
                    println!("📸 Foto {}/{} capturada: {}", photo_count, count, filename);
                    
                    if photo_count >= count {
                        println!("✅ {} fotos capturadas para {}!", count, current_person);
                        photo_count = 0;
                        current_person.clear();
                        
                        println!("👤 Digite o nome da próxima pessoa (ou ENTER para terminar): ");
                        let mut next_person = String::new();
                        // Note: Em uma implementação real, isso seria não-bloqueante
                    }
                }
                Err(e) => {
                    eprintln!("❌ Erro ao salvar foto: {}", e);
                }
            }
        }
        
        if capture.is_key_pressed(Key::N) {
            print!("👤 Digite o nome da nova pessoa: ");
            io::stdout().flush()?;
            current_person.clear();
            // Note: Implementação simplificada - em produção seria não-bloqueante
        }
        
        if capture.is_key_pressed(Key::R) {
            frame_count = 0;
            last_fps_update = Instant::now();
        }
        
        // Calcular FPS
        frame_count += 1;
        let now = Instant::now();
        if now.duration_since(last_fps_update) >= Duration::from_secs(1) {
            let fps = frame_count as f64 / now.duration_since(last_fps_update).as_secs_f64();
            frame_count = 0;
            last_fps_update = now;
            
            println!("📊 FPS: {:.1} | Pessoa: {} | Fotos: {}/{}", 
                    fps, current_person, photo_count, count);
        }
        
        // Manter taxa de quadros estável
        let frame_time = frame_start.elapsed();
        let target_frame_time = Duration::from_millis(33); // ~30 FPS
        if frame_time < target_frame_time {
            std::thread::sleep(target_frame_time - frame_time);
        }
    }
    
    println!("📊 Sessão de captura finalizada");
    Ok(())
}

pub async fn capture_single_frame() -> Result<Vec<u8>, Box<dyn Error>> {
    println!("📷 Capturando frame único...");
    
    // Simular captura de frame único
    let config = Config::load()?;
    let width = config.camera.width;
    let height = config.camera.height;
    let size = width * height * 3;
    
    let mut frame = vec![0u8; size];
    
    // Gerar frame de teste
    let timestamp = get_timestamp() as f32 * 0.001;
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            
            if idx + 2 < frame.len() {
                frame[idx] = ((x as f32 / width as f32) * 255.0) as u8;     // R
                frame[idx + 1] = ((y as f32 / height as f32) * 255.0) as u8; // G
                frame[idx + 2] = ((timestamp.sin() * 0.5 + 0.5) * 255.0) as u8; // B
            }
        }
    }
    
    println!("✅ Frame capturado: {}x{} ({} bytes)", width, height, frame.len());
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_webcam_creation() {
        // Note: Este teste pode falhar se não houver display
        // Em CI/CD, seria necessário um display virtual
        let result = WebcamCapture::new();
        // Apenas verificar que a função não causa pânico
        println!("Webcam creation result: {:?}", result.is_ok());
    }
    
    #[tokio::test]
    async fn test_single_frame_capture() {
        let result = capture_single_frame().await;
        assert!(result.is_ok());
        
        let frame = result.unwrap();
        assert!(!frame.is_empty());
    }
}