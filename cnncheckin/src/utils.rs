// file: cnncheckin/src/utils.rs
// Módulo de funções utilitárias

use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Salvar foto em formato PPM
pub fn save_photo(
    rgb_data: &[u8], 
    width: usize, 
    height: usize, 
    filename: &str
) -> Result<(), Box<dyn Error>> {
    let mut file = fs::File::create(filename)?;
    
    // Escrever cabeçalho PPM
    writeln!(file, "P6")?;
    writeln!(file, "# CNN CheckIn - Foto capturada")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    
    // Escrever dados RGB
    let expected_size = width * height * 3;
    let data_to_write = if rgb_data.len() >= expected_size {
        &rgb_data[..expected_size]
    } else {
        rgb_data
    };
    
    file.write_all(data_to_write)?;
    
    println!("📸 Foto salva: {} ({} bytes)", filename, data_to_write.len());
    Ok(())
}

/// Converter formato YUYV para RGB bytes
pub fn yuyv_to_rgb_bytes(yuyv_data: &[u8], rgb_data: &mut [u8], width: usize, height: usize) {
    let pixels_per_line = width;
    
    if yuyv_data.len() < width * height * 2 {
        eprintln!("YUYV data size mismatch: expected {}, got {}", width * height * 2, yuyv_data.len());
        return;
    }
    
    if rgb_data.len() < width * height * 3 {
        eprintln!("RGB buffer size mismatch: expected {}, got {}", width * height * 3, rgb_data.len());
        return;
    }
    
    for y in 0..height {
        for x in 0..(width / 2) {
            let yuyv_idx = (y * pixels_per_line + x * 2) * 2;
            let rgb_idx1 = (y * pixels_per_line + x * 2) * 3;
            let rgb_idx2 = rgb_idx1 + 3;
            
            if yuyv_idx + 3 < yuyv_data.len() && rgb_idx2 + 2 < rgb_data.len() {
                let y1 = yuyv_data[yuyv_idx] as f32;
                let u = yuyv_data[yuyv_idx + 1] as f32;
                let y2 = yuyv_data[yuyv_idx + 2] as f32;
                let v = yuyv_data[yuyv_idx + 3] as f32;
                
                // Converter primeiro pixel (Y1UV)
                let (r1, g1, b1) = yuv_to_rgb(y1, u, v);
                rgb_data[rgb_idx1] = r1;
                rgb_data[rgb_idx1 + 1] = g1;
                rgb_data[rgb_idx1 + 2] = b1;
                
                // Converter segundo pixel (Y2UV)
                let (r2, g2, b2) = yuv_to_rgb(y2, u, v);
                rgb_data[rgb_idx2] = r2;
                rgb_data[rgb_idx2 + 1] = g2;
                rgb_data[rgb_idx2 + 2] = b2;
            }
        }
    }
}

/// Conversão YUV para RGB
fn yuv_to_rgb(y: f32, u: f32, v: f32) -> (u8, u8, u8) {
    let y = y - 16.0;
    let u = u - 128.0;
    let v = v - 128.0;
    
    let r = (1.164 * y + 1.596 * v).clamp(0.0, 255.0) as u8;
    let g = (1.164 * y - 0.392 * u - 0.813 * v).clamp(0.0, 255.0) as u8;
    let b = (1.164 * y + 2.017 * u).clamp(0.0, 255.0) as u8;
    
    (r, g, b)
}

/// Obter timestamp atual
pub fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Formatar timestamp para string legível
pub fn format_timestamp(timestamp: u64) -> String {
    use chrono::{DateTime, Utc, TimeZone};
    
    let dt = Utc.timestamp_opt(timestamp as i64, 0).single();
    match dt {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        None => "Invalid timestamp".to_string(),
    }
}

/// Calcular hash SHA256 de dados
pub fn calculate_hash(data: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    
    format!("{:x}", result)
}

/// Verificar se um diretório existe e criar se necessário
pub fn ensure_directory_exists(path: &str) -> Result<(), Box<dyn Error>> {
    if !std::path::Path::new(path).exists() {
        fs::create_dir_all(path)?;
        println!("📁 Diretório criado: {}", path);
    }
    Ok(())
}

/// Obter informações do sistema
pub fn get_system_info() -> SystemInfo {
    SystemInfo {
        num_cpus: num_cpus::get(),
        hostname: get_hostname(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub num_cpus: usize,
    pub hostname: String,
    pub os: String,
    pub arch: String,
}

fn get_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Comprimir dados usando gzip
pub fn compress_data(data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

/// Descomprimir dados gzip
pub fn decompress_data(compressed_data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    
    let mut decoder = GzDecoder::new(compressed_data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

/// Medidor de performance para operações
pub struct PerformanceTimer {
    start_time: std::time::Instant,
    operation_name: String,
}

impl PerformanceTimer {
    pub fn new(operation_name: &str) -> Self {
        println!("⏱️  Iniciando: {}", operation_name);
        Self {
            start_time: std::time::Instant::now(),
            operation_name: operation_name.to_string(),
        }
    }
    
    pub fn checkpoint(&self, checkpoint_name: &str) {
        let elapsed = self.start_time.elapsed();
        println!("📊 {} - {}: {:.2?}", self.operation_name, checkpoint_name, elapsed);
    }
    
    pub fn finish(self) {
        let elapsed = self.start_time.elapsed();
        println!("✅ {} concluído em: {:.2?}", self.operation_name, elapsed);
    }
}

/// Contador de progresso
pub struct ProgressCounter {
    total: usize,
    current: usize,
    last_percentage: usize,
    operation_name: String,
}

impl ProgressCounter {
    pub fn new(total: usize, operation_name: &str) -> Self {
        println!("🚀 Iniciando: {} (0/{})", operation_name, total);
        Self {
            total,
            current: 0,
            last_percentage: 0,
            operation_name: operation_name.to_string(),
        }
    }
    
    pub fn increment(&mut self) {
        self.current += 1;
        let percentage = (self.current * 100) / self.total;
        
        // Só mostrar progresso a cada 10%
        if percentage != self.last_percentage && percentage % 10 == 0 {
            println!("📈 {} - {}% ({}/{})", 
                    self.operation_name, percentage, self.current, self.total);
            self.last_percentage = percentage;
        }
    }
    
    pub fn finish(self) {
        println!("✅ {} concluído: {}/{}", self.operation_name, self.current, self.total);
    }
}

/// Validador de entrada de dados
pub struct InputValidator;

impl InputValidator {
    pub fn validate_person_name(name: &str) -> Result<String, &'static str> {
        let trimmed = name.trim();
        
        if trimmed.is_empty() {
            return Err("Nome não pode estar vazio");
        }
        
        if trimmed.len() < 2 {
            return Err("Nome deve ter pelo menos 2 caracteres");
        }
        
        if trimmed.len() > 50 {
            return Err("Nome não pode ter mais de 50 caracteres");
        }
        
        // Verificar se contém apenas caracteres válidos
        if !trimmed.chars().all(|c| c.is_alphabetic() || c.is_whitespace() || "áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ".contains(c)) {
            return Err("Nome contém caracteres inválidos");
        }
        
        Ok(trimmed.to_string())
    }
    
    pub fn validate_confidence(confidence: f32) -> Result<f32, &'static str> {
        if confidence < 0.0 || confidence > 1.0 {
            return Err("Confiança deve estar entre 0.0 e 1.0");
        }
        Ok(confidence)
    }
    
    pub fn validate_image_dimensions(width: usize, height: usize) -> Result<(usize, usize), &'static str> {
        if width == 0 || height == 0 {
            return Err("Dimensões da imagem devem ser maiores que zero");
        }
        
        if width > 4096 || height > 4096 {
            return Err("Dimensões da imagem muito grandes (máx: 4096x4096)");
        }
        
        Ok((width, height))
    }
}

/// Gerador de nomes únicos para arquivos
pub struct FileNameGenerator;

impl FileNameGenerator {
    pub fn generate_photo_name(person_name: &str, sequence: u32) -> String {
        let timestamp = get_timestamp();
        let clean_name = person_name.replace(' ', "_").to_lowercase();
        format!("photo_{}_{:04}_{}.ppm", clean_name, sequence, timestamp)
    }
    
    pub fn generate_model_name(accuracy: f32) -> String {
        let timestamp = get_timestamp();
        let accuracy_percent = (accuracy * 100.0) as u32;
        format!("cnn_model_acc{}%_{}.bin", accuracy_percent, timestamp)
    }
    
    pub fn generate_log_name() -> String {
        let now = chrono::Utc::now();
        format!("cnncheckin_{}.log", now.format("%Y%m%d"))
    }
}

/// Estatísticas de memória e performance
pub struct MemoryStats;

impl MemoryStats {
    pub fn get_memory_usage() -> Result<MemoryInfo, Box<dyn Error>> {
        // Implementação simplificada - em produção usaria bibliotecas específicas do SO
        Ok(MemoryInfo {
            total_mb: 0,
            used_mb: 0,
            available_mb: 0,
        })
    }
}

#[derive(Debug)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
}

/// Utilitários para manipulação de arrays
pub mod array_utils {
    use ndarray::{Array3, Array1};
    
    pub fn normalize_array(mut arr: Array3<f32>) -> Array3<f32> {
        let mean = arr.mean().unwrap_or(0.0);
        let std = arr.std(0.0);
        
        if std > 0.0 {
            arr = (arr - mean) / std;
        }
        
        arr
    }
    
    pub fn calculate_cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product = a.dot(b);
        let norm_a = (a.dot(a)).sqrt();
        let norm_b = (b.dot(b)).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    pub fn array_to_vec(arr: &Array3<f32>) -> Vec<f32> {
        arr.as_slice().unwrap_or(&[]).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_yuv_to_rgb() {
        let (r, g, b) = yuv_to_rgb(128.0, 128.0, 128.0);
        // Valores esperados para conversão YUV neutra
        assert!(r > 100 && r < 150);
        assert!(g > 100 && g < 150);
        assert!(b > 100 && b < 150);
    }
    
    #[test]
    fn test_timestamp_functions() {
        let timestamp = get_timestamp();
        assert!(timestamp > 0);
        
        let formatted = format_timestamp(timestamp);
        assert!(formatted.contains("UTC"));
    }
    
    #[test]
    fn test_hash_calculation() {
        let data = b"test data";
        let hash = calculate_hash(data);
        assert_eq!(hash.len(), 64); // SHA256 produz hash de 64 caracteres hex
    }
    
    #[test]
    fn test_input_validator() {
        // Testes de validação de nome
        assert!(InputValidator::validate_person_name("João Silva").is_ok());
        assert!(InputValidator::validate_person_name("").is_err());
        assert!(InputValidator::validate_person_name("A").is_err());
        
        // Testes de validação de confiança
        assert!(InputValidator::validate_confidence(0.5).is_ok());
        assert!(InputValidator::validate_confidence(-0.1).is_err());
        assert!(InputValidator::validate_confidence(1.1).is_err());
    }
    
    #[test]
    fn test_file_name_generator() {
        let photo_name = FileNameGenerator::generate_photo_name("João Silva", 1);
        assert!(photo_name.contains("joão_silva"));
        assert!(photo_name.contains("0001"));
        assert!(photo_name.ends_with(".ppm"));
        
        let model_name = FileNameGenerator::generate_model_name(0.85);
        assert!(model_name.contains("acc85%"));
        assert!(model_name.ends_with(".bin"));
    }
    
    #[test]
    fn test_compression() {
        let original_data = b"Este é um teste de compressão de dados repetitivos repetitivos repetitivos";
        
        let compressed = compress_data(original_data).unwrap();
        let decompressed = decompress_data(&compressed).unwrap();
        
        assert_eq!(original_data.to_vec(), decompressed);
        assert!(compressed.len() < original_data.len()); // Deve comprimir
    }
    
    #[test]
    fn test_system_info() {
        let info = get_system_info();
        assert!(info.num_cpus > 0);
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
    }
}