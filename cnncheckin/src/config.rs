// file: cnncheckin/src/config.rs
// Módulo de configuração do sistema

 // file: cnncheckin/src/config.rs
// Módulo de configuração do sistema

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub camera: CameraConfig,
    pub model: ModelConfig,
    pub paths: PathsConfig,
    pub recognition: RecognitionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub device_path: String,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
    pub preferred_formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_size: [usize; 3],
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub validation_split: f32,
    pub early_stopping_patience: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub photos_dir: String,
    pub training_dir: String,
    pub models_dir: String,
    pub temp_dir: String,
    pub logs_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    pub confidence_threshold: f32,
    pub similarity_threshold: f32,
    pub max_faces_per_frame: usize,
    pub face_detection_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                host: "localhost".to_string(),
                port: 5432,
                database: "cnncheckin".to_string(),
                username: "postgres".to_string(),
                password: "postgres".to_string(),
                max_connections: 10,
            },
            camera: CameraConfig {
                device_path: "/dev/video0".to_string(),
                width: 640,
                height: 480,
                fps: 30,
                preferred_formats: vec!["RGB3".to_string(), "YUYV".to_string()],
            },
            model: ModelConfig {
                input_size: [3, 128, 128],
                batch_size: 32,
                epochs: 50,
                learning_rate: 0.001,
                validation_split: 0.2,
                early_stopping_patience: 10,
            },
            paths: PathsConfig {
                photos_dir: "../../dados/fotos_webcam".to_string(),
                training_dir: "../../dados/fotos_treino".to_string(),
                models_dir: "../../dados/modelos".to_string(),
                temp_dir: "../../dados/temp".to_string(),
                logs_dir: "../../dados/logs".to_string(),
            },
            recognition: RecognitionConfig {
                confidence_threshold: 0.7,
                similarity_threshold: 0.8,
                max_faces_per_frame: 5,
                face_detection_model: "simple".to_string(),
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let config_path = "config.toml";
        
        if Path::new(config_path).exists() {
            let content = fs::read_to_string(config_path)?;
            let config: Config = toml::from_str(&content)?;
            println!("⚙️ Configuração carregada de: {}", config_path);
            Ok(config)
        } else {
            println!("📄 Arquivo de configuração não encontrado. Criando padrão...");
            let default_config = Config::default();
            default_config.save()?;
            Ok(default_config)
        }
    }
    
    pub fn save(&self) -> Result<(), Box<dyn Error>> {
        let config_toml = toml::to_string_pretty(self)?;
        fs::write("config.toml", config_toml)?;
        println!("💾 Configuração salva em: config.toml");
        Ok(())
    }
    
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        // Validar configurações críticas
        if self.database.host.is_empty() {
            return Err("Host do banco de dados não pode estar vazio".into());
        }
        
        if self.database.port == 0 {
            return Err("Porta do banco de dados inválida".into());
        }
        
        if self.camera.width == 0 || self.camera.height == 0 {
            return Err("Dimensões da câmera inválidas".into());
        }
        
        if self.model.batch_size == 0 {
            return Err("Batch size deve ser maior que zero".into());
        }
        
        if self.model.epochs == 0 {
            return Err("Número de épocas deve ser maior que zero".into());
        }
        
        if self.recognition.confidence_threshold < 0.0 || self.recognition.confidence_threshold > 1.0 {
            return Err("Threshold de confiança deve estar entre 0.0 e 1.0".into());
        }
        
        if self.recognition.similarity_threshold < 0.0 || self.recognition.similarity_threshold > 1.0 {
            return Err("Threshold de similaridade deve estar entre 0.0 e 1.0".into());
        }
        
        println!("✅ Configuração validada com sucesso");
        Ok(())
    }
    
    pub fn get_database_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.database.username,
            self.database.password,
            self.database.host,
            self.database.port,
            self.database.database
        )
    }
    
    pub fn ensure_directories(&self) -> Result<(), Box<dyn Error>> {
        let dirs = [
            &self.paths.photos_dir,
            &self.paths.training_dir,
            &self.paths.models_dir,
            &self.paths.temp_dir,
            &self.paths.logs_dir,
        ];
        
        for dir in &dirs {
            if !Path::new(dir).exists() {
                fs::create_dir_all(dir)?;
                println!("📁 Diretório criado: {}", dir);
            }
        }
        
        Ok(())
    }
    
    pub fn get_model_input_shape(&self) -> (usize, usize, usize) {
        (
            self.model.input_size[0],
            self.model.input_size[1],
            self.model.input_size[2],
        )
    }
    
    pub fn is_debug_mode(&self) -> bool {
        std::env::var("RUST_LOG").unwrap_or_default().contains("debug")
    }
    
    pub fn print_summary(&self) {
        println!("📋 Resumo da Configuração:");
        println!("  🗄️ Banco: {}@{}:{}", 
                self.database.username, self.database.host, self.database.port);
        println!("  📷 Câmera: {}x{} @ {}fps", 
                self.camera.width, self.camera.height, self.camera.fps);
        println!("  🧠 Modelo: batch_size={}, epochs={}", 
                self.model.batch_size, self.model.epochs);
        println!("  🎯 Reconhecimento: confiança={:.2}, similaridade={:.2}", 
                self.recognition.confidence_threshold, self.recognition.similarity_threshold);
    }
}

// Funcões utilitárias para configuração
pub fn create_default_config() -> Result<(), Box<dyn Error>> {
    let config = Config::default();
    config.save()?;
    println!("✅ Arquivo de configuração padrão criado");
    Ok(())
}

pub fn validate_config_file(path: &str) -> Result<bool, Box<dyn Error>> {
    if !Path::new(path).exists() {
        return Ok(false);
    }
    
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    config.validate()?;
    
    Ok(true)
}

pub fn backup_config(backup_name: Option<&str>) -> Result<(), Box<dyn Error>> {
    let config_path = "config.toml";
    
    if !Path::new(config_path).exists() {
        return Err("Arquivo de configuração não existe".into());
    }
    
    let backup_path = match backup_name {
        Some(name) => format!("config_{}.toml", name),
        None => {
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            format!("config_backup_{}.toml", timestamp)
        }
    };
    
    fs::copy(config_path, &backup_path)?;
    println!("💾 Backup da configuração salvo em: {}", backup_path);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.database.host, "localhost");
        assert_eq!(config.camera.width, 640);
        assert_eq!(config.model.batch_size, 32);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());
        
        // Teste com configuração inválida
        config.camera.width = 0;
        assert!(config.validate().is_err());
        
        // Corrigir e testar novamente
        config.camera.width = 640;
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_database_url() {
        let config = Config::default();
        let url = config.get_database_url();
        assert!(url.contains("postgresql://"));
        assert!(url.contains("localhost:5432"));
    }
    
    #[test]
    fn test_model_input_shape() {
        let config = Config::default();
        let (c, h, w) = config.get_model_input_shape();
        assert_eq!(c, 3);
        assert_eq!(h, 128);
        assert_eq!(w, 128);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("[database]"));
        assert!(toml_str.contains("[camera]"));
        
        // Teste de deserialização
        let parsed_config: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed_config.database.host, config.database.host);
    }
    
    #[test]
    fn test_config_file_operations() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        // Salvar configuração
        let config = Config::default();
        let config_toml = toml::to_string_pretty(&config).unwrap();
        fs::write(&config_path, config_toml).unwrap();
        
        // Validar arquivo
        let is_valid = validate_config_file(config_path.to_str().unwrap()).unwrap();
        assert!(is_valid);
    }
}