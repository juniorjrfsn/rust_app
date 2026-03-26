// projeto cnncheckin
// file: cnncheckin/src/config.rs
// Módulo de configuração do sistema
// src/config.rs — Configuração do sistema

use serde::{Deserialize, Serialize};
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
    pub host: Option<String>,
    pub port: Option<u16>,
    pub database: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub max_connections: Option<u32>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub device_index: u32,
    pub width: usize,
    pub height: usize,
    pub fps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_height: usize,
    pub input_width: usize,
    pub input_channels: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub k_neighbors: usize,
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                host: Some("localhost".to_string()),
                port: Some(5432),
                database: Some("cnncheckin".to_string()),
                username: Some("postgres".to_string()),
                password: Some("postgres".to_string()),
                max_connections: Some(10),
                path: Some("cnncheckin.db".to_string()),
            },
            camera: CameraConfig {
                device_index: 0,
                width: 640,
                height: 480,
                fps: 30,
            },
            model: ModelConfig {
                input_height: 128,
                input_width: 128,
                input_channels: 3,
                batch_size: 32,
                epochs: 50,
                learning_rate: 0.001,
                k_neighbors: 5,
            },
            paths: PathsConfig {
                photos_dir: "dados/fotos_webcam".to_string(),
                training_dir: "dados/fotos_treino".to_string(),
                models_dir: "dados/modelos".to_string(),
                temp_dir: "dados/temp".to_string(),
                logs_dir: "dados/logs".to_string(),
            },
            recognition: RecognitionConfig {
                confidence_threshold: 0.70,
                similarity_threshold: 0.80,
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = "config.toml";
        if Path::new(config_path).exists() {
            let content = fs::read_to_string(config_path)?;
            Ok(toml::from_str(&content)?)
        } else {
            let config = Self::default();
            config.save()?;
            println!("📝 Arquivo config.toml criado com valores padrão.");
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        fs::write("config.toml", toml::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.database.path.is_none() && self.database.host.is_none() {
            return Err("Caminho ou host do banco de dados não pode ser vazio.".into());
        }
        if self.camera.width == 0 || self.camera.height == 0 {
            return Err("Dimensões da câmera inválidas.".into());
        }
        if self.model.batch_size == 0 || self.model.epochs == 0 {
            return Err("Configuração do modelo inválida.".into());
        }
        Ok(())
    }

    pub fn ensure_directories(&self) -> Result<(), Box<dyn std::error::Error>> {
        for dir in [
            &self.paths.photos_dir,
            &self.paths.training_dir,
            &self.paths.models_dir,
            &self.paths.temp_dir,
            &self.paths.logs_dir,
        ] {
            if !Path::new(dir).exists() {
                fs::create_dir_all(dir)?;
            }
        }
        Ok(())
    }

    /// Retorna as dimensões de entrada do modelo como (channels, height, width)
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (
            self.model.input_channels,
            self.model.input_height,
            self.model.input_width,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.camera.width, 640);
        assert_eq!(config.camera.height, 480);
        assert_eq!(config.model.epochs, 50);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        assert_eq!(deserialized.database.path, config.database.path);
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }
}