// file: cnncheckin/src/config.rs
// Módulo de configuração do sistema

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_size: [usize; 3],
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
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
            },
            model: ModelConfig {
                input_size: [3, 128, 128],
                batch_size: 32,
                epochs: 50,
                learning_rate: 0.001,
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
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        fs::write("config.toml", toml::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.database.host.is_empty() || self.database.port == 0 {
            return Err("Configuração de banco de dados inválida".into());
        }
        if self.camera.width == 0 || self.camera.height == 0 {
            return Err("Dimensões da câmera inválidas".into());
        }
        if self.model.batch_size == 0 || self.model.epochs == 0 {
            return Err("Configuração do modelo inválida".into());
        }
        Ok(())
    }

    pub fn get_database_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.database.username, self.database.password, self.database.host,
            self.database.port, self.database.database
        )
    }

    pub fn ensure_directories(&self) -> Result<(), Box<dyn std::error::Error>> {
        for dir in [&self.paths.photos_dir, &self.paths.training_dir, &self.paths.models_dir, &self.paths.temp_dir, &self.paths.logs_dir] {
            if !Path::new(dir).exists() {
                fs::create_dir_all(dir)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_load_save() {
        let config = Config::default();
        config.save().unwrap();
        let loaded = Config::load().unwrap();
        assert_eq!(loaded.database.host, "localhost");
    }
}