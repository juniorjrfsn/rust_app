// file: src/model/mod.rs
// Módulo de modelo de reconhecimento facial

mod cnn;
mod manager;
mod metadata;

pub use cnn::{FaceRecognitionModel, ModelArchitecture};
pub use manager::ModelManager;
pub use metadata::{ModelMetadata, TrainedModel};

use ndarray::Array3;
use crate::error::Result;

/// Trait para modelos de reconhecimento facial
pub trait FaceModel: Send + Sync {
    /// Prediz a classe de uma face
    fn predict(&self, image: &Array3<f32>) -> Result<(usize, f32)>;
    
    /// Cria embedding de uma face
    fn create_embedding(&self, image: &Array3<f32>) -> Result<Vec<f32>>;
    
    /// Retorna nomes das classes
    fn class_names(&self) -> &[String];
    
    /// Número de classes
    fn num_classes(&self) -> usize {
        self.class_names().len()
    }
}

/// Resultado de predição
#[derive(Debug, Clone)]
pub struct Prediction {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub embedding: Vec<f32>,
}

impl Prediction {
    pub fn new(class_id: usize, class_name: String, confidence: f32, embedding: Vec<f32>) -> Self {
        Self {
            class_id,
            class_name,
            confidence,
            embedding,
        }
    }
    
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Configuração de arquitetura do modelo
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub input_size: (usize, usize, usize), // (channels, height, width)
    pub num_classes: usize,
    pub hidden_layers: Vec<usize>,
    pub dropout_rate: f32,
}

impl ModelConfig {
    pub fn default_cnn() -> Self {
        Self {
            input_size: (3, 128, 128),
            num_classes: 10,
            hidden_layers: vec![128, 256, 512],
            dropout_rate: 0.3,
        }
    }
    
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        self
    }
}