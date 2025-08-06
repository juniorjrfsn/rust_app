// projeto: lstmfiletrain
// file: src/neural/metrics.rs



use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub asset: String,
    pub source: String,
    pub train_loss: f32,
    pub val_loss: f32,
    pub rmse: f32,
    pub mae: f32,
    pub mape: f32,
    pub directional_accuracy: f32,
    pub timestamp: String,
}