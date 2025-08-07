// projeto: lstmfiletrain
// file: src/neural/metrics.rs
// Defines the structure for storing training metrics.
 

 
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub asset: String,
    pub source: String,
    pub train_loss: f64,
    pub val_loss: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub directional_accuracy: f64,
    pub timestamp: String,
}

impl TrainingMetrics {
    pub fn new(asset: String, source: String) -> Self {
        Self {
            asset,
            source,
            train_loss: 0.0,
            val_loss: 0.0,
            rmse: 0.0,
            mae: 0.0,
            mape: 0.0,
            directional_accuracy: 0.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}