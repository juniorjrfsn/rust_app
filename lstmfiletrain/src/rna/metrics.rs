// projeto: lstmfiletrain
// file: src/rna/metrics.rs


 
  

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub asset: String,
    pub source: String,
    pub final_loss: f32,
    pub final_val_loss: f32,
    pub directional_accuracy: f32,
    pub mape: f32,
    pub rmse: f32,
    pub epochs_trained: usize,
    pub training_time: f64,
    pub timestamp: String,
}

pub fn calculate_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    (predictions.iter().zip(targets).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / n).sqrt()
}

pub fn calculate_mape(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    predictions.iter().zip(targets).map(|(p, t)| ((p - t).abs() / t.abs().max(1e-8))).sum::<f32>() / n
}

pub fn calculate_directional_accuracy(predictions: &[f32], targets: &[f32], sequences: &[Vec<f32>]) -> f32 {
    let mut correct = 0;
    for (i, (pred, target)) in predictions.iter().zip(targets).enumerate() {
        let last_close = sequences[i][0];
        let pred_dir = if *pred > last_close { 1.0 } else { -1.0 };
        let actual_dir = if *target > last_close { 1.0 } else { -1.0 };
        if pred_dir == actual_dir {
            correct += 1;
        }
    }
    correct as f32 / predictions.len() as f32
}
