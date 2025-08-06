// projeto: lstmfiletrain
// file: src/neural/utils.rs





use ndarray::{Array1, Array2};
use thiserror::Error;
use postgres::Error;

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Model error: {0}")]
    ModelError(String),
}

impl From<ndarray::ShapeError> for TrainingError {
    fn from(err: ndarray::ShapeError) -> Self {
        TrainingError::DataProcessing(err.to_string())
    }
}

impl From<Error> for TrainingError {
    fn from(err: Error) -> Self {
        TrainingError::DatabaseError(err.to_string())
    }
}

pub fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

pub fn mse_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>() / predictions.len() as f32
}

pub fn mae_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f32>() / predictions.len() as f32
}

pub fn mape_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| if t.abs() > 1e-6 { ((p - t).abs() / t.abs()) * 100.0 } else { 0.0 })
        .sum::<f32>() / predictions.len() as f32
}

pub fn directional_accuracy(predictions: &[f32], targets: &[f32]) -> f32 {
    let mut correct = 0;
    for i in 1..predictions.len() {
        let pred_direction = (predictions[i] - predictions[i-1]).signum();
        let true_direction = (targets[i] - targets[i-1]).signum();
        if pred_direction == true_direction {
            correct += 1;
        }
    }
    (correct as f32) / (predictions.len() as f32 - 1.0)
}

pub fn apply_l2_regularization<T: ndarray::Dimension>(weights: &Array2<f32>, gradients: &mut Array2<f32>, l2_weight: f32) {
    if l2_weight > 0.0 {
        for (g, w) in gradients.iter_mut().zip(weights.iter()) {
            *g += l2_weight * w;
        }
    }
}

pub fn clip_gradients_by_norm(gradients: &mut Array1<f32>, max_norm: f32) {
    let total_norm: f32 = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
    if total_norm > max_norm && total_norm > 1e-6 {
        let scale = max_norm / total_norm;
        gradients.iter_mut().for_each(|g| *g *= scale);
    }
}

#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamOptimizer {
    pub fn new(num_params: usize, learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        AdamOptimizer {
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }

    pub fn update(&mut self, weights: &mut [f32], gradients: &[f32]) -> Result<(), TrainingError> {
        if weights.len() != gradients.len() || weights.len() != self.m.len() {
            return Err(TrainingError::ModelError("Mismatched dimensions".to_string()));
        }
        self.t += 1;
        let t_f = self.t as f32;
        let beta1_pow = self.beta1.powf(t_f);
        let beta2_pow = self.beta2.powf(t_f);

        for i in 0..weights.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];
            let m_hat = self.m[i] / (1.0 - beta1_pow);
            let v_hat = self.v[i] / (1.0 - beta2_pow);
            weights[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        Ok(())
    }
}