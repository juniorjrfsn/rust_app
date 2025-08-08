// projeto: lstmrnntrain
// file: src/neural/utils.rs
// Utility functions, optimizers, and error handling for neural networks





use ndarray::{Array1, Array2, ShapeError};
use serde::{Deserialize, Serialize};
use rand::Rng;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("Model configuration error: {0}")]
    ModelConfiguration(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Database error: {0}")]
    Database(#[from] postgres::Error),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Shape error: {0}")]
    Shape(String),
}

impl From<ShapeError> for TrainingError {
    fn from(err: ShapeError) -> Self {
        TrainingError::Shape(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,
    m: HashMap<String, Array1<f64>>,
    v: HashMap<String, Array1<f64>>,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        AdamOptimizer {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn set_learning_rate(&mut self, new_lr: f64) {
        self.learning_rate = new_lr;
    }

    pub fn update(&mut self, param_name: &str, gradient: &Array1<f64>) -> Array1<f64> {
        self.t += 1;
        
        let m = self.m.entry(param_name.to_string())
            .or_insert_with(|| Array1::zeros(gradient.len()));
        
        let v = self.v.entry(param_name.to_string())
            .or_insert_with(|| Array1::zeros(gradient.len()));

        *m = &*m * self.beta1 + gradient * (1.0 - self.beta1);
        *v = &*v * self.beta2 + &gradient.mapv(|x| x.powi(2)) * (1.0 - self.beta2);

        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

        let update = &m_hat / (&v_hat.mapv(|x| x.sqrt()) + self.epsilon) * self.learning_rate;
        update
    }

    pub fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    StepDecay {
        initial_rate: f64,
        decay_rate: f64,
        step_size: usize,
    },
    ExponentialDecay {
        initial_rate: f64,
        decay_rate: f64,
    },
    CosineAnnealing {
        initial_rate: f64,
        min_rate: f64,
        cycle_length: usize,
    },
}

impl LearningRateScheduler {
    pub fn get_rate(&self, epoch: usize) -> f64 {
        match self {
            LearningRateScheduler::StepDecay { initial_rate, decay_rate, step_size } => {
                initial_rate * decay_rate.powi((epoch / step_size) as i32)
            },
            LearningRateScheduler::ExponentialDecay { initial_rate, decay_rate } => {
                initial_rate * decay_rate.powi(epoch as i32)
            },
            LearningRateScheduler::CosineAnnealing { initial_rate, min_rate, cycle_length } => {
                let progress = (epoch % cycle_length) as f64 / *cycle_length as f64;
                min_rate + (initial_rate - min_rate) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
            },
        }
    }
}

pub fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
}

pub fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| val.tanh())
}

pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| val.max(0.0))
}
 

pub fn tanh_scalar(x: f64) -> f64 {
    x.tanh()
}

pub fn relu_scalar(x: f64) -> f64 {
    x.max(0.0)
}

pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len());
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n
}

pub fn mae_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len());
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>() / n
}

pub fn standardize(data: &mut Array2<f64>, means: Option<&Array1<f64>>, stds: Option<&Array1<f64>>) -> (Array1<f64>, Array1<f64>) {
    let (_rows, cols) = data.dim();
    let mut computed_means = Array1::zeros(cols);
    let mut computed_stds = Array1::zeros(cols);

    for col in 0..cols {
        let column = data.column(col);
        let mean = means.map(|m| m[col]).unwrap_or_else(|| {
            let mean_val = column.sum() / column.len() as f64;
            computed_means[col] = mean_val;
            mean_val
        });

        let std = stds.map(|s| s[col]).unwrap_or_else(|| {
            let variance = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
            let std_val = variance.sqrt().max(1e-8);
            computed_stds[col] = std_val;
            std_val
        });

        data.column_mut(col).mapv_inplace(|x| (x - mean) / std);
    }

    (computed_means, computed_stds)
}

pub fn min_max_scale(data: &mut Array2<f64>, mins: Option<&Array1<f64>>, maxs: Option<&Array1<f64>>) -> (Array1<f64>, Array1<f64>) {
    let (_rows, cols) = data.dim();
    let mut computed_mins = Array1::zeros(cols);
    let mut computed_maxs = Array1::zeros(cols);

    for col in 0..cols {
        let column = data.column(col);
        let min_val = mins.map(|m| m[col]).unwrap_or_else(|| {
            let min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            computed_mins[col] = min;
            min
        });

        let max_val = maxs.map(|m| m[col]).unwrap_or_else(|| {
            let max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            computed_maxs[col] = max;
            max
        });

        let range = max_val - min_val;
        if range > 1e-8 {
            data.column_mut(col).mapv_inplace(|x| (x - min_val) / range);
        } else {
            data.column_mut(col).mapv_inplace(|_| 0.0);
        }
    }

    (computed_mins, computed_maxs)
}

pub fn xavier_init(input_size: usize, output_size: usize) -> Array2<f64> {
    let std = (2.0 / (input_size + output_size) as f64).sqrt();
    let mut rng = rand::rng();
    Array2::from_shape_fn((output_size, input_size), |_| rng.random::<f64>() * std)
}

pub fn bias_init(size: usize) -> Array1<f64> {
    let std = 0.1;
    let mut rng = rand::rng();
    Array1::from_shape_fn(size, |_| rng.random::<f64>() * std)
}

pub fn clip_gradient_norm(gradient: &mut Array1<f64>, max_norm: f64) -> f64 {
    let norm = gradient.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        gradient.mapv_inplace(|x| x * scale);
    }
    norm
}

#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    pub start_time: std::time::Instant,
    pub epoch_times: Vec<f64>,
    pub memory_usage: Vec<usize>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        PerformanceMonitor {
            start_time: std::time::Instant::now(),
            epoch_times: Vec::new(),
            memory_usage: Vec::new(),
        }
    }

    pub fn record_epoch(&mut self, duration: f64) {
        self.epoch_times.push(duration);
    }

    pub fn get_average_epoch_time(&self) -> f64 {
        if self.epoch_times.is_empty() {
            0.0
        } else {
            self.epoch_times.iter().sum::<f64>() / self.epoch_times.len() as f64
        }
    }

    pub fn get_total_time(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn print_summary(&self) {
        println!("⏱️ [Performance] Training Summary:");
        println!("   ├── Total Time: {:.2}s", self.get_total_time());
        println!("   ├── Total Epochs: {}", self.epoch_times.len());
        println!("   ├── Avg Epoch Time: {:.2}s", self.get_average_epoch_time());
        if !self.epoch_times.is_empty() {
            let min_time = self.epoch_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_time = self.epoch_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            println!("   ├── Min Epoch Time: {:.2}s", min_time);
            println!("   └── Max Epoch Time: {:.2}s", max_time);
        }
    }
}

pub fn validate_input_data(data: &Array2<f64>, name: &str) -> Result<(), TrainingError> {
    if data.is_empty() {
        return Err(TrainingError::DataProcessing(format!("{} is empty", name)));
    }

    for (i, row) in data.axis_iter(ndarray::Axis(0)).enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value.is_nan() {
                return Err(TrainingError::DataProcessing(
                    format!("{} contains NaN at position ({}, {})", name, i, j)
                ));
            }
            if value.is_infinite() {
                return Err(TrainingError::DataProcessing(
                    format!("{} contains infinite value at position ({}, {})", name, i, j)
                ));
            }
        }
    }

    Ok(())
}

pub fn validate_targets(targets: &[f64], name: &str) -> Result<(), TrainingError> {
    if targets.is_empty() {
        return Err(TrainingError::DataProcessing(format!("{} is empty", name)));
    }

    for (i, &value) in targets.iter().enumerate() {
        if value.is_nan() {
            return Err(TrainingError::DataProcessing(
                format!("{} contains NaN at position {}", name, i)
            ));
        }
        if value.is_infinite() {
            return Err(TrainingError::DataProcessing(
                format!("{} contains infinite value at position {}", name, i)
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
        let gradient = Array1::from_vec(vec![0.1, -0.2, 0.3]);
        let update = optimizer.update("test_param", &gradient);
        assert_eq!(update.len(), 3);
        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.learning_rate, 0.01);
    }

    #[test]
    fn test_activation_functions() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let sig_result = sigmoid(&x);
        assert!(sig_result[0] < 0.5);
        assert!((sig_result[1] - 0.5).abs() < 1e-10);
        assert!(sig_result[2] > 0.5);
        let tanh_result = tanh(&x);
        assert!(tanh_result[0] < 0.0);
        assert!((tanh_result[1]).abs() < 1e-10);
        assert!(tanh_result[2] > 0.0);
        let relu_result = relu(&x);
        assert_eq!(relu_result[0], 0.0);
        assert_eq!(relu_result[1], 0.0);
        assert_eq!(relu_result[2], 1.0);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let scheduler = LearningRateScheduler::StepDecay {
            initial_rate: 0.1,
            decay_rate: 0.5,
            step_size: 10,
        };
        assert_eq!(scheduler.get_rate(5), 0.1);
        assert_eq!(scheduler.get_rate(10), 0.05);
        assert_eq!(scheduler.get_rate(20), 0.025);
    }

    #[test]
    fn test_loss_functions() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.9, 3.1];
        let mse = mse_loss(&predictions, &targets);
        assert!(mse > 0.0);
        let mae = mae_loss(&predictions, &targets);
        assert!(mae > 0.0);
        assert!(mae < mse);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradient = Array1::from_vec(vec![3.0, 4.0]);
        let norm = clip_gradient_norm(&mut gradient, 2.0);
        assert!((norm - 5.0).abs() < 1e-10);
        assert!((gradient[0] - 1.2).abs() < 1e-10);
        assert!((gradient[1] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn test_data_validation() {
        let valid_data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(validate_input_data(&valid_data, "test").is_ok());
        let invalid_data = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 3.0, 4.0]).unwrap();
        assert!(validate_input_data(&invalid_data, "test").is_err());
        let valid_targets = vec![1.0, 2.0, 3.0];
        assert!(validate_targets(&valid_targets, "test").is_ok());
        let invalid_targets = vec![1.0, f64::INFINITY, 3.0];
        assert!(validate_targets(&invalid_targets, "test").is_err());
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        monitor.record_epoch(1.5);
        monitor.record_epoch(1.2);
        monitor.record_epoch(1.8);
        assert_eq!(monitor.epoch_times.len(), 3);
        assert!((monitor.get_average_epoch_time() - 1.5).abs() < 1e-10);
    }
}
