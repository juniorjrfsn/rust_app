// projeto: lstmrnntrain
// file: src/neural/utils.rs
// Utility functions, optimizers, and error handling for neural networks

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Database error: {0}")]
    Database(#[from] postgres::Error),
    
    #[error("Model configuration error: {0}")]
    ModelConfiguration(String),
    
    #[error("Data processing error: {0}")]
    DataProcessing(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("Array shape error: {0}")]
    ArrayShape(#[from] ndarray::ShapeError),
}

/// Adam optimizer implementation
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m: HashMap<String, Array1<f64>>,  // First moment estimates
    v: HashMap<String, Array1<f64>>,  // Second moment estimates
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

    pub fn step(&mut self, param_name: &str, param: &mut Array1<f64>, grad: &Array1<f64>) {
        self.t += 1;
        
        // Initialize moments if not present
        if !self.m.contains_key(param_name) {
            self.m.insert(param_name.to_string(), Array1::zeros(param.len()));
            self.v.insert(param_name.to_string(), Array1::zeros(param.len()));
        }

        let m = self.m.get_mut(param_name).unwrap();
        let v = self.v.get_mut(param_name).unwrap();

        // Update biased first moment estimate
        *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
        
        // Update biased second raw moment estimate
        *v = &*v * self.beta2 + &grad.mapv(|x| x.powi(2)) * (1.0 - self.beta2);
        
        // Compute bias-corrected first moment estimate
        let m_hat = m / (1.0 - self.beta1.powi(self.t as i32));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = v / (1.0 - self.beta2.powi(self.t as i32));
        
        // Update parameters
        let update = &m_hat / (&v_hat.mapv(|x| x.sqrt()) + self.epsilon);
        *param = &*param - &update * self.learning_rate;
    }

    pub fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

/// Activation functions
pub fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
}

pub fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| val.tanh())
}

pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| val.max(0.0))
}

pub fn leaky_relu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|val| if val > 0.0 { val } else { alpha * val })
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals = x.mapv(|val| (val - max_val).exp());
    let sum_exp = exp_vals.sum();
    exp_vals / sum_exp
}

/// Activation function derivatives
pub fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    let sig = sigmoid(x);
    &sig * &(1.0 - &sig)
}

pub fn tanh_derivative(x: &Array1<f64>) -> Array1<f64> {
    let tanh_x = tanh(x);
    1.0 - &tanh_x * &tanh_x
}

pub fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 })
}

/// Loss functions
pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n
}

pub fn mae_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>() / n
}

pub fn huber_loss(predictions: &[f64], targets: &[f64], delta: f64) -> f64 {
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| {
            let diff = (p - t).abs();
            if diff <= delta {
                0.5 * diff.powi(2)
            } else {
                delta * (diff - 0.5 * delta)
            }
        })
        .sum::<f64>() / n
}

/// Weight initialization strategies
pub fn xavier_uniform(fan_in: usize, fan_out: usize) -> f64 {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    rand::random::<f64>() * 2.0 * limit - limit
}

pub fn xavier_normal(fan_in: usize, fan_out: usize) -> f64 {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    rand_distr::Normal::new(0.0, std)
        .map(|dist| rand_distr::Distribution::sample(&dist, &mut rand::thread_rng()))
        .unwrap_or(0.0)
}

pub fn he_uniform(fan_in: usize) -> f64 {
    let limit = (6.0 / fan_in as f64).sqrt();
    rand::random::<f64>() * 2.0 * limit - limit
}

pub fn he_normal(fan_in: usize) -> f64 {
    let std = (2.0 / fan_in as f64).sqrt();
    rand_distr::Normal::new(0.0, std)
        .map(|dist| rand_distr::Distribution::sample(&dist, &mut rand::thread_rng()))
        .unwrap_or(0.0)
}

/// Learning rate scheduling
#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    Constant { rate: f64 },
    StepDecay { initial_rate: f64, decay_rate: f64, step_size: usize },
    ExponentialDecay { initial_rate: f64, decay_rate: f64 },
    CosineAnnealing { initial_rate: f64, min_rate: f64, t_max: usize },
}

impl LearningRateScheduler {
    pub fn get_rate(&self, epoch: usize) -> f64 {
        match self {
            LearningRateScheduler::Constant { rate } => *rate,
            LearningRateScheduler::StepDecay { initial_rate, decay_rate, step_size } => {
                let steps = epoch / step_size;
                initial_rate * decay_rate.powi(steps as i32)
            },
            LearningRateScheduler::ExponentialDecay { initial_rate, decay_rate } => {
                initial_rate * decay_rate.powi(epoch as i32)
            },
            LearningRateScheduler::CosineAnnealing { initial_rate, min_rate, t_max } => {
                let t = (epoch % t_max) as f64;
                min_rate + (initial_rate - min_rate) * 
                    (1.0 + (std::f64::consts::PI * t / *t_max as f64).cos()) / 2.0
            },
        }
    }
}

/// Data normalization utilities
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub fitted: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler {
            mean: Array1::zeros(0),
            std: Array1::zeros(0),
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        let n_samples = data.nrows() as f64;
        let n_features = data.ncols();

        self.mean = Array1::zeros(n_features);
        self.std = Array1::zeros(n_features);

        // Calculate mean
        for i in 0..n_features {
            self.mean[i] = data.column(i).sum() / n_samples;
        }

        // Calculate standard deviation
        for i in 0..n_features {
            let variance = data.column(i)
                .iter()
                .map(|&x| (x - self.mean[i]).powi(2))
                .sum::<f64>() / n_samples;
            self.std[i] = variance.sqrt().max(1e-8); // Prevent division by zero
        }

        self.fitted = true;
    }

    pub fn transform(&self, data: &mut Array2<f64>) -> Result<(), TrainingError> {
        if !self.fitted {
            return Err(TrainingError::DataProcessing(
                "Scaler must be fitted before transform".to_string()
            ));
        }

        for i in 0..data.ncols() {
            let mut col = data.column_mut(i);
            col.mapv_inplace(|x| (x - self.mean[i]) / self.std[i]);
        }

        Ok(())
    }

    pub fn fit_transform(&mut self, data: &mut Array2<f64>) -> Result<(), TrainingError> {
        self.fit(data);
        self.transform(data)
    }

    pub fn inverse_transform(&self, data: &mut Array2<f64>) -> Result<(), TrainingError> {
        if !self.fitted {
            return Err(TrainingError::DataProcessing(
                "Scaler must be fitted before inverse_transform".to_string()
            ));
        }

        for i in 0..data.ncols() {
            let mut col = data.column_mut(i);
            col.mapv_inplace(|x| x * self.std[i] + self.mean[i]);
        }

        Ok(())
    }
}

/// Min-Max scaler
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    pub min: Array1<f64>,
    pub max: Array1<f64>,
    pub feature_range: (f64, f64),
    pub fitted: bool,
}

impl MinMaxScaler {
    pub fn new(feature_range: (f64, f64)) -> Self {
        MinMaxScaler {
            min: Array1::zeros(0),
            max: Array1::zeros(0),
            feature_range,
            fitted: false,
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        let n_features = data.ncols();
        self.min = Array1::zeros(n_features);
        self.max = Array1::zeros(n_features);

        for i in 0..n_features {
            let col = data.column(i);
            self.min[i] = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            self.max[i] = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }

        self.fitted = true;
    }

    pub fn transform(&self, data: &mut Array2<f64>) -> Result<(), TrainingError> {
        if !self.fitted {
            return Err(TrainingError::DataProcessing(
                "Scaler must be fitted before transform".to_string()
            ));
        }

        let (min_range, max_range) = self.feature_range;
        let scale = max_range - min_range;

        for i in 0..data.ncols() {
            let mut col = data.column_mut(i);
            let data_range = self.max[i] - self.min[i];
            if data_range > 1e-8 {
                col.mapv_inplace(|x| {
                    let normalized = (x - self.min[i]) / data_range;
                    min_range + normalized * scale
                });
            }
        }

        Ok(())
    }
}

/// Utility functions for array operations
pub fn shuffle_arrays<T: Clone>(arrays: &mut [Vec<T>]) {
    if arrays.is_empty() {
        return;
    }

    let len = arrays[0].len();
    for array in arrays.iter() {
        assert_eq!(array.len(), len, "All arrays must have the same length");
    }

    for i in (1..len).rev() {
        let j = (rand::random::<f64>() * (i + 1) as f64) as usize;
        for array in arrays.iter_mut() {
            array.swap(i, j);
        }
    }
}

pub fn create_batches<T: Clone>(data: &[T], batch_size: usize) -> Vec<Vec<T>> {
    data.chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Progress tracking utilities
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub elapsed_time: std::time::Duration,
}

impl TrainingProgress {
    pub fn print_progress(&self) {
        let progress_percent = (self.epoch as f64 / self.total_epochs as f64) * 100.0;
        let elapsed_secs = self.elapsed_time.as_secs_f64();
        
        print!("\r🎓 [Training] Epoch {}/{} ({:.1}%) | Train Loss: {:.6} | LR: {:.6} | Time: {:.1}s",
               self.epoch, self.total_epochs, progress_percent, 
               self.train_loss, self.learning_rate, elapsed_secs);

        if let Some(val_loss) = self.val_loss {
            print!(" | Val Loss: {:.6}", val_loss);
        }

        if self.epoch == self.total_epochs {
            println!(); // New line at the end
        }
    }
}

/// Model checkpointing utilities
pub struct ModelCheckpoint {
    pub filepath: String,
    pub save_best_only: bool,
    pub monitor: String,
    pub mode: String,
    pub best_score: Option<f64>,
}

impl ModelCheckpoint {
    pub fn new(filepath: String, monitor: String, mode: String, save_best_only: bool) -> Self {
        ModelCheckpoint {
            filepath,
            save_best_only,
            monitor,
            mode,
            best_score: None,
        }
    }

    pub fn should_save(&mut self, current_score: f64) -> bool {
        if !self.save_best_only {
            return true;
        }

        match self.best_score {
            None => {
                self.best_score = Some(current_score);
                true
            },
            Some(best) => {
                let is_better = match self.mode.as_str() {
                    "min" => current_score < best,
                    "max" => current_score > best,
                    _ => false,
                };

                if is_better {
                    self.best_score = Some(current_score);
                    true
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
        let mut param = Array1::from(vec![1.0, 2.0, 3.0]);
        let grad = Array1::from(vec![0.1, 0.2, 0.1]);
        
        let initial_param = param.clone();
        optimizer.step("test_param", &mut param, &grad);
        
        // Parameters should have changed
        assert_ne!(param, initial_param);
    }

    #[test]
    fn test_activation_functions() {
        let x = Array1::from(vec![-1.0, 0.0, 1.0]);
        
        let sig = sigmoid(&x);
        assert!(sig.iter().all(|&val| val >= 0.0 && val <= 1.0));
        
        let tanh_result = tanh(&x);
        assert!(tanh_result.iter().all(|&val| val >= -1.0 && val <= 1.0));
        
        let relu_result = relu(&x);
        assert_eq!(relu_result, Array1::from(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_standard_scaler() {
        let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut scaler = StandardScaler::new();
        
        scaler.fit_transform(&mut data).unwrap();
        
        // Check that columns are approximately normalized
        for col in 0..data.ncols() {
            let col_mean = data.column(col).mean().unwrap();
            let col_std = data.column(col).std(0.0);
            assert!((col_mean).abs() < 1e-10);  // Mean should be close to 0
            assert!((col_std - 1.0).abs() < 1e-10);  // Std should be close to 1
        }
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let scheduler = LearningRateScheduler::StepDecay {
            initial_rate: 0.1,
            decay_rate: 0.5,
            step_size: 10,
        };
        
        assert_eq!(scheduler.get_rate(0), 0.1);
        assert_eq!(scheduler.get_rate(9), 0.1);
        assert_eq!(scheduler.get_rate(10), 0.05);
        assert_eq!(scheduler.get_rate(20), 0.025);
    }
}