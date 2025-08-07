// projeto: lstmfiletrain
// file: src/neural/utils.rs
// Contains utility functions, optimizer, and error handling.

 


 
use ndarray::{Array1, Array2};
use std::error::Error;

#[derive(Debug)]
pub enum TrainingError {
    ModelError(String),
    DatabaseError(Box<dyn Error + Send + Sync>),
    DataProcessing(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TrainingError::ModelError(msg) => write!(f, "Model Error: {}", msg),
            TrainingError::DatabaseError(err) => write!(f, "Database Error: {}", err),
            TrainingError::DataProcessing(msg) => write!(f, "Data Processing Error: {}", msg),
        }
    }
}

impl Error for TrainingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TrainingError::DatabaseError(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

impl From<postgres::Error> for TrainingError {
    fn from(err: postgres::Error) -> Self {
        TrainingError::DatabaseError(Box::new(err))
    }
}

impl From<serde_json::Error> for TrainingError {
    fn from(err: serde_json::Error) -> Self {
        TrainingError::DatabaseError(Box::new(err))
    }
}

pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Array1<f64>,
    v: Array1<f64>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        println!("🛠️ [AdamOptimizer] Initializing new optimizer...");
        let optimizer = AdamOptimizer {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Array1::zeros(0),
            v: Array1::zeros(0),
            t: 0,
        };
        println!("✅ [AdamOptimizer] Optimizer initialized successfully.");
        optimizer
    }

    pub fn update(&mut self, weights: &mut Array1<f64>, grads: &Array1<f64>) -> Result<(), TrainingError> {
        println!("🔄 [AdamOptimizer] Starting weight update...");
        if self.m.len() != weights.len() {
            self.m = Array1::zeros(weights.len());
            self.v = Array1::zeros(weights.len());
        }
        self.t += 1;

        for i in 0..weights.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
        }

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));

        for i in 0..weights.len() {
            weights[i] -= self.learning_rate * m_hat[i] / (v_hat[i].sqrt() + self.epsilon);
        }
        println!("✅ [AdamOptimizer] Weight update completed.");
        Ok(())
    }
}

pub fn sigmoid(x: f64) -> f64 {
    if x > 500.0 {
        1.0
    } else if x < -500.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

pub fn tanh(x: f64) -> f64 {
    if x > 500.0 {
        1.0
    } else if x < -500.0 {
        -1.0
    } else {
        x.tanh()
    }
}

pub fn apply_l2_regularization(weights: &Array2<f64>, l2_weight: f64) -> f64 {
    println!("🔧 [Utils] Applying L2 regularization with weight: {}", l2_weight);
    let result = l2_weight * weights.mapv(|x| x * x).sum();
    println!("✅ [Utils] L2 regularization applied: {}", result);
    result
}

pub fn clip_gradients_by_norm(grads: &mut Array2<f64>, clip_norm: f64) -> f64 {
    println!("🔧 [Utils] Clipping gradients with norm: {}", clip_norm);
    let norm = (grads.mapv(|x| x * x).sum()).sqrt();
    if norm > clip_norm && norm > 0.0 {
        let scale = clip_norm / norm;
        grads.mapv_inplace(|x| x * scale);
    }
    println!("✅ [Utils] Gradients clipped, norm: {}", norm);
    norm
}