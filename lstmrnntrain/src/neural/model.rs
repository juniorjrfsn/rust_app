// projeto: lstmrnntrain
// file: src/neural/model.rs
// Multi-model neural network implementation with LSTM, RNN, MLP, and CNN


// projeto: lstmrnntrain
// file: src/neural/model.rs
// Multi-model neural network implementation with LSTM, RNN, MLP, and CNN

use chrono::Utc;
use ndarray::{Array1, Array2, Array3, Zip, array};
use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::neural::utils::{AdamOptimizer, TrainingError, sigmoid, tanh, relu, relu_scalar};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    LSTM,
    RNN,
    MLP,
    CNN,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmLayerWeights {
    pub w_ii: Array2<f64>,  // Input gate input weights
    pub w_if: Array2<f64>,  // Forget gate input weights
    pub w_ig: Array2<f64>,  // Cell gate input weights
    pub w_io: Array2<f64>,  // Output gate input weights
    pub w_hi: Array2<f64>,  // Input gate hidden weights
    pub w_hf: Array2<f64>,  // Forget gate hidden weights
    pub w_hg: Array2<f64>,  // Cell gate hidden weights
    pub w_ho: Array2<f64>,  // Output gate hidden weights
    pub b_i: Array1<f64>,   // Input gate bias
    pub b_f: Array1<f64>,   // Forget gate bias
    pub b_g: Array1<f64>,   // Cell gate bias
    pub b_o: Array1<f64>,   // Output gate bias
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RnnLayerWeights {
    pub w_ih: Array2<f64>,  // Input to hidden weights
    pub w_hh: Array2<f64>,  // Hidden to hidden weights
    pub b_h: Array1<f64>,   // Hidden bias
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpLayerWeights {
    pub w: Array2<f64>,     // Layer weights
    pub b: Array1<f64>,     // Layer bias
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnLayerWeights {
    pub kernels: Array3<f64>,  // Convolution kernels
    pub bias: Array1<f64>,     // Bias terms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    pub asset: String,
    pub model_type: ModelType,
    pub lstm_layers: Vec<LstmLayerWeights>,
    pub rnn_layers: Vec<RnnLayerWeights>,
    pub mlp_layers: Vec<MlpLayerWeights>,
    pub cnn_layers: Vec<CnnLayerWeights>,
    pub final_layer: MlpLayerWeights,
    pub closing_mean: f64,
    pub closing_std: f64,
    pub seq_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub feature_dim: usize,
    pub epoch: usize,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub directional_accuracy: f64,
    pub r_squared: f64,
}

pub struct NeuralNetwork {
    model_type: ModelType,
    feature_dim: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout_rate: f64,
    seq_length: usize,
    lstm_layers: Vec<LstmLayerWeights>,
    rnn_layers: Vec<RnnLayerWeights>,
    mlp_layers: Vec<MlpLayerWeights>,
    cnn_layers: Vec<CnnLayerWeights>,
    final_layer: MlpLayerWeights,
}

impl NeuralNetwork {
    pub fn new(
        model_type: ModelType,
        feature_dim: usize,
        hidden_size: usize,
        num_layers: usize,
        dropout_rate: f64,
        seq_length: usize,
    ) -> Result<Self, TrainingError> {
        println!("🛠️ [Model] Initializing {:?} model with {} features, {} hidden units, {} layers, seq_length {}", 
                 model_type, feature_dim, hidden_size, num_layers, seq_length);
        
        let mut rng = rand::rng();
        let mut lstm_layers = Vec::new();
        let mut rnn_layers = Vec::new();
        let mut mlp_layers = Vec::new();
        let mut cnn_layers = Vec::new();

        let final_input_size = match model_type {
            ModelType::LSTM | ModelType::RNN | ModelType::CNN => hidden_size,
            ModelType::MLP => {
                let layer_sizes = Self::calculate_mlp_sizes(feature_dim * seq_length, hidden_size, num_layers);
                *layer_sizes.last().unwrap()
            }
        };

        match model_type {
            ModelType::LSTM => {
                for i in 0..num_layers {
                    let input_size = if i == 0 { feature_dim } else { hidden_size };
                    lstm_layers.push(LstmLayerWeights {
                        w_ii: Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-0.1..0.1)),
                        w_if: Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-0.1..0.1)),
                        w_ig: Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-0.1..0.1)),
                        w_io: Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-0.1..0.1)),
                        w_hi: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.random_range(-0.1..0.1)),
                        w_hf: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.random_range(-0.1..0.1)),
                        w_hg: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.random_range(-0.1..0.1)),
                        w_ho: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.random_range(-0.1..0.1)),
                        b_i: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
                        b_f: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
                        b_g: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
                        b_o: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
                    });
                }
            },
            ModelType::RNN => {
                for i in 0..num_layers {
                    let input_size = if i == 0 { feature_dim } else { hidden_size };
                    rnn_layers.push(RnnLayerWeights {
                        w_ih: Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-0.1..0.1)),
                        w_hh: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.random_range(-0.1..0.1)),
                        b_h: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
                    });
                }
            },
            ModelType::MLP => {
                let layer_sizes = Self::calculate_mlp_sizes(feature_dim * seq_length, hidden_size, num_layers);
                for i in 0..layer_sizes.len()-1 {
                    mlp_layers.push(MlpLayerWeights {
                        w: Array2::from_shape_fn((layer_sizes[i+1], layer_sizes[i]), |_| rng.random_range(-0.1..0.1)),
                        b: Array1::from_shape_fn(layer_sizes[i+1], |_| rng.random_range(-0.1..0.1)),
                    });
                }
            },
            ModelType::CNN => {
                for i in 0..num_layers {
                    let kernel_size = 3;
                    let in_channels = if i == 0 { feature_dim } else { hidden_size };
                    let out_channels = hidden_size;
                    cnn_layers.push(CnnLayerWeights {
                        kernels: Array3::from_shape_fn((out_channels, in_channels, kernel_size), |_| rng.random_range(-0.1..0.1)),
                        bias: Array1::from_shape_fn(out_channels, |_| rng.random_range(-0.1..0.1)),
                    });
                }
            },
        }

        let final_layer = MlpLayerWeights {
            w: Array2::from_shape_fn((1, final_input_size), |_| rng.random_range(-0.1..0.1)),
            b: Array1::from_shape_fn(1, |_| rng.random_range(-0.1..0.1)),
        };

        Ok(NeuralNetwork {
            model_type,
            feature_dim,
            hidden_size,
            num_layers,
            dropout_rate,
            seq_length,
            lstm_layers,
            rnn_layers,
            mlp_layers,
            cnn_layers,
            final_layer,
        })
    }

    fn calculate_mlp_sizes(input_dim: usize, hidden_size: usize, num_layers: usize) -> Vec<usize> {
        let mut sizes = vec![input_dim];
        for _ in 0..num_layers {
            sizes.push(hidden_size);
        }
        sizes
    }

    pub fn forward(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        if input.nrows() != self.seq_length || input.ncols() != self.feature_dim {
            return Err(TrainingError::ModelConfiguration(
                format!("Input shape mismatch: expected ({}, {}), got ({}, {})",
                        self.seq_length, self.feature_dim, input.nrows(), input.ncols())
            ));
        }

        let mut hidden = Array1::zeros(self.hidden_size);
        match self.model_type {
            ModelType::RNN => {
                for t in 0..input.nrows() {
                    let x = input.slice(ndarray::s![t, ..]).to_owned();
                    let pre_activation = &self.rnn_layers[0].w_ih.dot(&x) + &self.rnn_layers[0].w_hh.dot(&hidden) + &self.rnn_layers[0].b_h;
                    hidden = tanh(&pre_activation);
                }
            },
            ModelType::LSTM => {
                let mut cell = Array1::zeros(self.hidden_size);
                for layer in &self.lstm_layers {
                    for t in 0..input.nrows() {
                        let x = input.slice(ndarray::s![t, ..]).to_owned();
                        let input_gate = sigmoid(&(&layer.w_ii.dot(&x) + &layer.w_hi.dot(&hidden) + &layer.b_i));
                        let forget_gate = sigmoid(&(&layer.w_if.dot(&x) + &layer.w_hf.dot(&hidden) + &layer.b_f));
                        let cell_gate = tanh(&(&layer.w_ig.dot(&x) + &layer.w_hg.dot(&hidden) + &layer.b_g));
                        let output_gate = sigmoid(&(&layer.w_io.dot(&x) + &layer.w_ho.dot(&hidden) + &layer.b_o));
                        cell = &forget_gate * &cell + &input_gate * &cell_gate;
                        hidden = &output_gate * tanh(&cell);
                    }
                }
            },
            ModelType::MLP => {
                let flattened = input.clone().to_shape(input.len()).unwrap();
                let mut current = flattened;
                for layer in &self.mlp_layers {
                    current = relu(&(&layer.w.dot(&current) + &layer.b));
                    if self.dropout_rate > 0.0 {
                        let mut rng = rand::rng();
                        current.mapv_inplace(|x| if rng.random_range(0.0..1.0) < self.dropout_rate { 0.0 } else { x });
                    }
                }
                hidden = current;
            },
            ModelType::CNN => {
                let mut layer_input = input.clone();
                for layer in &self.cnn_layers {
                    let kernel_size = layer.kernels.dim().2;
                    let out_channels = layer.kernels.dim().0;
                    let mut conv_output = Array2::zeros((layer_input.nrows() - kernel_size + 1, out_channels));
                    
                    for t in 0..layer_input.nrows() - kernel_size + 1 {
                        let window = layer_input.slice(ndarray::s![t..t+kernel_size, ..]).to_owned();
                        for c in 0..out_channels {
                            let kernel = layer.kernels.slice(ndarray::s![c, .., ..]).to_owned();
                            let conv = (window.clone() * kernel).sum() + layer.bias[c];
                            conv_output[[t, c]] = relu_scalar(conv);
                        }
                    }
                    
                    // Update layer_input for next layer
                    layer_input = conv_output.clone();
                }
                
                // Global average pooling for CNN output
                if !layer_input.is_empty() {
                    hidden = layer_input.mean_axis(ndarray::Axis(0)).unwrap();
                } else {
                    hidden = Array1::zeros(self.hidden_size);
                }
            },
        }

        let final_output = self.final_layer.w.dot(&hidden) + &self.final_layer.b;
        Ok(final_output[0])
    }

    pub fn train(
        &mut self,
        sequences: &[Array2<f64>],
        targets: &[f64],
        optimizer: &mut AdamOptimizer,
        batch_size: usize,
        l2_weight: f64,
        _clip_norm: f64,
    ) -> Result<f64, TrainingError> {
        if sequences.len() != targets.len() {
            return Err(TrainingError::DataProcessing(
                "Number of sequences must match number of targets".to_string()
            ));
        }

        let mut total_loss = 0.0;
        let n_batches = (sequences.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = std::cmp::min(start + batch_size, sequences.len());
            let batch_seqs = &sequences[start..end];
            let batch_targets = &targets[start..end];

            let mut batch_loss = 0.0;
            let mut predictions = Vec::new();

            // Forward pass
            for seq in batch_seqs {
                let pred = self.forward(seq)?;
                predictions.push(pred);
            }

            batch_loss += crate::neural::utils::mse_loss(&predictions, batch_targets);
            batch_loss += l2_weight * self.l2_regularization();

            // Simplified backward pass
            let mut d_output = Array1::zeros(1);
            for (pred, target) in predictions.iter().zip(batch_targets.iter()) {
                d_output[0] += 2.0 * (pred - target);
            }
            d_output /= batch_seqs.len() as f64;

            // Update final layer weights
            let learning_rate = optimizer.learning_rate;
            
            // Update final layer weights with simple gradient descent
            for i in 0..self.final_layer.w.nrows() {
                for j in 0..self.final_layer.w.ncols() {
                    self.final_layer.w[[i, j]] -= learning_rate * d_output[i] * 0.01; // Simplified gradient
                }
            }
            
            for i in 0..self.final_layer.b.len() {
                self.final_layer.b[i] -= learning_rate * d_output[i];
            }

            total_loss += batch_loss;
        }

        Ok(total_loss / n_batches as f64)
    }

    pub fn validate(&self, sequences: &[Array2<f64>], targets: &[f64]) -> Result<(f64, ValidationMetrics), TrainingError> {
        let mut predictions = Vec::new();
        
        for seq in sequences {
            let pred = self.forward(seq)?;
            predictions.push(pred);
        }

        let loss = crate::neural::utils::mse_loss(&predictions, targets);
        let metrics = crate::neural::metrics::calculate_regression_metrics(&predictions, targets);
        
        Ok((loss, ValidationMetrics {
            rmse: metrics.rmse,
            mae: metrics.mae,
            mape: metrics.mape,
            directional_accuracy: metrics.directional_accuracy,
            r_squared: metrics.r_squared,
        }))
    }

    fn l2_regularization(&self) -> f64 {
        let mut l2_sum = 0.0;

        for layer in &self.rnn_layers {
            l2_sum += layer.w_ih.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_hh.iter().map(|x| x.powi(2)).sum::<f64>();
        }

        for layer in &self.lstm_layers {
            l2_sum += layer.w_ii.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_if.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_ig.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_io.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_hi.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_hf.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_hg.iter().map(|x| x.powi(2)).sum::<f64>();
            l2_sum += layer.w_ho.iter().map(|x| x.powi(2)).sum::<f64>();
        }

        for layer in &self.mlp_layers {
            l2_sum += layer.w.iter().map(|x| x.powi(2)).sum::<f64>();
        }

        for layer in &self.cnn_layers {
            l2_sum += layer.kernels.iter().map(|x| x.powi(2)).sum::<f64>();
        }

        l2_sum += self.final_layer.w.iter().map(|x| x.powi(2)).sum::<f64>();
        l2_sum
    }

    pub fn get_weights(&self) -> ModelWeights {
        ModelWeights {
            asset: String::new(),
            model_type: self.model_type,
            lstm_layers: self.lstm_layers.clone(),
            rnn_layers: self.rnn_layers.clone(),
            mlp_layers: self.mlp_layers.clone(),
            cnn_layers: self.cnn_layers.clone(),
            final_layer: self.final_layer.clone(),
            closing_mean: 0.0,
            closing_std: 0.0,
            seq_length: self.seq_length,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            feature_dim: self.feature_dim,
            epoch: 0,
            timestamp: Utc::now().to_rfc3339(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_creation() {
        let model = NeuralNetwork::new(
            ModelType::RNN,
            5,  // feature_dim
            32, // hidden_size
            1,  // num_layers
            0.1, // dropout_rate
            10, // seq_length
        ).unwrap();

        assert_eq!(model.model_type, ModelType::RNN);
        assert_eq!(model.rnn_layers.len(), 1);
    }

    #[test]
    fn test_forward_rnn() {
        let model = NeuralNetwork::new(
            ModelType::RNN,
            3,  // feature_dim
            16, // hidden_size
            1,  // num_layers
            0.0, // dropout_rate
            5,  // seq_length
        ).unwrap();

        let input = Array2::from_shape_vec((5, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
        ]).unwrap();

        let result = model.forward(&input);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_nan());
    }
}