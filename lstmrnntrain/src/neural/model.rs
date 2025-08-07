// projeto: lstmrnntrain
// file: src/neural/model.rs
// Multi-model neural network implementation with LSTM, RNN, MLP, and CNN




// projeto: lstmrnntrain
// file: src/neural/model.rs
// Multi-model neural network implementation with LSTM, RNN, MLP, and CNN

use chrono::Utc;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::neural::utils::{AdamOptimizer, TrainingError, sigmoid, tanh, relu};
use rayon::prelude::*;

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
    ) -> Result<Self, TrainingError> {
        println!("🛠️ [Model] Initializing {:?} model with {} features, {} hidden units, {} layers", 
                 model_type, feature_dim, hidden_size, num_layers);
        
        let mut rng = rand::thread_rng();
        let mut lstm_layers = Vec::new();
        let mut rnn_layers = Vec::new();
        let mut mlp_layers = Vec::new();
        let mut cnn_layers = Vec::new();

        match model_type {
            ModelType::LSTM => {
                for i in 0..num_layers {
                    let input_size = if i == 0 { feature_dim } else { hidden_size };
                    lstm_layers.push(LstmLayerWeights {
                        w_ii: Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-0.1..0.1)),
                        w_if: Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-0.1..0.1)),
                        w_ig: Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-0.1..0.1)),
                        w_io: Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-0.1..0.1)),
                        w_hi: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-0.1..0.1)),
                        w_hf: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-0.1..0.1)),
                        w_hg: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-0.1..0.1)),
                        w_ho: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-0.1..0.1)),
                        b_i: Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.1..0.1)),
                        b_f: Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.1..0.1)),
                        b_g: Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.1..0.1)),
                        b_o: Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.1..0.1)),
                    });
                }
            },
            ModelType::RNN => {
                for i in 0..num_layers {
                    let input_size = if i == 0 { feature_dim } else { hidden_size };
                    rnn_layers.push(RnnLayerWeights {
                        w_ih: Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-0.1..0.1)),
                        w_hh: Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-0.1..0.1)),
                        b_h: Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.1..0.1)),
                    });
                }
            },
            ModelType::MLP => {
                let layer_sizes = Self::calculate_mlp_sizes(feature_dim, hidden_size, num_layers);
                for i in 0..layer_sizes.len()-1 {
                    mlp_layers.push(MlpLayerWeights {
                        w: Array2::from_shape_fn((layer_sizes[i+1], layer_sizes[i]), |_| rng.gen_range(-0.1..0.1)),
                        b: Array1::from_shape_fn(layer_sizes[i+1], |_| rng.gen_range(-0.1..0.1)),
                    });
                }
            },
            ModelType::CNN => {
                // CNN with 1D convolution for time series
                for i in 0..num_layers {
                    let kernel_size = 3;
                    let in_channels = if i == 0 { feature_dim } else { hidden_size };
                    let out_channels = hidden_size;
                    cnn_layers.push(CnnLayerWeights {
                        kernels: Array3::from_shape_fn((out_channels, in_channels, kernel_size), |_| rng.gen_range(-0.1..0.1)),
                        bias: Array1::from_shape_fn(out_channels, |_| rng.gen_range(-0.1..0.1)),
                    });
                }
            }
        }

        // Final output layer (always maps to 1 output for regression)
        let final_input_size = match model_type {
            ModelType::MLP => mlp_layers.last().map(|l| l.w.nrows()).unwrap_or(hidden_size),
            _ => hidden_size,
        };
        
        let final_layer = MlpLayerWeights {
            w: Array2::from_shape_fn((1, final_input_size), |_| rng.gen_range(-0.1..0.1)),
            b: Array1::from_shape_fn(1, |_| rng.gen_range(-0.1..0.1)),
        };

        let model = NeuralNetwork {
            model_type,
            feature_dim,
            hidden_size,
            num_layers,
            dropout_rate,
            lstm_layers,
            rnn_layers,
            mlp_layers,
            cnn_layers,
            final_layer,
        };

        println!("✅ [Model] {:?} model initialized with {} parameters", 
                 model_type, model.num_parameters());
        Ok(model)
    }

    fn calculate_mlp_sizes(input_dim: usize, hidden_size: usize, num_layers: usize) -> Vec<usize> {
        let mut sizes = vec![input_dim];
        for _ in 0..num_layers {
            sizes.push(hidden_size);
        }
        sizes
    }

    pub fn forward(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        match self.model_type {
            ModelType::LSTM => self.forward_lstm(input),
            ModelType::RNN => self.forward_rnn(input),
            ModelType::MLP => self.forward_mlp(input),
            ModelType::CNN => self.forward_cnn(input),
        }
    }

    fn forward_lstm(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        let mut current_input = input.clone();
        let mut hidden = Array1::zeros(self.hidden_size);
        let mut cell = Array1::zeros(self.hidden_size);

        for layer in &self.lstm_layers {
            let (outputs, final_hidden, final_cell) = self.forward_lstm_layer(
                layer, &current_input, &hidden, &cell
            );
            current_input = outputs;
            hidden = final_hidden;
            cell = final_cell;
        }

        // Get final timestep output
        let final_output = current_input.slice(ndarray::s![current_input.nrows() - 1, ..]);
        let prediction = self.final_layer.w.dot(&final_output) + &self.final_layer.b;
        Ok(prediction[0])
    }

    fn forward_lstm_layer(
        &self,
        layer: &LstmLayerWeights,
        input: &Array2<f64>,
        init_hidden: &Array1<f64>,
        init_cell: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let seq_len = input.nrows();
        let mut outputs = Array2::zeros((seq_len, self.hidden_size));
        let mut hidden = init_hidden.clone();
        let mut cell = init_cell.clone();

        for t in 0..seq_len {
            let x_t = input.slice(ndarray::s![t, ..]);
            
            // Forget gate
            let f_t = sigmoid(&(layer.w_if.dot(&x_t) + layer.w_hf.dot(&hidden) + &layer.b_f));
            
            // Input gate
            let i_t = sigmoid(&(layer.w_ii.dot(&x_t) + layer.w_hi.dot(&hidden) + &layer.b_i));
            
            // Cell candidate
            let g_t = tanh(&(layer.w_ig.dot(&x_t) + layer.w_hg.dot(&hidden) + &layer.b_g));
            
            // Output gate
            let o_t = sigmoid(&(layer.w_io.dot(&x_t) + layer.w_ho.dot(&hidden) + &layer.b_o));
            
            // Update cell state
            cell = &f_t * &cell + &i_t * &g_t;
            
            // Update hidden state
            hidden = &o_t * &tanh(&cell);
            
            outputs.slice_mut(ndarray::s![t, ..]).assign(&hidden);
        }

        (outputs, hidden, cell)
    }

    fn forward_rnn(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        let mut current_input = input.clone();
        
        for layer in &self.rnn_layers {
            current_input = self.forward_rnn_layer(layer, &current_input);
        }

        // Get final timestep output
        let final_output = current_input.slice(ndarray::s![current_input.nrows() - 1, ..]);
        let prediction = self.final_layer.w.dot(&final_output) + &self.final_layer.b;
        Ok(prediction[0])
    }

    fn forward_rnn_layer(&self, layer: &RnnLayerWeights, input: &Array2<f64>) -> Array2<f64> {
        let seq_len = input.nrows();
        let mut outputs = Array2::zeros((seq_len, self.hidden_size));
        let mut hidden = Array1::zeros(self.hidden_size);

        for t in 0..seq_len {
            let x_t = input.slice(ndarray::s![t, ..]);
            hidden = tanh(&(layer.w_ih.dot(&x_t) + layer.w_hh.dot(&hidden) + &layer.b_h));
            outputs.slice_mut(ndarray::s![t, ..]).assign(&hidden);
        }

        outputs
    }

    fn forward_mlp(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        // For MLP, flatten the input sequence
        let flattened = input.clone().into_shape(input.len()).unwrap();
        let mut current = flattened;

        for layer in &self.mlp_layers {
            current = relu(&(layer.w.dot(&current) + &layer.b));
        }

        let prediction = self.final_layer.w.dot(&current) + &self.final_layer.b;
        Ok(prediction[0])
    }

    fn forward_cnn(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        let mut current = input.clone();

        for layer in &self.cnn_layers {
            current = self.forward_cnn_layer(layer, &current);
        }

        // Global average pooling
        let pooled = current.mean_axis(Axis(0)).unwrap();
        let prediction = self.final_layer.w.dot(&pooled) + &self.final_layer.b;
        Ok(prediction[0])
    }

    fn forward_cnn_layer(&self, layer: &CnnLayerWeights, input: &Array2<f64>) -> Array2<f64> {
        let (seq_len, in_channels) = input.dim();
        let out_channels = layer.kernels.dim().0;
        let kernel_size = layer.kernels.dim().2;
        let output_len = seq_len - kernel_size + 1;
        
        let mut output = Array2::zeros((output_len, out_channels));

        for out_ch in 0..out_channels {
            for t in 0..output_len {
                let mut sum = layer.bias[out_ch];
                for in_ch in 0..in_channels {
                    for k in 0..kernel_size {
                        sum += input[[t + k, in_ch]] * layer.kernels[[out_ch, in_ch, k]];
                    }
                }
                output[[t, out_ch]] = relu_scalar(sum);
            }
        }

        output
    }

    pub fn train_step(
        &mut self,
        sequences: &[Array2<f64>],
        targets: &[f64],
        optimizer: &mut AdamOptimizer,
        batch_size: usize,
        l2_weight: f64,
        clip_norm: f64,
    ) -> Result<f64, TrainingError> {
        let mut total_loss = 0.0;
        let num_batches = (sequences.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = ((batch_idx + 1) * batch_size).min(sequences.len());
            let batch_sequences = &sequences[start..end];
            let batch_targets = &targets[start..end];

            let (loss, gradients) = self.compute_gradients(batch_sequences, batch_targets, l2_weight)?;
            
            // Apply gradient clipping
            let clipped_gradients = self.clip_gradients(gradients, clip_norm);
            
            // Update weights
            self.apply_gradients(clipped_gradients, optimizer)?;
            
            total_loss += loss;
        }

        Ok(total_loss / num_batches as f64)
    }

    pub fn validate(&self, sequences: &[Array2<f64>], targets: &[f64]) -> Result<(f64, ValidationMetrics), TrainingError> {
        let mut predictions = Vec::new();
        let mut total_loss = 0.0;

        for (seq, &target) in sequences.iter().zip(targets.iter()) {
            let pred = self.forward(seq)?;
            predictions.push(pred);
            total_loss += (pred - target).powi(2);
        }

        let mse = total_loss / sequences.len() as f64;
        let metrics = self.calculate_metrics(&predictions, targets);

        Ok((mse, metrics))
    }

    fn calculate_metrics(&self, predictions: &[f64], targets: &[f64]) -> ValidationMetrics {
        let n = predictions.len() as f64;
        
        // RMSE
        let mse = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / n;
        let rmse = mse.sqrt();

        // MAE
        let mae = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / n;

        // MAPE
        let mape = predictions.iter().zip(targets.iter())
            .map(|(p, t)| ((p - t) / t).abs())
            .sum::<f64>() / n * 100.0;

        // Directional Accuracy
        let mut correct_direction = 0;
        for i in 1..predictions.len() {
            let pred_change = predictions[i] - predictions[i-1];
            let actual_change = targets[i] - targets[i-1];
            if pred_change.signum() == actual_change.signum() {
                correct_direction += 1;
            }
        }
        let directional_accuracy = if predictions.len() > 1 {
            correct_direction as f64 / (predictions.len() - 1) as f64
        } else {
            0.0
        };

        // R-squared
        let target_mean = targets.iter().sum::<f64>() / n;
        let ss_res = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum::<f64>();
        let ss_tot = targets.iter()
            .map(|t| (t - target_mean).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - (ss_res / ss_tot);

        ValidationMetrics {
            rmse,
            mae,
            mape,
            directional_accuracy,
            r_squared,
        }
    }

    fn compute_gradients(
        &self,
        sequences: &[Array2<f64>],
        targets: &[f64],
        l2_weight: f64,
    ) -> Result<(f64, ModelWeights), TrainingError> {
        // Simplified gradient computation - in practice would use automatic differentiation
        let mut total_loss = 0.0;
        
        for (seq, &target) in sequences.iter().zip(targets.iter()) {
            let pred = self.forward(seq)?;
            total_loss += (pred - target).powi(2);
        }

        // Add L2 regularization
        let l2_loss = self.compute_l2_loss() * l2_weight;
        total_loss += l2_loss;

        // Create dummy gradients (in practice would compute actual gradients)
        let gradients = self.create_zero_gradients();

        Ok((total_loss / sequences.len() as f64, gradients))
    }

    fn compute_l2_loss(&self) -> f64 {
        let mut l2_loss = 0.0;
        
        for layer in &self.lstm_layers {
            l2_loss += layer.w_ii.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_if.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_ig.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_io.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_hi.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_hf.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_hg.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_ho.mapv(|x| x.powi(2)).sum();
        }

        for layer in &self.rnn_layers {
            l2_loss += layer.w_ih.mapv(|x| x.powi(2)).sum();
            l2_loss += layer.w_hh.mapv(|x| x.powi(2)).sum();
        }

        for layer in &self.mlp_layers {
            l2_loss += layer.w.mapv(|x| x.powi(2)).sum();
        }

        for layer in &self.cnn_layers {
            l2_loss += layer.kernels.mapv(|x| x.powi(2)).sum();
        }

        l2_loss += self.final_layer.w.mapv(|x| x.powi(2)).sum();

        l2_loss
    }

    fn create_zero_gradients(&self) -> ModelWeights {
        ModelWeights {
            asset: String::new(),
            model_type: self.model_type,
            lstm_layers: self.lstm_layers.iter().map(|layer| LstmLayerWeights {
                w_ii: Array2::zeros(layer.w_ii.raw_dim()),
                w_if: Array2::zeros(layer.w_if.raw_dim()),
                w_ig: Array2::zeros(layer.w_ig.raw_dim()),
                w_io: Array2::zeros(layer.w_io.raw_dim()),
                w_hi: Array2::zeros(layer.w_hi.raw_dim()),
                w_hf: Array2::zeros(layer.w_hf.raw_dim()),
                w_hg: Array2::zeros(layer.w_hg.raw_dim()),
                w_ho: Array2::zeros(layer.w_ho.raw_dim()),
                b_i: Array1::zeros(layer.b_i.raw_dim()),
                b_f: Array1::zeros(layer.b_f.raw_dim()),
                b_g: Array1::zeros(layer.b_g.raw_dim()),
                b_o: Array1::zeros(layer.b_o.raw_dim()),
            }).collect(),
            rnn_layers: self.rnn_layers.iter().map(|layer| RnnLayerWeights {
                w_ih: Array2::zeros(layer.w_ih.raw_dim()),
                w_hh: Array2::zeros(layer.w_hh.raw_dim()),
                b_h: Array1::zeros(layer.b_h.raw_dim()),
            }).collect(),
            mlp_layers: self.mlp_layers.iter().map(|layer| MlpLayerWeights {
                w: Array2::zeros(layer.w.raw_dim()),
                b: Array1::zeros(layer.b.raw_dim()),
            }).collect(),
            cnn_layers: self.cnn_layers.iter().map(|layer| CnnLayerWeights {
                kernels: Array3::zeros(layer.kernels.raw_dim()),
                bias: Array1::zeros(layer.bias.raw_dim()),
            }).collect(),
            final_layer: MlpLayerWeights {
                w: Array2::zeros(self.final_layer.w.raw_dim()),
                b: Array1::zeros(self.final_layer.b.raw_dim()),
            },
            closing_mean: 0.0,
            closing_std: 0.0,
            seq_length: 0,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            feature_dim: self.feature_dim,
            epoch: 0,
            timestamp: String::new(),
        }
    }

    fn clip_gradients(&self, mut gradients: ModelWeights, clip_norm: f64) -> ModelWeights {
        // Simplified gradient clipping - compute total norm and scale if necessary
        let mut total_norm = 0.0;

        for layer in &gradients.lstm_layers {
            total_norm += layer.w_ii.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_if.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_ig.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_io.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_hi.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_hf.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_hg.mapv(|x| x.powi(2)).sum();
            total_norm += layer.w_ho.mapv(|x| x.powi(2)).sum();
        }

        total_norm = total_norm.sqrt();

        if total_norm > clip_norm {
            let scale = clip_norm / total_norm;
            // Scale all gradients by the clipping factor
            for layer in &mut gradients.lstm_layers {
                layer.w_ii *= scale;
                layer.w_if *= scale;
                layer.w_ig *= scale;
                layer.w_io *= scale;
                layer.w_hi *= scale;
                layer.w_hf *= scale;
                layer.w_hg *= scale;
                layer.w_ho *= scale;
                layer.b_i *= scale;
                layer.b_f *= scale;
                layer.b_g *= scale;
                layer.b_o *= scale;
            }
        }

        gradients
    }

    fn apply_gradients(&mut self, _gradients: ModelWeights, _optimizer: &mut AdamOptimizer) -> Result<(), TrainingError> {
        // In practice, would apply gradients using the optimizer
        // This is a simplified stub
        Ok(())
    }

    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        for layer in &self.lstm_layers {
            count += layer.w_ii.len() + layer.w_if.len() + layer.w_ig.len() + layer.w_io.len();
            count += layer.w_hi.len() + layer.w_hf.len() + layer.w_hg.len() + layer.w_ho.len();
            count += layer.b_i.len() + layer.b_f.len() + layer.b_g.len() + layer.b_o.len();
        }

        for layer in &self.rnn_layers {
            count += layer.w_ih.len() + layer.w_hh.len() + layer.b_h.len();
        }

        for layer in &self.mlp_layers {
            count += layer.w.len() + layer.b.len();
        }

        for layer in &self.cnn_layers {
            count += layer.kernels.len() + layer.bias.len();
        }

        count += self.final_layer.w.len() + self.final_layer.b.len();
        count
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
            seq_length: 0,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            feature_dim: self.feature_dim,
            epoch: 0,
            timestamp: Utc::now().to_rfc3339(),
        }
    }

    pub fn load_weights(&mut self, weights: &ModelWeights) -> Result<(), TrainingError> {
        if weights.model_type != self.model_type {
            return Err(TrainingError::ModelConfiguration(
                format!("Model type mismatch: expected {:?}, got {:?}", 
                        self.model_type, weights.model_type)
            ));
        }

        self.lstm_layers = weights.lstm_layers.clone();
        self.rnn_layers = weights.rnn_layers.clone();
        self.mlp_layers = weights.mlp_layers.clone();
        self.cnn_layers = weights.cnn_layers.clone();
        self.final_layer = weights.final_layer.clone();

        Ok(())
    }

    pub fn predict(&self, sequence: &Array2<f64>) -> Result<f64, TrainingError> {
        self.forward(sequence)
    }

    pub fn predict_batch(&self, sequences: &[Array2<f64>]) -> Result<Vec<f64>, TrainingError> {
        sequences.par_iter()
            .map(|seq| self.forward(seq))
            .collect()
    }
}

// Helper functions
fn relu_scalar(x: f64) -> f64 {
    x.max(0.0)
}

fn sigmoid_scalar(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_scalar(x: f64) -> f64 {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_lstm_creation() {
        let model = NeuralNetwork::new(
            ModelType::LSTM,
            5,  // feature_dim
            64, // hidden_size
            2,  // num_layers
            0.2 // dropout_rate
        ).unwrap();

        assert_eq!(model.model_type, ModelType::LSTM);
        assert_eq!(model.lstm_layers.len(), 2);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_rnn_creation() {
        let model = NeuralNetwork::new(
            ModelType::RNN,
            5,  // feature_dim
            32, // hidden_size
            1,  // num_layers
            0.1 // dropout_rate
        ).unwrap();

        assert_eq!(model.model_type, ModelType::RNN);
        assert_eq!(model.rnn_layers.len(), 1);
    }

    #[test]
    fn test_mlp_creation() {
        let model = NeuralNetwork::new(
            ModelType::MLP,
            10, // feature_dim
            64, // hidden_size
            3,  // num_layers
            0.2 // dropout_rate
        ).unwrap();

        assert_eq!(model.model_type, ModelType::MLP);
        assert_eq!(model.mlp_layers.len(), 3);
    }

    #[test]
    fn test_cnn_creation() {
        let model = NeuralNetwork::new(
            ModelType::CNN,
            8,  // feature_dim
            32, // hidden_size
            2,  // num_layers
            0.1 // dropout_rate
        ).unwrap();

        assert_eq!(model.model_type, ModelType::CNN);
        assert_eq!(model.cnn_layers.len(), 2);
    }

    #[test]
    fn test_forward_lstm() {
        let model = NeuralNetwork::new(
            ModelType::LSTM,
            3,  // feature_dim
            16, // hidden_size
            1,  // num_layers
            0.0 // dropout_rate
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
    }

    #[test]
    fn test_forward_rnn() {
        let model = NeuralNetwork::new(
            ModelType::RNN,
            3,  // feature_dim
            16, // hidden_size
            1,  // num_layers
            0.0 // dropout_rate
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
    }

    #[test]
    fn test_validation_metrics() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.1, 1.9, 3.1, 3.8, 5.2];

        let model = NeuralNetwork::new(
            ModelType::LSTM,
            1, 8, 1, 0.0
        ).unwrap();

        let metrics = model.calculate_metrics(&predictions, &targets);
        
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.mape > 0.0);
        assert!(metrics.directional_accuracy >= 0.0 && metrics.directional_accuracy <= 1.0);
    }

    #[test]
    fn test_weights_serialization() {
        let model = NeuralNetwork::new(
            ModelType::LSTM,
            5, 32, 2, 0.1
        ).unwrap();

        let weights = model.get_weights();
        assert_eq!(weights.model_type, ModelType::LSTM);
        assert_eq!(weights.hidden_size, 32);
        assert_eq!(weights.num_layers, 2);
        assert_eq!(weights.feature_dim, 5);
    }
}