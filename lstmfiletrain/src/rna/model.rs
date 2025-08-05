// projeto: lstmfiletrain
// file: src/rna/model.rs


 

use ndarray::{Array1, Array2};
use rand::{rngs::ThreadRng, rng, Rng};
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};
use chrono::Utc;
use crate::rna::metrics::TrainingMetrics;
use crate::LSTMError;
use crate::Cli;

pub struct LSTMCell {
    hidden_size: usize,
    w_input: Array2<f32>,
    u_input: Array2<f32>,
    b_input: Array1<f32>,
    w_forget: Array2<f32>,
    u_forget: Array2<f32>,
    b_forget: Array1<f32>,
    w_output: Array2<f32>,
    u_output: Array2<f32>,
    b_output: Array1<f32>,
    w_cell: Array2<f32>,
    u_cell: Array2<f32>,
    b_cell: Array1<f32>,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize, rng: &mut ThreadRng) -> Self {
        let xavier_input = (2.0 / (input_size as f32)).sqrt();
        let xavier_hidden = (2.0 / (hidden_size as f32)).sqrt();
        let normal_input = Normal::new(0.0, xavier_input).unwrap();
        let normal_hidden = Normal::new(0.0, xavier_hidden).unwrap();

        Self {
            hidden_size,
            w_input: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
            u_input: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
            b_input: Array1::zeros(hidden_size),
            w_forget: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
            u_forget: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
            b_forget: Array1::ones(hidden_size),
            w_output: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
            u_output: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
            b_output: Array1::zeros(hidden_size),
            w_cell: Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(rng)),
            u_cell: Array2::from_shape_fn((hidden_size, hidden_size), |_| normal_hidden.sample(rng)),
            b_cell: Array1::zeros(hidden_size),
        }
    }

    pub fn forward(&self, input: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let i_t = (self.w_input.dot(input) + self.u_input.dot(h_prev) + &self.b_input).mapv(Self::sigmoid);
        let f_t = (self.w_forget.dot(input) + self.u_forget.dot(h_prev) + &self.b_forget).mapv(Self::sigmoid);
        let o_t = (self.w_output.dot(input) + self.u_output.dot(h_prev) + &self.b_output).mapv(Self::sigmoid);
        let g_t = (self.w_cell.dot(input) + self.u_cell.dot(h_prev) + &self.b_cell).mapv(Self::tanh);

        let c_t = &f_t * c_prev + &i_t * &g_t;
        let h_t = &o_t * &c_t.mapv(Self::tanh);

        (h_t, c_t)
    }

    fn sigmoid(x: f32) -> f32 {
        if x > 500.0 { 1.0 } else if x < -500.0 { 0.0 } else { 1.0 / (1.0 + (-x).exp()) }
    }

    fn tanh(x: f32) -> f32 {
        if x > 20.0 { 1.0 } else if x < -20.0 { -1.0 } else { x.tanh() }
    }

    pub fn to_weights(&self) -> LSTMLayerWeights {
        LSTMLayerWeights {
            w_input: self.w_input.clone(),
            u_input: self.u_input.clone(),
            b_input: self.b_input.clone(),
            w_forget: self.w_forget.clone(),
            u_forget: self.u_forget.clone(),
            b_forget: self.b_forget.clone(),
            w_output: self.w_output.clone(),
            u_output: self.u_output.clone(),
            b_output: self.b_output.clone(),
            w_cell: self.w_cell.clone(),
            u_cell: self.u_cell.clone(),
            b_cell: self.b_cell.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LSTMLayerWeights {
    pub w_input: Array2<f32>,
    pub u_input: Array2<f32>,
    pub b_input: Array1<f32>,
    pub w_forget: Array2<f32>,
    pub u_forget: Array2<f32>,
    pub b_forget: Array1<f32>,
    pub w_output: Array2<f32>,
    pub u_output: Array2<f32>,
    pub b_output: Array1<f32>,
    pub w_cell: Array2<f32>,
    pub u_cell: Array2<f32>,
    pub b_cell: Array1<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelWeights {
    pub asset: String,
    pub layers: Vec<LSTMLayerWeights>,
    pub w_final: Array1<f32>,
    pub b_final: f32,
    pub closing_mean: f32,
    pub closing_std: f32,
    pub opening_mean: f32,
    pub opening_std: f32,
    pub seq_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub timestamp: String,
    pub metrics: TrainingMetrics,
}

pub struct MultiLayerLSTM {
    layers: Vec<LSTMCell>,
    w_final: Array1<f32>,
    b_final: f32,
    dropout_rate: f32,
}

impl MultiLayerLSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, dropout_rate: f32, rng: &mut ThreadRng) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            layers.push(LSTMCell::new(layer_input_size, hidden_size, rng));
        }

        let xavier_final = (2.0 / (hidden_size as f32)).sqrt();
        let normal_final = Normal::new(0.0, xavier_final).unwrap();
        let w_final = Array1::from_shape_fn(hidden_size, |_| normal_final.sample(rng));

        Self { layers, w_final, b_final: 0.0, dropout_rate }
    }

    #[allow(dead_code)]
    pub fn from_weights(layers_weights: Vec<LSTMLayerWeights>, w_final: Array1<f32>, b_final: f32) -> Result<Self, LSTMError> {
        let mut layers = Vec::new();
        for weights in layers_weights {
            layers.push(LSTMCell {
                hidden_size: weights.w_input.shape()[0],
                w_input: weights.w_input,
                u_input: weights.u_input,
                b_input: weights.b_input,
                w_forget: weights.w_forget,
                u_forget: weights.u_forget,
                b_forget: weights.b_forget,
                w_output: weights.w_output,
                u_output: weights.u_output,
                b_output: weights.b_output,
                w_cell: weights.w_cell,
                u_cell: weights.u_cell,
                b_cell: weights.b_cell,
            });
        }
        Ok(Self {
            layers,
            w_final,
            b_final,
            dropout_rate: 0.0,
        })
    }

    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(&self, sequence: &[f32], training: bool) -> f32 {
        let hidden_size = self.layers[0].hidden_size;
        let num_layers = self.layers.len();
        let mut h_states = vec![Array1::zeros(hidden_size); num_layers];
        let mut c_states = vec![Array1::zeros(hidden_size); num_layers];
        let mut rng = rng();

        let mut current_input = Array1::from_vec(vec![
            sequence[0], sequence[1], sequence[2], sequence[3], sequence[4],
        ]);

        for i in (0..sequence.len()).step_by(5) {
            if i > 0 {
                current_input = Array1::from_vec(vec![
                    sequence[i], sequence[i + 1], sequence[i + 2], sequence[i + 3], sequence[i + 4],
                ]);
            }
            let mut next_h = Array1::zeros(hidden_size);
            for (j, layer) in self.layers.iter().enumerate() {
                let (mut h_new, c_new) = if j == 0 {
                    layer.forward(&current_input, &h_states[j], &c_states[j])
                } else {
                    layer.forward(&h_states[j - 1], &h_states[j], &c_states[j])
                };
                if training && j < num_layers - 1 && self.dropout_rate > 0.0 {
                    let dropout_mask = Array1::from_shape_fn(hidden_size, |_| {
                        if rng.random::<f32>() < self.dropout_rate { 0.0 } else { 1.0 }
                    });
                    h_new = h_new * &dropout_mask;
                }
                h_states[j] = h_new.clone();
                c_states[j] = c_new;
                next_h = h_new;
            }
            h_states[0] = next_h.clone();
        }

        let output = self.w_final.dot(&h_states[num_layers - 1]) + self.b_final;
        output.max(0.0)
    }

    pub fn train_step(&mut self, sequences: &[Vec<f32>], targets: &[f32], learning_rate: f32) -> f32 {
        let mut total_loss = 0.0;
        let batch_size = sequences.len().max(1) as f32;

        for (seq, &target) in sequences.iter().zip(targets.iter()) {
            let prediction = self.forward(seq, true);
            let loss = (prediction - target).powi(2);
            total_loss += loss;

            let error = 2.0 * (prediction - target) / batch_size;
            let lr_scaled = learning_rate * error;

            let hidden_size = self.layers[0].hidden_size;
            let mut h_states = vec![Array1::zeros(hidden_size); self.layers.len()];
            let mut c_states = vec![Array1::zeros(hidden_size); self.layers.len()];
            let mut layer_inputs = Vec::new();

            for i in (0..seq.len()).step_by(5) {
                let input = Array1::from_vec(vec![seq[i], seq[i + 1], seq[i + 2], seq[i + 3], seq[i + 4]]);
                layer_inputs.push(input.clone());
                for (j, layer) in self.layers.iter().enumerate() {
                    let (h_new, c_new) = if j == 0 {
                        layer.forward(&input, &h_states[j], &c_states[j])
                    } else {
                        layer.forward(&h_states[j - 1], &h_states[j], &c_states[j])
                    };
                    h_states[j] = h_new.clone();
                    c_states[j] = c_new;
                }
            }

            let final_hidden = &h_states[self.layers.len() - 1];
            self.w_final = &self.w_final - &(final_hidden * lr_scaled);
            self.b_final -= lr_scaled;

            for layer in &mut self.layers {
                for i in 0..layer.hidden_size {
                    let grad = lr_scaled * 0.01;
                    layer.b_input[i] -= grad.clamp(-0.1, 0.1);
                    layer.b_output[i] -= grad.clamp(-0.1, 0.1);
                    layer.b_cell[i] -= grad.clamp(-0.1, 0.1);
                    layer.b_forget[i] = (layer.b_forget[i] - grad * 0.05).max(0.1);
                }
            }
        }

        total_loss / batch_size
    }

    pub fn to_weights(&self, cli: &Cli, closing_mean: f32, closing_std: f32, opening_mean: f32, opening_std: f32, metrics: TrainingMetrics) -> ModelWeights {
        ModelWeights {
            asset: metrics.asset.clone(), // Use the asset from metrics
            layers: self.layers.iter().map(|layer| layer.to_weights()).collect(),
            w_final: self.w_final.clone(),
            b_final: self.b_final,
            closing_mean,
            closing_std,
            opening_mean,
            opening_std,
            seq_length: cli.seq_length,
            hidden_size: cli.hidden_size,
            num_layers: cli.num_layers,
            timestamp: Utc::now().to_rfc3339(),
            metrics,
        }
    }
}
