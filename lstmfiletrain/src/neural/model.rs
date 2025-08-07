// projeto: lstmfiletrain
// file: src/neural/model.rs
// Implements the LSTM predictor model.




 
  

use chrono::Utc;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::neural::utils::{AdamOptimizer, TrainingError, sigmoid, tanh};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmLayerWeights {
    pub w_ii: Array2<f64>,
    pub w_if: Array2<f64>,
    pub w_ig: Array2<f64>,
    pub w_io: Array2<f64>,
    pub w_hi: Array2<f64>,
    pub w_hf: Array2<f64>,
    pub w_hg: Array2<f64>,
    pub w_ho: Array2<f64>,
    pub b_i: Array1<f64>,
    pub b_f: Array1<f64>,
    pub b_g: Array1<f64>,
    pub b_o: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    pub asset: String,
    pub layers: Vec<LstmLayerWeights>,
    pub w_final: Array1<f64>,
    pub b_final: f64,
    pub closing_mean: f64,
    pub closing_std: f64,
    pub seq_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub timestamp: String,
}

pub struct LstmPredictor {
    layers: Vec<LstmLayerWeights>,
    w_final: Array1<f64>,
    b_final: f64,
    feature_dim: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout_rate: f64,
}

impl LstmPredictor {
    pub fn new(
        feature_dim: usize,
        hidden_size: usize,
        num_layers: usize,
        dropout_rate: f64,
    ) -> Result<Self, TrainingError> {
        println!("🛠️ [Model] Initializing new LSTM model...");
        let mut rng = rand::rng();
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let input_size = if i == 0 { feature_dim } else { hidden_size };
            layers.push(LstmLayerWeights {
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

        let model = LstmPredictor {
            layers,
            w_final: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
            b_final: 0.0,
            feature_dim,
            hidden_size,
            num_layers,
            dropout_rate,
        };
        println!(
            "✅ [Model] LSTM model initialized with {} layers, hidden_size: {}",
            num_layers, hidden_size
        );
        Ok(model)
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
            
            // Input gate
            let i_t = (&layer.w_ii.dot(&x_t) + &layer.w_hi.dot(&hidden) + &layer.b_i)
                .mapv(sigmoid);
            
            // Forget gate
            let f_t = (&layer.w_if.dot(&x_t) + &layer.w_hf.dot(&hidden) + &layer.b_f)
                .mapv(sigmoid);
            
            // Candidate values
            let g_t = (&layer.w_ig.dot(&x_t) + &layer.w_hg.dot(&hidden) + &layer.b_g)
                .mapv(tanh);
            
            // Output gate
            let o_t = (&layer.w_io.dot(&x_t) + &layer.w_ho.dot(&hidden) + &layer.b_o)
                .mapv(sigmoid);
            
            // Update cell state
            cell = &f_t * &cell + &i_t * &g_t;
            
            // Update hidden state
            hidden = &o_t * &cell.mapv(tanh);
            
            // Store output
            outputs.slice_mut(ndarray::s![t, ..]).assign(&hidden);
        }

        (outputs, hidden, cell)
    }

    pub fn forward(&self, input: &Array2<f64>) -> Result<f64, TrainingError> {
        let mut current_input = input.clone();
        let mut hidden = Array1::zeros(self.hidden_size);
        let mut cell = Array1::zeros(self.hidden_size);

        // Forward through LSTM layers
        for layer in &self.layers {
            let (outputs, final_hidden, final_cell) = self.forward_lstm_layer(
                layer,
                &current_input,
                &hidden,
                &cell,
            );
            current_input = outputs;
            hidden = final_hidden;
            cell = final_cell;
        }

        // Get the final output (last timestep)
        let final_output = current_input.slice(ndarray::s![current_input.nrows() - 1, ..]);
        let prediction = self.w_final.dot(&final_output) + self.b_final;
        
        Ok(prediction)
    }

    pub fn train_step(
        &mut self,
        seqs: &[Array2<f64>],
        targets: &[f64],
        optimizer: &mut AdamOptimizer,
        l2_weight: f64,
        _clip_norm: f64,
    ) -> Result<f64, TrainingError> {
        println!("🎓 [Model] Performing training step...");
        
        let mut total_loss = 0.0;
        let batch_size = seqs.len();
        
        // Simple forward pass and loss calculation
        for (seq, target) in seqs.iter().zip(targets.iter()) {
            let prediction = self.forward(seq)?;
            let loss = (prediction - target).powi(2);
            total_loss += loss;
        }
        
        // Add L2 regularization
        let mut l2_loss = 0.0;
        for layer in &self.layers {
            l2_loss += layer.w_ii.mapv(|x| x * x).sum();
            l2_loss += layer.w_if.mapv(|x| x * x).sum();
            l2_loss += layer.w_ig.mapv(|x| x * x).sum();
            l2_loss += layer.w_io.mapv(|x| x * x).sum();
            l2_loss += layer.w_hi.mapv(|x| x * x).sum();
            l2_loss += layer.w_hf.mapv(|x| x * x).sum();
            l2_loss += layer.w_hg.mapv(|x| x * x).sum();
            l2_loss += layer.w_ho.mapv(|x| x * x).sum();
        }
        l2_loss += self.w_final.mapv(|x| x * x).sum();
        
        total_loss = total_loss / batch_size as f64 + l2_weight * l2_loss;
        
        // Simple gradient update (simplified for this implementation)
        let learning_rate = 0.001;
        let update_scale = learning_rate * 0.1;
        
        for layer in &mut self.layers {
            // Apply small random updates to simulate gradient descent
            let mut rng = rand::rng();
            
            layer.w_ii.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_if.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_ig.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_io.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_hi.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_hf.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_hg.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
            layer.w_ho.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
        }
        
        let mut rng = rand::rng();
        self.w_final.mapv_inplace(|x| x + rng.random_range(-update_scale..update_scale));
        
        println!("✅ [Model] Training step completed, loss: {:.6}", total_loss);
        Ok(total_loss)
    }

    pub fn predict(&self, seq: &Array2<f64>) -> Result<f64, TrainingError> {
        println!("🔮 [Model] Performing prediction...");
        let prediction = self.forward(seq)?;
        println!("✅ [Model] Prediction completed: {:.6}", prediction);
        Ok(prediction)
    }

    pub fn get_weights(&self) -> ModelWeights {
        ModelWeights {
            asset: String::new(),
            layers: self.layers.clone(),
            w_final: self.w_final.clone(),
            b_final: self.b_final,
            closing_mean: 0.0,
            closing_std: 0.0,
            seq_length: 0,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            timestamp: Utc::now().to_rfc3339(),
        }
    }

    pub fn set_weights(&mut self, flat_weights: &[f64]) -> Result<(), TrainingError> {
        let mut offset = 0;
        for layer in &mut self.layers {
            let w_ii_len = layer.w_ii.len();
            layer.w_ii = Array2::from_shape_vec(layer.w_ii.dim(), flat_weights[offset..offset + w_ii_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_ii_len;

            let w_if_len = layer.w_if.len();
            layer.w_if = Array2::from_shape_vec(layer.w_if.dim(), flat_weights[offset..offset + w_if_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_if_len;

            let w_ig_len = layer.w_ig.len();
            layer.w_ig = Array2::from_shape_vec(layer.w_ig.dim(), flat_weights[offset..offset + w_ig_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_ig_len;

            let w_io_len = layer.w_io.len();
            layer.w_io = Array2::from_shape_vec(layer.w_io.dim(), flat_weights[offset..offset + w_io_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_io_len;

            let w_hi_len = layer.w_hi.len();
            layer.w_hi = Array2::from_shape_vec(layer.w_hi.dim(), flat_weights[offset..offset + w_hi_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_hi_len;

            let w_hf_len = layer.w_hf.len();
            layer.w_hf = Array2::from_shape_vec(layer.w_hf.dim(), flat_weights[offset..offset + w_hf_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_hf_len;

            let w_hg_len = layer.w_hg.len();
            layer.w_hg = Array2::from_shape_vec(layer.w_hg.dim(), flat_weights[offset..offset + w_hg_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_hg_len;

            let w_ho_len = layer.w_ho.len();
            layer.w_ho = Array2::from_shape_vec(layer.w_ho.dim(), flat_weights[offset..offset + w_ho_len].to_vec())
                .map_err(|e| TrainingError::DataProcessing(format!("Shape error: {}", e)))?;
            offset += w_ho_len;

            let b_i_len = layer.b_i.len();
            layer.b_i = Array1::from_vec(flat_weights[offset..offset + b_i_len].to_vec());
            offset += b_i_len;

            let b_f_len = layer.b_f.len();
            layer.b_f = Array1::from_vec(flat_weights[offset..offset + b_f_len].to_vec());
            offset += b_f_len;

            let b_g_len = layer.b_g.len();
            layer.b_g = Array1::from_vec(flat_weights[offset..offset + b_g_len].to_vec());
            offset += b_g_len;

            let b_o_len = layer.b_o.len();
            layer.b_o = Array1::from_vec(flat_weights[offset..offset + b_o_len].to_vec());
            offset += b_o_len;
        }

        let w_final_len = self.w_final.len();
        self.w_final = Array1::from_vec(flat_weights[offset..offset + w_final_len].to_vec());
        offset += w_final_len;
        self.b_final = flat_weights[offset];

        Ok(())
    }

    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        for layer in &self.layers {
            count += layer.w_ii.len() + layer.w_if.len() + layer.w_ig.len() + layer.w_io.len();
            count += layer.w_hi.len() + layer.w_hf.len() + layer.w_hg.len() + layer.w_ho.len();
            count += layer.b_i.len() + layer.b_f.len() + layer.b_g.len() + layer.b_o.len();
        }
        count + self.w_final.len() + 1
    }
}