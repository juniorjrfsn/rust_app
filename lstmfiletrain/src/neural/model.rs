// projeto: lstmfiletrain
// file: src/neural/model.rs






use ndarray::{Array1, Array2, s, Axis};
use rand::rng;
use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::neural::utils::{sigmoid, tanh, TrainingError, AdamOptimizer, apply_l2_regularization, clip_gradients_by_norm};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmLayerWeights {
    pub w_ii: Array2<f32>, // Input gate weights
    pub w_if: Array2<f32>, // Forget gate weights
    pub w_ig: Array2<f32>, // Cell gate weights
    pub w_io: Array2<f32>, // Output gate weights
    pub w_hi: Array2<f32>, // Hidden to input gate
    pub w_hf: Array2<f32>, // Hidden to forget gate
    pub w_hg: Array2<f32>, // Hidden to cell gate
    pub w_ho: Array2<f32>, // Hidden to output gate
    pub b_i: Array1<f32>,  // Input gate bias
    pub b_f: Array1<f32>,  // Forget gate bias
    pub b_g: Array1<f32>,  // Cell gate bias
    pub b_o: Array1<f32>,  // Output gate bias
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    pub asset: String,
    pub layers: Vec<LstmLayerWeights>,
    pub w_final: Array1<f32>,
    pub b_final: f32,
    pub closing_mean: f32,
    pub closing_std: f32,
    pub seq_length: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub timestamp: String,
}

pub struct LstmPredictor {
    layers: Vec<LstmLayerWeights>,
    w_final: Array1<f32>,
    b_final: f32,
    feature_dim: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout_rate: f32,
}

impl LstmPredictor {
    pub fn new(feature_dim: usize, hidden_size: usize, num_layers: usize, dropout_rate: f32) -> Result<Self, TrainingError> {
        let mut rng = rng();
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
        Ok(LstmPredictor {
            layers,
            w_final: Array1::from_shape_fn(hidden_size, |_| rng.random_range(-0.1..0.1)),
            b_final: 0.0,
            feature_dim,
            hidden_size,
            num_layers,
            dropout_rate,
        })
    }

    pub fn predict(&self, sequence: &Array2<f32>) -> Result<f32, TrainingError> {
        let mut hidden = vec![Array1::zeros(self.hidden_size); self.num_layers];
        let mut cell = vec![Array1::zeros(self.hidden_size); self.num_layers];
        for t in 0..sequence.dim().0 {
            let input = sequence.slice(s![t, ..]).to_owned();
            let mut layer_input = input;
            for (layer, (h, c)) in self.layers.iter().zip(hidden.iter_mut().zip(cell.iter_mut())) {
                let i = sigmoid(&(layer.w_ii.dot(&layer_input) + layer.w_hi.dot(h) + layer.b_i.clone()));
                let f = sigmoid(&(layer.w_if.dot(&layer_input) + layer.w_hf.dot(h) + layer.b_f.clone()));
                let g = tanh(&(layer.w_ig.dot(&layer_input) + layer.w_hg.dot(h) + layer.b_g.clone()));
                let o = sigmoid(&(layer.w_io.dot(&layer_input) + layer.w_ho.dot(h) + layer.b_o.clone()));
                *c = f * c.clone() + &i * &g;
                *h = &o * tanh(c);
                layer_input = h.clone();
            }
        }
        Ok(hidden.last().unwrap().dot(&self.w_final) + self.b_final)
    }

    pub fn train_step(
        &mut self,
        sequences: &[Array2<f32>],
        targets: &[f32],
        optimizer: &mut AdamOptimizer,
        l2_reg: f32,
        grad_clip: f32,
    ) -> Result<f32, TrainingError> {
        let mut loss = 0.0;
        let mut grad_w_final = Array1::zeros(self.w_final.len());
        let grad_b_final = 0.0;
        let mut grad_layers: Vec<LstmLayerWeights> = self.layers.iter()
            .map(|layer| LstmLayerWeights {
                w_ii: Array2::zeros(layer.w_ii.dim()),
                w_if: Array2::zeros(layer.w_if.dim()),
                w_ig: Array2::zeros(layer.w_ig.dim()),
                w_io: Array2::zeros(layer.w_io.dim()),
                w_hi: Array2::zeros(layer.w_hi.dim()),
                w_hf: Array2::zeros(layer.w_hf.dim()),
                w_hg: Array2::zeros(layer.w_hg.dim()),
                w_ho: Array2::zeros(layer.w_ho.dim()),
                b_i: Array1::zeros(layer.b_i.len()),
                b_f: Array1::zeros(layer.b_f.len()),
                b_g: Array1::zeros(layer.b_g.len()),
                b_o: Array1::zeros(layer.b_o.len()),
            })
            .collect();

        for (seq, &target) in sequences.iter().zip(targets.iter()) {
            let mut hidden = vec![Array1::zeros(self.hidden_size); self.num_layers];
            let mut cell = vec![Array1::zeros(self.hidden_size); self.num_layers];
            let mut inputs = vec![vec![Array1::zeros(self.feature_dim); seq.dim().0]; self.num_layers];
            let mut i_gates = vec![vec![Array1::zeros(self.hidden_size); seq.dim().0]; self.num_layers];
            let mut f_gates = vec![vec![Array1::zeros(self.hidden_size); seq.dim().0]; self.num_layers];
            let mut g_gates = vec![vec![Array1::zeros(self.hidden_size); seq.dim().0]; self.num_layers];
            let mut o_gates = vec![vec![Array1::zeros(self.hidden_size); seq.dim().0]; self.num_layers];

            // Forward pass
            for t in 0..seq.dim().0 {
                let input = seq.slice(s![t, ..]).to_owned();
                let mut layer_input = input.clone();
                for l in 0..self.num_layers {
                    inputs[l][t] = layer_input.clone();
                    let layer = &self.layers[l];
                    i_gates[l][t] = sigmoid(&(layer.w_ii.dot(&layer_input) + layer.w_hi.dot(&hidden[l]) + layer.b_i.clone()));
                    f_gates[l][t] = sigmoid(&(layer.w_if.dot(&layer_input) + layer.w_hf.dot(&hidden[l]) + layer.b_f.clone()));
                    g_gates[l][t] = tanh(&(layer.w_ig.dot(&layer_input) + layer.w_hg.dot(&hidden[l]) + layer.b_g.clone()));
                    o_gates[l][t] = sigmoid(&(layer.w_io.dot(&layer_input) + layer.w_ho.dot(&hidden[l]) + layer.b_o.clone()));
                    cell[l] = f_gates[l][t].clone() * cell[l].clone() + &i_gates[l][t] * &g_gates[l][t];
                    hidden[l] = &o_gates[l][t] * tanh(&cell[l]);
                    layer_input = hidden[l].clone();
                }
            }
            let pred = hidden.last().unwrap().dot(&self.w_final) + self.b_final;
            loss += (pred - target).powi(2);

            // Backward pass
            let delta = 2.0 * (pred - target);
            let mut delta_h = delta * &self.w_final.clone();
            for l in (0..self.num_layers).rev() {
                for t in (0..seq.dim().0).rev() {
                    let h_prev = if t == 0 { &Array1::zeros(self.hidden_size) } else { &hidden[l] };
                    let c_prev = if t == 0 { &Array1::zeros(self.hidden_size) } else { &cell[l] };
                    let c_tanh = tanh(&cell[l]);
                    let delta_o = delta_h.clone() * &o_gates[l][t] * (1.0 - &o_gates[l][t]);
                    let delta_c = delta_h.clone() * &o_gates[l][t] * (1.0 - c_tanh.mapv(|x| x.powi(2)));
                    let delta_i = delta_c.clone() * &g_gates[l][t] * (1.0 - &i_gates[l][t]);
                    let delta_f = delta_c.clone() * c_prev * (1.0 - &f_gates[l][t]);
                    let delta_g = delta_c.clone() * &i_gates[l][t] * (1.0 - g_gates[l][t].mapv(|x| x.powi(2)));

                    grad_layers[l].w_ii += &delta_i.clone().insert_axis(Axis(1)).dot(&inputs[l][t].clone().insert_axis(Axis(0)));
                    grad_layers[l].w_if += &delta_f.clone().insert_axis(Axis(1)).dot(&inputs[l][t].clone().insert_axis(Axis(0)));
                    grad_layers[l].w_ig += &delta_g.clone().insert_axis(Axis(1)).dot(&inputs[l][t].clone().insert_axis(Axis(0)));
                    grad_layers[l].w_io += &delta_o.clone().insert_axis(Axis(1)).dot(&inputs[l][t].clone().insert_axis(Axis(0)));
                    grad_layers[l].w_hi += &delta_i.clone().insert_axis(Axis(1)).dot(&h_prev.clone().insert_axis(Axis(0)));
                    grad_layers[l].w_hf += &delta_f.clone().insert_axis(Axis(1)).dot(&h_prev.clone().insert_axis(Axis(0)));
                    grad_layers[l].w_hg += &delta_g.clone().insert_axis(Axis(1)).dot(&h_prev.clone().insert_axis(Axis(0)));
                    grad_layers[l].w_ho += &delta_o.clone().insert_axis(Axis(1)).dot(&h_prev.clone().insert_axis(Axis(0)));
                    grad_layers[l].b_i += &delta_i;
                    grad_layers[l].b_f += &delta_f;
                    grad_layers[l].b_g += &delta_g;
                    grad_layers[l].b_o += &delta_o;

                    delta_h = &self.layers[l].w_hi.t().dot(&delta_i) + &self.layers[l].w_hf.t().dot(&delta_f) +
                              &self.layers[l].w_hg.t().dot(&delta_g) + &self.layers[l].w_ho.t().dot(&delta_o);
                }
            }
        }

        // Apply L2 regularization and gradient clipping
        for layer in &mut grad_layers {
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_ii, &mut layer.w_ii, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_if, &mut layer.w_if, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_ig, &mut layer.w_ig, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_io, &mut layer.w_io, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_hi, &mut layer.w_hi, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_hf, &mut layer.w_hf, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_hg, &mut layer.w_hg, l2_reg);
            apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.layers[0].w_ho, &mut layer.w_ho, l2_reg);
        }
        let mut grad_w_final_2d = grad_w_final.clone().insert_axis(Axis(1));
        apply_l2_regularization::<ndarray::Dim<[usize; 2]>>(&self.w_final.clone().insert_axis(Axis(1)), &mut grad_w_final_2d, l2_reg);
        clip_gradients_by_norm(&mut grad_w_final, grad_clip);

        // Update weights
        let mut flat_weights = self.flatten_weights()?;
        let flat_grads = self.flatten_grads(&grad_layers, &grad_w_final, grad_b_final)?;
        optimizer.update(&mut flat_weights, &flat_grads)?;
        self.unflatten_weights(&flat_weights)?;

        Ok(loss / sequences.len() as f32)
    }

    pub fn get_weights(&self) -> ModelWeights {
        ModelWeights {
            asset: "".to_string(),
            layers: self.layers.clone(),
            w_final: self.w_final.clone(),
            b_final: self.b_final,
            closing_mean: 0.0,
            closing_std: 1.0,
            seq_length: 0,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn flatten_weights(&self) -> Result<Vec<f32>, TrainingError> {
        let mut weights = Vec::new();
        for layer in &self.layers {
            weights.extend(layer.w_ii.iter());
            weights.extend(layer.w_if.iter());
            weights.extend(layer.w_ig.iter());
            weights.extend(layer.w_io.iter());
            weights.extend(layer.w_hi.iter());
            weights.extend(layer.w_hf.iter());
            weights.extend(layer.w_hg.iter());
            weights.extend(layer.w_ho.iter());
            weights.extend(layer.b_i.iter());
            weights.extend(layer.b_f.iter());
            weights.extend(layer.b_g.iter());
            weights.extend(layer.b_o.iter());
        }
        weights.extend(self.w_final.iter());
        weights.push(self.b_final);
        Ok(weights)
    }

    fn flatten_grads(&self, grad_layers: &[LstmLayerWeights], grad_w_final: &Array1<f32>, grad_b_final: f32) -> Result<Vec<f32>, TrainingError> {
        let mut grads = Vec::new();
        for layer in grad_layers {
            grads.extend(layer.w_ii.iter());
            grads.extend(layer.w_if.iter());
            grads.extend(layer.w_ig.iter());
            grads.extend(layer.w_io.iter());
            grads.extend(layer.w_hi.iter());
            grads.extend(layer.w_hf.iter());
            grads.extend(layer.w_hg.iter());
            grads.extend(layer.w_ho.iter());
            grads.extend(layer.b_i.iter());
            grads.extend(layer.b_f.iter());
            grads.extend(layer.b_g.iter());
            grads.extend(layer.b_o.iter());
        }
        grads.extend(grad_w_final.iter());
        grads.push(grad_b_final);
        Ok(grads)
    }

    fn unflatten_weights(&mut self, flat_weights: &[f32]) -> Result<(), TrainingError> {
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