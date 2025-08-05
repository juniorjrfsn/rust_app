// projeto: lstmfilepredict
// file: src/rna/model.rs



pub mod model {
    use serde::{Deserialize, Serialize};
    use ndarray::{Array1, Array2};
    use crate::LSTMError;
    use log;

    #[derive(Debug, Deserialize, Serialize, Clone)]
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

    #[derive(Debug, Deserialize, Serialize, Clone)]
    pub struct ModelWeights {
        pub asset: String,
        pub source: String,
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
    }

    #[derive(Clone)]
    pub struct LSTMCell {
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
        pub fn from_weights(weights: &LSTMLayerWeights) -> Self {
            LSTMCell {
                w_input: weights.w_input.clone(),
                u_input: weights.u_input.clone(),
                b_input: weights.b_input.clone(),
                w_forget: weights.w_forget.clone(),
                u_forget: weights.u_forget.clone(),
                b_forget: weights.b_forget.clone(),
                w_output: weights.w_output.clone(),
                u_output: weights.u_output.clone(),
                b_output: weights.b_output.clone(),
                w_cell: weights.w_cell.clone(),
                u_cell: weights.u_cell.clone(),
                b_cell: weights.b_cell.clone(),
            }
        }

        fn sigmoid(x: f32) -> f32 {
            let x_clamped = x.max(-500.0).min(500.0);
            1.0 / (1.0 + (-x_clamped).exp())
        }

        fn tanh(x: f32) -> f32 {
            let x_clamped = x.max(-20.0).min(20.0);
            x_clamped.tanh()
        }

        pub fn forward(&self, input: &Array1<f32>, h_prev: &Array1<f32>, c_prev: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
            let i_t = (&self.w_input.dot(input) + &self.u_input.dot(h_prev) + &self.b_input).mapv(Self::sigmoid);
            let f_t = (&self.w_forget.dot(input) + &self.u_forget.dot(h_prev) + &self.b_forget).mapv(Self::sigmoid);
            let o_t = (&self.w_output.dot(input) + &self.u_output.dot(h_prev) + &self.b_output).mapv(Self::sigmoid);
            let c_hat_t = (&self.w_cell.dot(input) + &self.u_cell.dot(h_prev) + &self.b_cell).mapv(Self::tanh);
            let c_t = &f_t * c_prev + &i_t * &c_hat_t;
            let h_t = &o_t * &c_t.mapv(Self::tanh);
            (h_t, c_t)
        }
    }

    #[derive(Clone)]
    pub struct MultiLayerLSTM {
        layers: Vec<LSTMCell>,
        w_final: Array1<f32>,
        b_final: f32,
        seq_length: usize,
        hidden_size: usize,
        num_layers: usize,
    }

    impl MultiLayerLSTM {
        pub fn from_weights(weights: &ModelWeights) -> Self {
            log::info!("  🏗️ Building MultiLayerLSTM model from weights...");
            let layers: Vec<LSTMCell> = weights.layers.iter().map(LSTMCell::from_weights).collect();
            log::info!("  ✅ Model built with {} layers", layers.len());
            MultiLayerLSTM {
                layers,
                w_final: weights.w_final.clone(),
                b_final: weights.b_final,
                seq_length: weights.seq_length,
                hidden_size: weights.hidden_size,
                num_layers: weights.num_layers,
            }
        }

        pub fn forward(&self, sequence: &[f32], h_states: &mut [Array1<f32>], c_states: &mut [Array1<f32>]) -> Result<f32, LSTMError> {
            if sequence.len() % 4 != 0 {
                return Err(LSTMError::ForwardError(format!("Invalid sequence length: {}. Expected a multiple of 4.", sequence.len())));
            }
            let seq_timesteps = sequence.len() / 4;
            if seq_timesteps != self.seq_length {
                return Err(LSTMError::ForwardError(format!("Sequence length mismatch: got {} timesteps, expected {}.", seq_timesteps, self.seq_length)));
            }

            for i in 0..seq_timesteps {
                let start_idx = i * 4;
                let input_vec = &sequence[start_idx..start_idx + 4];
                let input = Array1::from_vec(input_vec.to_vec());
                let mut layer_input = input;

                for (j, layer) in self.layers.iter().enumerate() {
                    let (h_new, c_new) = layer.forward(&layer_input, &h_states[j], &c_states[j]);
                    h_states[j] = h_new.clone();
                    c_states[j] = c_new;
                    layer_input = h_new;
                }
            }

            let output = self.w_final.dot(&h_states[self.num_layers - 1]) + self.b_final;
            if output.is_nan() || output.is_infinite() {
                Err(LSTMError::ForwardError("Invalid output value (NaN/Inf)".to_string()))
            } else {
                Ok(output)
            }
        }

        pub fn predict_future(&self, initial_sequence: Vec<f32>, num_predictions: usize, closing_mean: f32, closing_std: f32) -> Result<Vec<f32>, LSTMError> {
            let mut predictions = Vec::new();
            let mut current_sequence = initial_sequence.clone();
            let mut h_states = vec![Array1::zeros(self.hidden_size); self.num_layers];
            let mut c_states = vec![Array1::zeros(self.hidden_size); self.num_layers];

            self.forward(&current_sequence, &mut h_states, &mut c_states)?;

            for _ in 0..num_predictions {
                let last_input = Array1::from_vec(vec![
                    current_sequence[current_sequence.len() - 4],
                    current_sequence[current_sequence.len() - 3],
                    current_sequence[current_sequence.len() - 2],
                    current_sequence[current_sequence.len() - 1],
                ]);
                let mut layer_input = last_input.clone();

                for (j, layer) in self.layers.iter().enumerate() {
                    let (h_new, c_new) = layer.forward(&layer_input, &h_states[j], &c_states[j]);
                    h_states[j] = h_new.clone();
                    c_states[j] = c_new;
                    layer_input = h_new;
                }

                let pred_normalized = self.w_final.dot(&h_states[self.num_layers - 1]) + self.b_final;
                let pred_denorm = pred_normalized * closing_std + closing_mean;
                predictions.push(pred_denorm);

                let pred_normalized_shifted = pred_normalized;
                current_sequence.push(pred_normalized_shifted);
                if current_sequence.len() > self.seq_length * 4 {
                    current_sequence.drain(0..4);
                }
            }

            Ok(predictions)
        }
    }
}