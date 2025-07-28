
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
use std::path::Path;
use ndarray::{Array1, Array2, s};
use log::{info, error};
use env_logger;
use rand::Rng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    asset: String,
    #[arg(long)]
    source: String,
    #[arg(long, default_value_t = 10)]
    seq_length: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct StockRecord {
    date: String,
    closing: f32,
    opening: f32,
    high: f32,
    low: f32,
    volume: f32,
    variation: f32,
}

#[derive(Serialize)]
struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Vec<Vec<f32>>, // Input to hidden weights
    weight_hh: Vec<Vec<f32>>, // Hidden to hidden weights
    bias: Vec<f32>,           // Bias terms
    weight_out: Vec<f32>,     // Output linear layer weights
    bias_out: f32,            // Output linear layer bias
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> LSTMCell {
        let mut rng = rand::thread_rng(); // Note: thread_rng is deprecated but kept for compatibility; see warnings
        let weight_ih = (0..4 * hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let weight_hh = (0..4 * hidden_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let bias = (0..4 * hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let weight_out = (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let bias_out = rng.gen_range(-0.1..0.1);
        LSTMCell {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias,
            weight_out,
            bias_out,
        }
    }

    fn forward(
        &self,
        x: &Array1<f32>,
        h_prev: &Array1<f32>,
        c_prev: &Array1<f32>,
    ) -> (Array1<f32>, Array1<f32>, f32) {
        let gates = {
            let ih = Array2::from_shape_vec(
                (4 * self.hidden_size, self.input_size),
                self.weight_ih.iter().flatten().cloned().collect(),
            )
            .unwrap();
            let hh = Array2::from_shape_vec(
                (4 * self.hidden_size, self.hidden_size),
                self.weight_hh.iter().flatten().cloned().collect(),
            )
            .unwrap();
            let bias = Array1::from_vec(self.bias.clone());
            &ih.dot(x) + &hh.dot(h_prev) + &bias
        };
        let i = sigmoid(&gates.slice(s![..self.hidden_size]).to_owned());
        let f = sigmoid(&gates.slice(s![self.hidden_size..2 * self.hidden_size]).to_owned());
        let o = sigmoid(&gates.slice(s![2 * self.hidden_size..3 * self.hidden_size]).to_owned());
        let g = tanh(&gates.slice(s![3 * self.hidden_size..]).to_owned());

        let c = &f * c_prev + &i * &g;
        let h = &o * tanh(&c);

        // Linear output layer
        let output = h.dot(&Array1::from_vec(self.weight_out.clone())) + self.bias_out;

        (h.to_owned(), c.to_owned(), output)
    }

    fn update_weights(
        &mut self,
        x: &Array1<f32>,
        h_prev: &Array1<f32>,
        c_prev: &Array1<f32>,
        target: f32,
        learning_rate: f32,
    ) -> f32 {
        let (h, c, pred) = self.forward(x, h_prev, c_prev);
        let loss = (pred - target).powi(2);

        // Gradient of loss w.r.t. prediction
        let d_loss_d_pred = 2.0 * (pred - target);

        // Gradients for output layer
        let d_pred_d_wo = h.clone();
        let d_pred_d_bo = 1.0;
        let d_pred_d_h = Array1::from_vec(self.weight_out.clone());

        // Update output weights and bias
        for (wo, &d_wo) in self.weight_out.iter_mut().zip(d_pred_d_wo.iter()) {
            *wo -= learning_rate * d_loss_d_pred * d_wo;
        }
        self.bias_out -= learning_rate * d_loss_d_pred * d_pred_d_bo;

        // Simplified gradients for LSTM weights
        let ih = Array2::from_shape_vec(
            (4 * self.hidden_size, self.input_size),
            self.weight_ih.iter().flatten().cloned().collect(),
        )
        .unwrap();
        let hh = Array2::from_shape_vec(
            (4 * self.hidden_size, self.hidden_size),
            self.weight_hh.iter().flatten().cloned().collect(),
        )
        .unwrap();
        let d_h = d_loss_d_pred * &d_pred_d_h;
        let d_weight_ih = d_h
            .to_shape((self.hidden_size, 1))
            .unwrap()
            .dot(&x.to_shape((1, self.input_size)).unwrap());
        let d_weight_hh = d_h
            .to_shape((self.hidden_size, 1))
            .unwrap()
            .dot(&h_prev.to_shape((1, self.hidden_size)).unwrap());
        let d_bias = d_h.clone();

        // Update LSTM weights
        for (wi, d_wi) in self.weight_ih.iter_mut().zip(d_weight_ih.rows()) {
            for (w, &d_w) in wi.iter_mut().zip(d_wi.iter()) {
                *w -= learning_rate * d_w;
            }
        }
        for (wh, d_wh) in self.weight_hh.iter_mut().zip(d_weight_hh.rows()) {
            for (w, &d_w) in wh.iter_mut().zip(d_wh.iter()) {
                *w -= learning_rate * d_w;
            }
        }
        for (b, &d_b) in self.bias.iter_mut().zip(d_bias.iter()) {
            *b -= learning_rate * d_b;
        }

        loss
    }
}

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.tanh())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let input_file_path = format!("../dados/{}_{}_output.json", args.asset, args.source);
    let output_model_path = format!("../dados/{}_{}_lstm_model.json", args.asset, args.source);

    let input_path = Path::new(&input_file_path);
    if !input_path.exists() {
        error!("Input file not found: {}", input_file_path);
        return Ok(());
    }

    info!("Loading JSON data from {}", input_file_path);
    let json_data = fs::read_to_string(&input_file_path)?;
    let records: Vec<StockRecord> = serde_json::from_str(&json_data)?;

    if records.len() <= args.seq_length {
        error!("Not enough records ({}) for sequence length {}", records.len(), args.seq_length);
        return Ok(());
    }

    // Extract closing prices
    let prices: Array1<f32> = Array1::from_vec(records.iter().map(|r| r.closing).collect());
    info!("Loaded {} price records", prices.len());

    // Normalize data
    let mean = prices.mean().unwrap_or(0.0);
    let std = prices.std_axis(ndarray::Axis(0), 0.0).into_scalar();
    let normalized_prices = prices.mapv(|x| (x - mean) / std);

    // Prepare sequences
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    for i in 0..(normalized_prices.len() - args.seq_length) {
        let seq = normalized_prices.slice(s![i..i + args.seq_length]).to_vec();
        let target = normalized_prices[i + args.seq_length];
        sequences.push(seq);
        targets.push(target);
    }

    let hidden_size = 10;
    let mut lstm = LSTMCell::new(1, hidden_size);

    // Training loop
    let learning_rate = 0.01;
    for epoch in 0..50 {
        let mut total_loss = 0.0;
        for i in 0..sequences.len() {
            let mut h_t = Array1::zeros(hidden_size);
            let mut c_t = Array1::zeros(hidden_size);

            // Process sequence
            for &input_val in &sequences[i] {
                let x = Array1::from_vec(vec![input_val]);
                let (h_next, c_next, _output) = lstm.forward(&x, &h_t, &c_t);
                h_t = h_next;
                c_t = c_next;
            }

            // Update weights using the last input and target
            let last_x = Array1::from_vec(vec![sequences[i].last().unwrap().clone()]);
            let loss = lstm.update_weights(&last_x, &h_t, &c_t, targets[i], learning_rate);
            total_loss += loss;
        }
        info!("Epoch: {}, Loss: {:.4}", epoch, total_loss / sequences.len() as f32);
        println!("Epoch: {}, Loss: {:.4}", epoch, total_loss / sequences.len() as f32);
    }

    // Save model
    let model_state = serde_json::to_value(&lstm)?;
    fs::write(&output_model_path, serde_json::to_string_pretty(&model_state)?)?;
    info!("Model saved to: {}", output_model_path);

    Ok(())
}



// cargo run -- --asset WEGE3 --source investing




// cd lstmfiletrain

// rm -rf target Cargo.lock
// rm -rf ~/.cargo/registry/cache/*
